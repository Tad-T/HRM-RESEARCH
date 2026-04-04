from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
from coolname.impl import generate_slug
import hydra
import pydantic
from omegaconf import DictConfig

from models.layers import LoRALinear, CastedLinear
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed

from torch.optim import Optimizer
from typing import Optional, Callable

class AdamATan2(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Any:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                # We cast to float to satisfy the IDE's return type requirement
                loss = float(closure())

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Use Atan2 for stable updates in reasoning loops
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                update = torch.atan2(exp_avg, denom)

                p.add_(update, alpha=-group['lr'])

        return loss

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_path: str

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    pretrained_weight_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []
    limit_eval_batches: int = -1  # For quick eval runs, set a positive number to limit eval batches


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,

        dataset_path=config.data_path,

        rank=rank,
        num_replicas=world_size,
        
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,

        num_workers=1,
        prefetch_factor=8,

        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata

def apply_lora_to_reasoning(module: nn.Module, r: int = 32, alpha: int = 16):
    """
    Recursively finds all CastedLinear layers and wraps them in LoRA.
    """
    for name, child in module.named_children():
        if isinstance(child, CastedLinear):
            # 1. Create the LoRA wrapper around the existing CastedLinear
            # We use alpha=r*2 as a common heuristic for stability
            lora_layer = LoRALinear(child, r=r, alpha=alpha)
            
            # 2. Replace the original attribute with the LoRA version
            setattr(module, name, lora_layer)
            
        else:
            # 3. If it's not a linear layer, dive deeper (into Blocks, etc.)
            apply_lora_to_reasoning(child, r, alpha)

def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore

        batch_size=config.global_batch_size // world_size,

        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        # Always start with the raw architecture
        base_model: nn.Module = model_cls(model_cfg)

        # CASE A: FINETUNING (Load weights + Inject LoRA)
        if config.pretrained_weight_path and os.path.exists(config.pretrained_weight_path):
            print(f">>> LORA MODE: Loading weights from {config.pretrained_weight_path}")
            
            # 1. Load the frozen knowledge
            sd = torch.load(config.pretrained_weight_path, map_location="cuda")

            # 2. Strip the 'model.' prefix from every key
            new_sd = {}
            for k, v in sd.items():
                if k.startswith("model."):
                    new_sd[k[6:]] = v  # [6:] removes 'model.'
                else:
                    new_sd[k] = v

            # 3. Load into the base model BEFORE applying LoRA
            print(f">>> Loading {len(new_sd)} cleaned keys into base_model...")
            base_model.load_state_dict(new_sd, strict=True)
            
            # 2. Add the trainable "side-cars"
            apply_lora_to_reasoning(base_model.inner, r=8)
            loss_extra = config.arch.loss.model_dump(exclude={'name'})
            model: nn.Module = loss_head_cls(base_model, **loss_extra)

            # 3. Lock the base, unlock the adapters
            for param in model.parameters():
                param.requires_grad = False
            for name, param in model.named_parameters():
                if "lora_" in name or "q_head" in name:
                    param.requires_grad = True

        # CASE B: PRETRAINING (From Scratch)
        else:
            print(">>> PRETRAIN MODE: Fresh weights, full gradients.")
            loss_extra = config.arch.loss.model_dump(exclude={'name'})
            model = loss_head_cls(base_model, **loss_extra)
            # All params stay requires_grad = True (default)
            
            # Ensure everything is trainable
            for param in model.parameters():
                param.requires_grad = True

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f">>> [Model Setup] Trainable: {trainable_params:,} | Total: {total_params:,} ({100 * trainable_params/total_params:.2f}%)")

        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)  # type: ignore

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    optimizers = [
        CastedSparseEmbeddingSignSGD_Distributed(
            model.model.puzzle_emb.buffers(),  # type: ignore
            
            lr=0,  # Needs to be set by scheduler
            weight_decay=config.puzzle_emb_weight_decay,

            world_size=world_size
        ),
        AdamATan2(
            #model.parameters(),
            [p for p in model.parameters() if p.requires_grad],

            lr=config.lr,  # Needs to be set by scheduler
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
    ]
    optimizer_lrs = [
        config.puzzle_emb_lr,
        config.lr
    ]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return

    # Ensure all processes reach this point
    if dist.is_initialized():
        dist.barrier()

    # ONLY Rank 0 saves
    if dist.get_rank() == 0:
        print("RANK 0: Saving checkpoint...")
        os.makedirs(config.checkpoint_path, exist_ok=True)
        save_path = os.path.join(config.checkpoint_path, f"step_{train_state.step}.pt")
        
        # Save a temporary file then move it to ensure atomicity
        torch.save(train_state.model.state_dict(), save_path)
        print(f'Rank 0: Successfully saved checkpoint to {save_path}')

    if dist.is_initialized():
        dist.barrier()


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: DataLoader, eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    with torch.inference_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        all_preds = {} # For eval_save_outputs
        metric_keys = []
        metric_values = None

        for batch_i, (set_name, batch, global_batch_size) in enumerate(eval_loader):
            if config.limit_eval_batches > 0 and batch_i >= config.limit_eval_batches:
                break
            # Prepare Batch
            batch = {k: v.cuda() for k, v in batch.items()}
            
            with torch.device("cuda"):
                # Create the starting state
                initial_carry = train_state.model.initial_carry(batch)

            # Baseline check on a "dummy" copy
            import copy
            _, step1_loss, _, _, _ = train_state.model(
                carry=copy.copy(initial_carry), batch=batch, return_keys=[]
            )
            step1_loss = step1_loss.detach() # DETACH HERE to save VRAM

            steps_taken = 0
            current_carry = initial_carry 
            while True:
                # Force logits to exist
                keys = list(set(config.eval_save_outputs) | {"logits"})
                current_carry, final_loss, metrics, preds, all_finish = train_state.model(
                    carry=current_carry, batch=batch, return_keys=keys
                )
                steps_taken += 1
                if all_finish or steps_taken >= 15:
                    break

            # Scoped and Detached Metrics
            final_logits = preds['logits']
            targets = batch['labels']
            
            # Use .detach() on everything going into the metrics dict
            is_correct = (final_logits.argmax(-1) == targets).all(dim=-1).float().detach()
            probs = torch.softmax(final_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().detach()

            metrics["solve_rate"] = is_correct.mean()
            metrics["final_entropy"] = entropy
            metrics["steps_to_halt"] = torch.tensor(float(steps_taken), device="cuda").detach()
            metrics["reasoning_gain"] = (step1_loss - final_loss).detach()
            metrics["count"] = torch.tensor(float(batch['labels'].size(0)), device="cuda").detach()

            # Handle eval_save_outputs
            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        all_preds.setdefault(k, [])
                        all_preds[k].append(v.cpu())

            # Aggregate Results
            set_id = set_ids[set_name]
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros((len(set_ids), len(metric_keys)), dtype=torch.float32, device="cuda")
                
            metric_values[set_id] += torch.stack([metrics[k].detach() for k in metric_keys])

            del initial_carry, current_carry, preds, batch

        # --- MANDATORY DISTRIBUTED SYNC ---
        if world_size > 1:
            # 1. Barrier ensures everyone has exited the loop
            dist.barrier()
            
            # 2. Safety: Sync the number of metrics so "empty" ranks can participate
            num_metrics = torch.tensor(len(metric_keys) if metric_keys else 0, device="cuda")
            dist.all_reduce(num_metrics, op=dist.ReduceOp.MAX)
            
            # 3. Create zero-filled tensor for ranks that didn't process data
            if metric_values is None:
                metric_values = torch.zeros((len(set_ids), (int)(num_metrics.item())), device="cuda")
            
            # 4. Global reduction to Rank 0
            dist.reduce(metric_values, dst=0)

        # --- RANK 0 REPORTING ---
        if rank == 0 and metric_values is not None:
            reduced_data = metric_values.cpu().numpy()
            final_report = {}
            
            for set_id, set_name in enumerate(set_ids):
                if not metric_keys: break
                
                res = {metric_keys[i]: reduced_data[set_id, i] for i in range(len(metric_keys))}
                
                # Check for "count" to perform averaging
                if "count" in res and res["count"] > 0:
                    count = res.pop("count")
                    final_report[set_name] = {k: v / count for k, v in res.items()}
                else:
                    final_report[set_name] = {k: v for k, v in res.items()}
            
            return final_report
            
    return None


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_path).capitalize()} ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("/kaggle/working/HRM-RESEARCH/checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    # Train state
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)

        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    # Training Loop
    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore

        ############ Evaluation
        print("=== TRAINING LOOP DONE ===")
        train_state.model.eval()
        print("Starting evaluation...")

        metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)
        print("Evaluation done")

        if RANK == 0 and metrics is not None:
            wandb.log(metrics, step=train_state.step)
            print("Logged evaluation metrics to WandB")

        if (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
            save_train_state(config, train_state)

        print("=== TRAINING CELL COMPLETE ===")

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()