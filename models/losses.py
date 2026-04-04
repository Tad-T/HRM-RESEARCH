from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x < 0,
        1 / (1 - x + epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


# Added **kwargs to absorb 'reduction' and other arguments gracefully
def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, **kwargs):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    # Returns the loss per token (effectively reduction="none")
    return -torch.where(valid_mask, prediction_logprobs.to(logits.dtype), 0.0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100, reduction: str = "none"):
    # Reshape to (Batch, Seq) to match stablemax behavior
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]), 
        labels.to(torch.long).view(-1), 
        ignore_index=ignore_index, 
        reduction=reduction
    ).view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        
        # --- PREPARE DATA ---
        inputs = model_kwargs["batch"]["inputs"]
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # IDENTIFY TRUE BLANKS (Value 1)
        active_loss_mask = (inputs == 1) 

        with torch.no_grad():
            full_mask = labels != IGNORE_LABEL_ID
            active_counts = active_loss_mask.sum(-1).clamp_min(1)
            loss_divisor = active_counts.unsqueeze(-1)

            is_correct = full_mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            
            # REAL ACCURACY (Blanks only)
            active_is_correct = is_correct & active_loss_mask
            active_acc = (active_is_correct.to(torch.float32).sum(-1) / active_counts)
            
            seq_is_correct = active_is_correct.sum(-1) == active_loss_mask.sum(-1)
            
            valid_metrics = new_carry.halted & (full_mask.sum(-1) > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "active_accuracy": torch.where(valid_metrics, active_acc, 0.0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps.to(torch.float32), 0.0).sum(),
            }

        # --- MASKED LM LOSS ---
        # Pass reduction='none' - stablemax now accepts it via **kwargs
        raw_lm_loss = self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, reduction='none')
        
        # Mask the loss: zero out the hints and clues
        lm_loss = ((raw_lm_loss * active_loss_mask.float()) / loss_divisor).sum()

        # Q-Halt uses the reasoning-based success
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"], 
            seq_is_correct.to(outputs["q_halt_logits"].dtype), 
            reduction="sum"
        )

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"], 
                (~seq_is_correct).to(outputs["q_continue_logits"].dtype), 
                reduction="sum"
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()