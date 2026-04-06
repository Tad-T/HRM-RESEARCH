from typing import Optional
import os
import csv
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from common import PuzzleDatasetMetadata


cli = ArgParser()

class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/sudoku-extreme"
    # New: Path to a simpler dataset or a flag to generate easy ones
    easy_repo: str = "steven-terner/sudoku" 
    output_dir: str = "data/sudoku-curriculum"

    easy_ratio: float = 0.5  # 50% easy puzzles, 50% extreme
    subsample_size: Optional[int] = 2000 # Smaller, high-quality set
    num_aug: int = 2 # Increase augmentation to help logic generalization


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    
    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


import random
def convert_subset(set_name: str, config: DataProcessConfig):
    # 1. Initialize storage for both types
    extreme_inputs, extreme_labels = [], []
    easy_inputs, easy_labels = [], []
    
    # --- PART A: LOAD EXTREME DATA (Goal) ---
    print(f">>> Fetching Extreme samples for {set_name}...")
    extreme_path = hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset")
    with open(extreme_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            # sapientinc/sudoku-extreme format: source, puzzle, solution, rating
            q, a = row[1], row[2]
            extreme_inputs.append(np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
            extreme_labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))

    # --- PART B: LOAD EASY DATA (Anchor) ---
    # We use a standard dataset like 'steven-terner/sudoku' or similar for 35+ clue puzzles
    print(f">>> Fetching Easy/Medium anchors for {set_name}...")
    try:
        # Note: Adjust filename if the easy repo uses different naming (e.g., 'sudoku.csv')
        easy_path = hf_hub_download("steven-terner/sudoku", "sudoku.csv", repo_type="dataset")
        with open(easy_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            count = 0
            for row in reader:
                q, a = row[0], row[1] # standard format is usually q, a
                easy_inputs.append(np.frombuffer(q.encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
                easy_labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
                count += 1
                if count > len(extreme_inputs): break # Keep pools somewhat balanced
    except Exception as e:
        print(f"⚠️ Could not load easy repo, falling back to extreme only: {e}")
        easy_inputs, easy_labels = extreme_inputs, extreme_labels

    # --- PART C: CURRICULUM MIXING ---
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    # Determine total iterations
    total_to_process = config.subsample_size if (set_name == "train" and config.subsample_size) else len(extreme_inputs)
    
    print(f">>> Mixing {total_to_process} samples with {config.easy_ratio*100}% Easy/Extreme ratio...")

    for i in tqdm(range(total_to_process)):
        # Determine if this slot gets an Easy or Extreme puzzle
        if random.random() < config.easy_ratio and len(easy_inputs) > 0:
            idx = random.randrange(len(easy_inputs))
            orig_inp, orig_out = easy_inputs[idx], easy_labels[idx]
        else:
            idx = random.randrange(len(extreme_inputs))
            orig_inp, orig_out = extreme_inputs[idx], extreme_labels[idx]

        # Apply Augmentations (Rotation/Permutation)
        num_augments = config.num_aug if set_name == "train" else 0
        for aug_idx in range(1 + num_augments):
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)

            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
            
        results["group_indices"].append(puzzle_id)

    # --- PART D: SAVE AS NUMPY (Same as original) ---
    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        assert np.all((arr >= 0) & (arr <= 9))
        return arr + 1
    
    results_final = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=81, vocab_size=11, pad_id=0, ignore_label_id=0,
        blank_identifier_id=0, num_puzzle_identifiers=1,
        total_groups=len(results_final["group_indices"]) - 1,
        mean_puzzle_examples=1, sets=["all"]
    )
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    for k, v in results_final.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
