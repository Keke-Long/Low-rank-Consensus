"""
Recompute error columns for all scene CSV files in Nuscenes/results_caption.

New columns:
- single_agent_Error_r=1/2/3
- multi_agent_Error_r=1/2/3/4

Rules:
- Keep row order unchanged.
- Keep Timestamp/Model/Caption/hallucination_check values unchanged.
- Remove legacy residual columns before writing new results.
"""

from __future__ import annotations

import csv
import gc
import os
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD

INPUT_FOLDER = "./Nuscenes/results_caption/"
SINGLE_RANKS = [1, 2, 3]
MULTI_RANKS = [1, 2, 3, 4]

MODEL_ORDER = [
    "deepseek-vl2-tiny",
    "Qwen2-VL-7B-Instruct",
    "llava-1.5-7b-hf",
]

MULTI_AGENT_SAMPLE_COUNTS = {
    "deepseek-vl2-tiny": 4,
    "Qwen2-VL-7B-Instruct": 3,
    "llava-1.5-7b-hf": 3,
}

IMMUTABLE_COLUMNS = ["Timestamp", "Model", "Caption", "hallucination_check"]

LEGACY_COLUMNS = {
    "ResidualError_r=3",
    "ResidualError_r=5",
    "ResidualError_r=7",
    "ResidualError_multi_agent_r=1",
    "ResidualError_multi_agent_r=2",
    "ResidualError_multi_agent_r=3",
    "ResidualError_multi_agent_r=4",
    "ResidualError_multi_agent_r=5",
    "ResidualError_multi_agent_r=7",
}

NEW_COLUMNS = [
    "single_agent_Error_r=1",
    "single_agent_Error_r=2",
    "single_agent_Error_r=3",
    "multi_agent_Error_r=1",
    "multi_agent_Error_r=2",
    "multi_agent_Error_r=3",
    "multi_agent_Error_r=4",
]


def compute_residual_norms(embeddings: np.ndarray, rank: int) -> np.ndarray:
    """Compute per-row residual L2 norms under rank-r SVD reconstruction."""
    max_rank = min(embeddings.shape[0] - 1, embeddings.shape[1])
    if max_rank < 1 or rank > max_rank:
        return np.full((embeddings.shape[0],), np.nan, dtype=float)

    svd = TruncatedSVD(n_components=rank, random_state=42)
    low_rank = svd.fit_transform(embeddings)
    reconstructed = svd.inverse_transform(low_rank)
    residuals = embeddings - reconstructed
    return np.linalg.norm(residuals, axis=1)


def encode_captions(model: SentenceTransformer, captions: List[str]) -> np.ndarray:
    return model.encode(captions, convert_to_numpy=True)


def clean_and_prepare_fieldnames(original_fieldnames: List[str]) -> List[str]:
    kept = [
        c
        for c in original_fieldnames
        if c not in LEGACY_COLUMNS and c not in NEW_COLUMNS and c != "temperature"
    ]

    for c in NEW_COLUMNS:
        if c not in kept:
            kept.append(c)
    return kept


def process_one_csv(csv_path: str, model: SentenceTransformer) -> None:
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    # Remove legacy/new cols in-memory; keep immutable cols unchanged.
    for row in rows:
        for c in list(row.keys()):
            if c in LEGACY_COLUMNS or c in NEW_COLUMNS or c == "temperature":
                row.pop(c, None)
        for c in NEW_COLUMNS:
            row[c] = ""

    # Single-agent residuals per model over all rows of that model.
    for model_name in MODEL_ORDER:
        idx = [i for i, r in enumerate(rows) if r.get("Model", "") == model_name]
        if not idx:
            continue
        captions = [rows[i].get("Caption", "") for i in idx]
        embeddings = encode_captions(model, captions)

        for rank in SINGLE_RANKS:
            vals = compute_residual_norms(embeddings, rank)
            col = f"single_agent_Error_r={rank}"
            for j, row_idx in enumerate(idx):
                if not np.isnan(vals[j]):
                    rows[row_idx][col] = f"{vals[j]:.4f}"

    # Multi-agent residuals over fixed 4+3+3 samples by original row order.
    selected_idx: List[int] = []
    for model_name in MODEL_ORDER:
        k = MULTI_AGENT_SAMPLE_COUNTS[model_name]
        idx = [i for i, r in enumerate(rows) if r.get("Model", "") == model_name][:k]
        selected_idx.extend(idx)

    if selected_idx:
        captions = [rows[i].get("Caption", "") for i in selected_idx]
        embeddings = encode_captions(model, captions)

        for rank in MULTI_RANKS:
            vals = compute_residual_norms(embeddings, rank)
            col = f"multi_agent_Error_r={rank}"
            for j, row_idx in enumerate(selected_idx):
                if not np.isnan(vals[j]):
                    rows[row_idx][col] = f"{vals[j]:.4f}"

    # Write back with updated field order.
    fieldnames = clean_and_prepare_fieldnames(original_fieldnames)
    with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    csv_files = sorted(f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv"))

    print("Recomputing new error columns for all files...")
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Files: {len(csv_files)}")

    st_model = SentenceTransformer("intfloat/e5-large-v2", device="cpu", local_files_only=True)
    st_model.to("cuda")

    for i, csv_file in enumerate(csv_files, start=1):
        csv_path = os.path.join(INPUT_FOLDER, csv_file)
        process_one_csv(csv_path, st_model)
        if i % 25 == 0 or i == len(csv_files):
            print(f"Processed {i}/{len(csv_files)}")

    torch.cuda.empty_cache()
    gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()
