from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD


ENCODER_ID = "intfloat/e5-large-v2"
LOCAL_ENCODER_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--intfloat--e5-large-v2/snapshots/"
    "f169b11e22de13617baa190a028a32f3493550b6"
)

INPUT_DIR_DEFAULT = "./Nuscenes/sensitivity_analysis/large_vlm/results_caption_20260418"
RANK = 1
MODELS = ["deepseek-vl2-small", "llava-1.5-13b-hf", "Qwen2.5-VL-32B-Instruct"]


def parse_hallucination_check(value: object) -> Optional[Tuple[int, int]]:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    hallucinated = 0
    total = 0
    for part in parts:
        if ":" not in part:
            continue
        _, label = part.split(":", 1)
        total += 1
        if label.strip().lower() == "yes":
            hallucinated += 1
    if total == 0:
        return None
    return hallucinated, total


def compute_residuals(embeddings: np.ndarray, rank: int) -> np.ndarray:
    max_possible_rank = min(embeddings.shape[0] - 1, embeddings.shape[1])
    if rank > max_possible_rank:
        raise ValueError(f"Rank {rank} exceeds max possible rank {max_possible_rank}")
    svd = TruncatedSVD(n_components=rank, random_state=42)
    low_rank = svd.fit_transform(embeddings)
    reconstructed = svd.inverse_transform(low_rank)
    return np.linalg.norm(embeddings - reconstructed, axis=1)


@dataclass
class Agg:
    captions: int = 0
    captions_correct: int = 0
    sentences: int = 0
    sentences_correct: int = 0

    def add(self, hallucination_check: object) -> None:
        parsed = parse_hallucination_check(hallucination_check)
        if parsed is None:
            return
        hallucinated, total = parsed
        self.captions += 1
        self.captions_correct += 1 if hallucinated == 0 else 0
        self.sentences += total
        self.sentences_correct += total - hallucinated

    def caption_accuracy(self) -> float:
        return self.captions_correct / self.captions if self.captions else float("nan")

    def sentence_accuracy(self) -> float:
        return self.sentences_correct / self.sentences if self.sentences else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=INPUT_DIR_DEFAULT)
    ap.add_argument("--device", default=None, help="Override device: cpu/cuda")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder_source = LOCAL_ENCODER_SNAPSHOT if os.path.isdir(LOCAL_ENCODER_SNAPSHOT) else ENCODER_ID
    model = SentenceTransformer(encoder_source, device=device, local_files_only=True)

    aggs: Dict[str, Agg] = {m: Agg() for m in MODELS}

    scene_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".csv")])
    for scene_file in scene_files:
        df = pd.read_csv(os.path.join(args.input_dir, scene_file))
        for m in MODELS:
            rows = df.loc[df["Model"] == m].head(10)
            if len(rows) != 10:
                raise RuntimeError(f"{scene_file}: expected 10 rows for {m}, got {len(rows)}")
            captions = rows["Caption"].astype(str).tolist()
            embeddings = model.encode(
                captions,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            residuals = compute_residuals(embeddings, rank=RANK)
            best_pos = int(np.argmin(residuals.astype(float)))
            best_row = rows.iloc[best_pos]
            aggs[m].add(best_row.get("hallucination_check"))

    out_rows = []
    for m in MODELS:
        out_rows.append(
            {
                "model": m,
                "caption_accuracy": aggs[m].caption_accuracy(),
                "sentence_accuracy": aggs[m].sentence_accuracy(),
            }
        )
    out_df = pd.DataFrame(out_rows)
    print(out_df.to_csv(index=False).strip())


if __name__ == "__main__":
    main()

