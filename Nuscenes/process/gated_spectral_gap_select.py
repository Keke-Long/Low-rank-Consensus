"""
Gated spectral-gap multi-agent selector (fixed policy).

Policy (main version):
1) Build 10-candidate pool per scene: deepseek(4) + qwen(3) + llava(3)
2) Compute SVD on caption embedding matrix X
3) If sigma1/sigma2 >= gate, choose r=1
4) Else choose r in {2,3,4} by largest local spectral gap, then cap by max_rank
5) Select caption with minimum reconstruction residual under chosen rank

Outputs:
- rank_per_scene CSV
- selected caption CSV (with correctness + candidate pool diagnostics)
- summary CSV
- by-rank breakdown CSV
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Nuscenes.config import project_path


DEFAULT_RESULTS_DIR = str(project_path("Nuscenes", "results_caption"))
DEFAULT_OUTPUT_DIR = str(project_path("Nuscenes", "results", "gated_spectral_gap_main"))
DEFAULT_GATE = 4.75
DEFAULT_MAX_RANK = 3

MODEL_SAMPLE_COUNTS = {
    "deepseek-vl2-tiny": 4,
    "Qwen2-VL-7B-Instruct": 3,
    "llava-1.5-7b-hf": 3,
}

ENCODER_ID = "intfloat/e5-large-v2"
LOCAL_ENCODER_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--intfloat--e5-large-v2/snapshots/"
    "f169b11e22de13617baa190a028a32f3493550b6"
)


def parse_hallucination_check(hc: object) -> Tuple[int, int, int]:
    s = "" if hc is None else str(hc).strip()
    if not s or s.lower() == "nan":
        return 0, 0, 0
    parts = [p.strip() for p in s.split(",") if p.strip()]
    total = 0
    hallu = 0
    for p in parts:
        if ":" not in p:
            continue
        _, label = p.split(":", 1)
        total += 1
        if label.strip().lower() == "yes":
            hallu += 1
    if total == 0:
        return 0, 0, 0
    return hallu, total, 1 if hallu == 0 else 0


def choose_rank_by_gated_spectral_gap(singular_values: np.ndarray, gate: float, max_rank: int) -> Tuple[int, List[float]]:
    # Need at least 5 singular values to evaluate gaps for ranks 1..4.
    # For 10xD matrix this is normally available; keep safe fallback anyway.
    if len(singular_values) < 2:
        return 1, []

    k = min(4, len(singular_values) - 1)
    gaps = [float(singular_values[i] / max(singular_values[i + 1], 1e-12)) for i in range(k)]

    if gaps[0] >= gate:
        rank = 1
    else:
        if k <= 1:
            rank = 1
        else:
            # ranks 2..(k+1), from gaps[1:]
            rank = int(np.argmax(gaps[1:]) + 2)

    rank = max(1, min(rank, max_rank, len(singular_values) - 1))
    return rank, gaps


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--gate", type=float, default=DEFAULT_GATE)
    ap.add_argument("--max-rank", type=int, default=DEFAULT_MAX_RANK)
    args = ap.parse_args()

    ensure_dir(args.output_dir)

    encoder_source = LOCAL_ENCODER_SNAPSHOT if os.path.isdir(LOCAL_ENCODER_SNAPSHOT) else ENCODER_ID
    model = SentenceTransformer(
        encoder_source,
        device="cpu",
        local_files_only=os.path.isdir(LOCAL_ENCODER_SNAPSHOT),
    )

    scene_files = sorted(
        f for f in os.listdir(args.results_dir) if f.endswith(".csv") and "synthetic" not in f.lower()
    )

    rank_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    by_rank_stats = defaultdict(lambda: {"scenes": 0, "correct": 0, "hallu": 0, "total": 0, "any_clean": 0, "missed_clean": 0})

    for idx, scene_file in enumerate(scene_files, start=1):
        csv_path = os.path.join(args.results_dir, scene_file)
        df = pd.read_csv(csv_path)

        candidates = []
        for model_name, k in MODEL_SAMPLE_COUNTS.items():
            candidates.append(df[df["Model"] == model_name].head(k))
        cand = pd.concat(candidates).copy()
        cand["row_index"] = cand.index

        captions = cand["Caption"].astype(str).tolist()
        X = model.encode(
            captions,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )

        _, S, _ = np.linalg.svd(X, full_matrices=False)
        rank, gaps = choose_rank_by_gated_spectral_gap(S, args.gate, args.max_rank)

        U, _, Vt = np.linalg.svd(X, full_matrices=False)
        X_hat = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
        residual = np.linalg.norm(X - X_hat, axis=1)
        cand["selection_residual"] = residual

        hallu_info = cand["hallucination_check"].apply(parse_hallucination_check)
        cand["hallu"] = hallu_info.apply(lambda x: x[0])
        cand["total"] = hallu_info.apply(lambda x: x[1])
        cand["caption_correct"] = hallu_info.apply(lambda x: x[2])

        best = cand.sort_values("selection_residual").iloc[0]

        any_clean = int((cand["caption_correct"] == 1).any())
        clean_count = int((cand["caption_correct"] == 1).sum())
        missed_clean = int(any_clean == 1 and int(best["caption_correct"]) == 0)

        rank_rows.append(
            {
                "scene_file": scene_file,
                "scene_id": scene_file.split("_", 1)[0],
                "rank": rank,
                "gate": args.gate,
                "max_rank": args.max_rank,
                "gap_1_over_2": gaps[0] if len(gaps) > 0 else np.nan,
                "gap_2_over_3": gaps[1] if len(gaps) > 1 else np.nan,
                "gap_3_over_4": gaps[2] if len(gaps) > 2 else np.nan,
                "gap_4_over_5": gaps[3] if len(gaps) > 3 else np.nan,
            }
        )

        selected_rows.append(
            {
                "scene_file": scene_file,
                "scene_id": scene_file.split("_", 1)[0],
                "rank": rank,
                "selected_model": best["Model"],
                "selected_row_index": int(best["row_index"]),
                "selected_caption": best["Caption"],
                "selection_residual": float(best["selection_residual"]),
                "hallucination_check": best.get("hallucination_check", ""),
                "hallucinated_sentences": int(best["hallu"]),
                "total_sentences": int(best["total"]),
                "caption_correct": int(best["caption_correct"]),
                "any_clean_candidate_in_pool": any_clean,
                "num_clean_candidates_in_pool": clean_count,
                "missed_clean_candidate": missed_clean,
                "pool_size": int(len(cand)),
            }
        )

        s = by_rank_stats[rank]
        s["scenes"] += 1
        s["correct"] += int(best["caption_correct"])
        s["hallu"] += int(best["hallu"])
        s["total"] += int(best["total"])
        s["any_clean"] += any_clean
        s["missed_clean"] += missed_clean

        if idx % 60 == 0 or idx == len(scene_files):
            print(f"Processed {idx}/{len(scene_files)} scenes")

    rank_df = pd.DataFrame(rank_rows)
    selected_df = pd.DataFrame(selected_rows)

    caption_acc = float(selected_df["caption_correct"].mean()) if len(selected_df) else float("nan")
    sentence_acc = (
        1.0 - float(selected_df["hallucinated_sentences"].sum()) / float(selected_df["total_sentences"].sum())
        if float(selected_df["total_sentences"].sum()) > 0
        else float("nan")
    )
    any_clean_rate = float(selected_df["any_clean_candidate_in_pool"].mean()) if len(selected_df) else float("nan")
    missed_clean_rate = float(selected_df["missed_clean_candidate"].mean()) if len(selected_df) else float("nan")

    summary_df = pd.DataFrame(
        [
            {"metric": "scene_count", "value": len(selected_df)},
            {"metric": "caption_accuracy", "value": caption_acc},
            {"metric": "sentence_accuracy", "value": sentence_acc},
            {"metric": "any_clean_candidate_rate", "value": any_clean_rate},
            {"metric": "missed_clean_candidate_rate", "value": missed_clean_rate},
            {"metric": "gate", "value": args.gate},
            {"metric": "max_rank", "value": args.max_rank},
        ]
    )

    rank_breakdown_rows = []
    for r in sorted(by_rank_stats.keys()):
        s = by_rank_stats[r]
        rank_breakdown_rows.append(
            {
                "rank": r,
                "scene_count": s["scenes"],
                "caption_accuracy": s["correct"] / s["scenes"] if s["scenes"] else np.nan,
                "sentence_accuracy": (1.0 - (s["hallu"] / s["total"])) if s["total"] else np.nan,
                "any_clean_candidate_rate": s["any_clean"] / s["scenes"] if s["scenes"] else np.nan,
                "missed_clean_candidate_rate": s["missed_clean"] / s["scenes"] if s["scenes"] else np.nan,
            }
        )
    rank_breakdown_df = pd.DataFrame(rank_breakdown_rows)

    rank_path = os.path.join(args.output_dir, "rank_per_scene_gate4p75_cap3.csv")
    selected_path = os.path.join(args.output_dir, "selected_per_scene_gate4p75_cap3.csv")
    summary_path = os.path.join(args.output_dir, "summary_gate4p75_cap3.csv")
    rank_breakdown_path = os.path.join(args.output_dir, "by_rank_breakdown_gate4p75_cap3.csv")

    rank_df.to_csv(rank_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    rank_breakdown_df.to_csv(rank_breakdown_path, index=False)

    print("\nSaved outputs:")
    print(rank_path)
    print(selected_path)
    print(summary_path)
    print(rank_breakdown_path)
    print("\nKey metrics:")
    print(f"caption_accuracy={caption_acc:.6f}")
    print(f"sentence_accuracy={sentence_acc:.6f}")


if __name__ == "__main__":
    main()
