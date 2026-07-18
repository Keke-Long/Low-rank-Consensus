from __future__ import annotations

import argparse
import os
from collections import defaultdict
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
DEFAULT_INPUT_DIR = (
    "./Nuscenes/sensitivity_analysis/large_vlm/results_caption_20260418"
)
DEFAULT_OUTPUT_DIR = (
    "./Nuscenes/sensitivity_analysis/large_vlm/large_vlm_low_rank_r1"
)
RANK = 1
RESIDUAL_COL = "multi_agent_Error_r=1"
MODEL_SAMPLE_COUNTS = {
    "deepseek-vl2-small": 4,
    "Qwen2.5-VL-32B-Instruct": 3,
    "llava-1.5-13b-hf": 3,
}


def parse_hallucination_check(value: object) -> Optional[Tuple[int, int]]:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return None

    hallucinated = 0
    total = 0
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if ":" not in part:
            continue
        _, label = part.split(":", 1)
        total += 1
        if label.strip().lower() == "yes":
            hallucinated += 1

    if total == 0:
        return None
    return hallucinated, total


def pick_candidates(df: pd.DataFrame) -> List[int]:
    picked_indices: List[int] = []
    for model_name, sample_count in MODEL_SAMPLE_COUNTS.items():
        model_indices = df.index[df["Model"] == model_name].tolist()[:sample_count]
        if len(model_indices) != sample_count:
            raise ValueError(
                f"Model {model_name} expected {sample_count} rows, got {len(model_indices)}"
            )
        picked_indices.extend(model_indices)
    return picked_indices


def compute_residuals(embeddings: np.ndarray, rank: int) -> List[float]:
    max_possible_rank = min(embeddings.shape[0] - 1, embeddings.shape[1])
    if rank > max_possible_rank:
        raise ValueError(f"Rank {rank} exceeds max possible rank {max_possible_rank}")

    svd = TruncatedSVD(n_components=rank, random_state=42)
    low_rank_matrix = svd.fit_transform(embeddings)
    reconstructed = svd.inverse_transform(low_rank_matrix)
    return np.linalg.norm(embeddings - reconstructed, axis=1).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--write-back",
        action="store_true",
        help=f"Update {RESIDUAL_COL} in each source CSV.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_source = LOCAL_ENCODER_SNAPSHOT if os.path.isdir(LOCAL_ENCODER_SNAPSHOT) else ENCODER_ID
    model = SentenceTransformer(encoder_source, device=device, local_files_only=True)

    scene_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".csv"))
    selection_rows: List[Dict[str, object]] = []
    aggregate = {
        "scene_count": 0,
        "caption_correct_count": 0,
        "selected_hallucinated_sentences": 0,
        "selected_total_sentences": 0,
        "missing_hallucination_check_count": 0,
    }
    model_breakdown = defaultdict(lambda: {"selected_count": 0, "correct_count": 0})

    for scene_file in scene_files:
        csv_path = os.path.join(args.input_dir, scene_file)
        df = pd.read_csv(csv_path)

        if RESIDUAL_COL not in df.columns:
            df[RESIDUAL_COL] = pd.NA
        else:
            df[RESIDUAL_COL] = pd.NA

        candidate_indices = pick_candidates(df)
        candidate_df = df.loc[candidate_indices].copy()
        embeddings = model.encode(
            candidate_df["Caption"].astype(str).tolist(),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        residuals = compute_residuals(embeddings, rank=RANK)
        candidate_df[RESIDUAL_COL] = residuals

        best_pos = int(candidate_df[RESIDUAL_COL].astype(float).argmin())
        best_row = candidate_df.iloc[best_pos]
        best_index = int(candidate_df.index[best_pos])

        df.loc[candidate_df.index, RESIDUAL_COL] = candidate_df[RESIDUAL_COL]
        if args.write_back:
            df.to_csv(csv_path, index=False)

        parsed = parse_hallucination_check(best_row.get("hallucination_check"))
        if parsed is None:
            aggregate["missing_hallucination_check_count"] += 1
            hallucinated_sentences = None
            total_sentences = None
            sentence_hallucination_rate = None
            caption_correct = None
        else:
            hallucinated_sentences, total_sentences = parsed
            sentence_hallucination_rate = hallucinated_sentences / total_sentences
            caption_correct = 1 if hallucinated_sentences == 0 else 0
            aggregate["caption_correct_count"] += caption_correct
            aggregate["selected_hallucinated_sentences"] += hallucinated_sentences
            aggregate["selected_total_sentences"] += total_sentences

        aggregate["scene_count"] += 1
        model_breakdown[best_row["Model"]]["selected_count"] += 1
        if caption_correct is not None:
            model_breakdown[best_row["Model"]]["correct_count"] += caption_correct

        selection_rows.append(
            {
                "scene_file": scene_file,
                "selected_row_index": best_index,
                "selected_model": best_row["Model"],
                "selected_caption": best_row["Caption"],
                RESIDUAL_COL: float(best_row[RESIDUAL_COL]),
                "hallucination_check": best_row.get("hallucination_check"),
                "hallucinated_sentences": hallucinated_sentences,
                "total_sentences": total_sentences,
                "sentence_hallucination_rate": sentence_hallucination_rate,
                "caption_correct": caption_correct,
            }
        )

    selection_path = os.path.join(args.output_dir, "selected_captions_r1_multi.csv")
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False)

    denominator = aggregate["scene_count"] - aggregate["missing_hallucination_check_count"]
    accuracy = aggregate["caption_correct_count"] / denominator if denominator else None
    hallucination_rate = (
        aggregate["selected_hallucinated_sentences"] / aggregate["selected_total_sentences"]
        if aggregate["selected_total_sentences"]
        else None
    )

    summary_rows = [
        {"Metric": "scene_count", "Value": aggregate["scene_count"]},
        {"Metric": "caption_accuracy", "Value": accuracy},
        {
            "Metric": "selected_sentence_hallucination_rate",
            "Value": hallucination_rate,
        },
        {
            "Metric": "selected_hallucinated_sentences",
            "Value": aggregate["selected_hallucinated_sentences"],
        },
        {
            "Metric": "selected_total_sentences",
            "Value": aggregate["selected_total_sentences"],
        },
        {
            "Metric": "missing_hallucination_check_count",
            "Value": aggregate["missing_hallucination_check_count"],
        },
    ]
    summary_path = os.path.join(args.output_dir, "selection_summary_r1_multi.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    breakdown_rows = []
    for model_name, stats in sorted(model_breakdown.items()):
        selected_count = stats["selected_count"]
        correct_count = stats["correct_count"]
        breakdown_rows.append(
            {
                "selected_model": model_name,
                "selected_count": selected_count,
                "correct_count": correct_count,
                "selection_accuracy_within_selected": (
                    correct_count / selected_count if selected_count else None
                ),
            }
        )
    breakdown_path = os.path.join(args.output_dir, "selection_breakdown_by_model_r1_multi.csv")
    pd.DataFrame(breakdown_rows).to_csv(breakdown_path, index=False)

    print(f"Processed {aggregate['scene_count']} scene CSVs")
    print(f"Caption accuracy: {accuracy}")
    print(f"Selected sentence hallucination rate: {hallucination_rate}")
    print(f"Selection CSV: {selection_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Breakdown CSV: {breakdown_path}")


if __name__ == "__main__":
    main()
