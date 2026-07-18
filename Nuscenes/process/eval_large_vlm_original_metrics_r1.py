from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


INPUT_DIR_DEFAULT = "./Nuscenes/sensitivity_analysis/large_vlm/results_caption_20260418"
OUTPUT_DIR_DEFAULT = "./Nuscenes/sensitivity_analysis/large_vlm/large_vlm_low_rank_r1"

MODELS = ["deepseek-vl2-small", "Qwen2.5-VL-32B-Instruct", "llava-1.5-13b-hf"]

# Paper-style multi-agent candidate set: 4 + 3 + 3 = 10 captions per scene (by row order).
MULTI_AGENT_SAMPLE_COUNTS = {
    "deepseek-vl2-small": 4,
    "Qwen2.5-VL-32B-Instruct": 3,
    "llava-1.5-13b-hf": 3,
}

LOWRANK_COL = "multi_agent_Error_r=1"


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


@dataclass
class Agg:
    captions: int = 0
    captions_correct: int = 0
    sentences: int = 0
    sentences_correct: int = 0
    missing_hallu_check: int = 0

    def add_row(self, hallucination_check: object) -> None:
        parsed = parse_hallucination_check(hallucination_check)
        if parsed is None:
            self.missing_hallu_check += 1
            return
        hallucinated, total = parsed
        self.captions += 1
        self.captions_correct += 1 if hallucinated == 0 else 0
        self.sentences += total
        self.sentences_correct += total - hallucinated

    def caption_accuracy(self) -> Optional[float]:
        return (self.captions_correct / self.captions) if self.captions else None

    def sentence_accuracy(self) -> Optional[float]:
        return (self.sentences_correct / self.sentences) if self.sentences else None


def first_n_indices(df: pd.DataFrame, model: str, n: int) -> List[int]:
    return df.index[df["Model"] == model].tolist()[:n]


def pick_multi_agent_candidates(df: pd.DataFrame) -> List[int]:
    picked: List[int] = []
    for model_name, k in MULTI_AGENT_SAMPLE_COUNTS.items():
        idx = first_n_indices(df, model_name, k)
        if len(idx) != k:
            raise ValueError(f"{model_name}: expected {k} rows, got {len(idx)}")
        picked.extend(idx)
    return picked


def safe_float(x: object) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def iter_scene_files(input_dir: str) -> Iterable[str]:
    for name in sorted(os.listdir(input_dir)):
        if name.endswith(".csv"):
            yield name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=INPUT_DIR_DEFAULT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    args = parser.parse_args()

    # (1) Per-model: its own 10 captions average correctness (caption-level + sentence-level).
    per_model_10 = {m: Agg() for m in MODELS}
    per_model_10_scene_missing = defaultdict(int)  # model -> scenes missing enough rows

    # (2) Low-rank candidate set: the 10 extracted captions (4+3+3) average correctness.
    cand_10 = Agg()

    # (3) Low-rank selected 1: caption chosen by min(LOWRANK_COL) among the 10 candidates.
    selected_1 = Agg()
    selected_1_by_model: Dict[str, Agg] = defaultdict(Agg)

    scene_count = 0
    for scene_file in iter_scene_files(args.input_dir):
        scene_count += 1
        df = pd.read_csv(os.path.join(args.input_dir, scene_file))

        # Per-model 10 captions: first 10 rows of that model (paper convention).
        for model_name in MODELS:
            idx10 = first_n_indices(df, model_name, 10)
            if len(idx10) != 10:
                per_model_10_scene_missing[model_name] += 1
                continue
            for i in idx10:
                per_model_10[model_name].add_row(df.at[i, "hallucination_check"])

        # Candidate set 10: fixed 4+3+3 sampling.
        cand_idx = pick_multi_agent_candidates(df)
        for i in cand_idx:
            cand_10.add_row(df.at[i, "hallucination_check"])

        # Selected 1: argmin in LOWRANK_COL among candidates (must exist after running low-rank update).
        cand_col_vals = []
        for i in cand_idx:
            v = safe_float(df.at[i, LOWRANK_COL] if LOWRANK_COL in df.columns else None)
            cand_col_vals.append(v)

        if any(v is None for v in cand_col_vals):
            raise ValueError(
                f"{scene_file}: missing {LOWRANK_COL} for some candidate rows; "
                "run low-rank rank=1 write-back first."
            )

        best_pos = int(np.argmin(np.array(cand_col_vals, dtype=float)))
        best_idx = cand_idx[best_pos]
        selected_1.add_row(df.at[best_idx, "hallucination_check"])
        selected_model = str(df.at[best_idx, "Model"])
        selected_1_by_model[selected_model].add_row(df.at[best_idx, "hallucination_check"])

    rows: List[Dict[str, object]] = []
    for model_name in MODELS:
        agg = per_model_10[model_name]
        rows.append(
            {
                "group": "model_self_10",
                "model": model_name,
                "caption_accuracy": agg.caption_accuracy(),
                "sentence_accuracy": agg.sentence_accuracy(),
                "captions_counted": agg.captions,
                "sentences_counted": agg.sentences,
                "missing_hallucination_check": agg.missing_hallu_check,
                "scenes_missing_10_rows": per_model_10_scene_missing.get(model_name, 0),
                "scene_files_total": scene_count,
            }
        )

    rows.append(
        {
            "group": "lowrank_candidates_10",
            "model": "mixed(4+3+3)",
            "caption_accuracy": cand_10.caption_accuracy(),
            "sentence_accuracy": cand_10.sentence_accuracy(),
            "captions_counted": cand_10.captions,
            "sentences_counted": cand_10.sentences,
            "missing_hallucination_check": cand_10.missing_hallu_check,
            "scenes_missing_10_rows": 0,
            "scene_files_total": scene_count,
        }
    )

    rows.append(
        {
            "group": "lowrank_selected_1",
            "model": "mixed(argmin multi_agent_Error_r=1)",
            "caption_accuracy": selected_1.caption_accuracy(),
            "sentence_accuracy": selected_1.sentence_accuracy(),
            "captions_counted": selected_1.captions,
            "sentences_counted": selected_1.sentences,
            "missing_hallucination_check": selected_1.missing_hallu_check,
            "scenes_missing_10_rows": 0,
            "scene_files_total": scene_count,
        }
    )

    for model_name in sorted(selected_1_by_model.keys()):
        agg = selected_1_by_model[model_name]
        rows.append(
            {
                "group": "lowrank_selected_1_by_model",
                "model": model_name,
                "caption_accuracy": agg.caption_accuracy(),
                "sentence_accuracy": agg.sentence_accuracy(),
                "captions_counted": agg.captions,
                "sentences_counted": agg.sentences,
                "missing_hallucination_check": agg.missing_hallu_check,
                "scenes_missing_10_rows": 0,
                "scene_files_total": scene_count,
            }
        )

    out_df = pd.DataFrame(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "performance_original_metrics_rank1.csv")
    out_df.to_csv(out_path, index=False)
    print(out_path)


if __name__ == "__main__":
    main()

