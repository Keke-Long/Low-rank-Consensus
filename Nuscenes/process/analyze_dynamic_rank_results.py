from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_ADAPTIVE_RESULTS = "./adaptive_rank_95_results.csv"
DEFAULT_SOURCE_DIR = "./Nuscenes/results_caption"
DEFAULT_OUTPUT_DIR = "./Nuscenes/results/dynamic_rank_analysis"

MODEL_SAMPLE_COUNTS: Dict[str, int] = {
    "Qwen2-VL-7B-Instruct": 3,
    "llava-1.5-7b-hf": 3,
    "deepseek-vl2-tiny": 4,
}


def parse_hallucination_check(s: object) -> Optional[Tuple[int, int]]:
    text = "" if s is None else str(s).strip()
    if not text or text.lower() == "nan":
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return None

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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_scene_candidates(scene_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(scene_csv_path)
    df = df[df["Model"].isin(MODEL_SAMPLE_COUNTS.keys())].copy()

    picked_parts: List[pd.DataFrame] = []
    for model_name, k in MODEL_SAMPLE_COUNTS.items():
        sub = df[df["Model"] == model_name].head(k)
        if len(sub) != k:
            raise ValueError(
                f"{os.path.basename(scene_csv_path)}: model={model_name} expected {k} rows, got {len(sub)}"
            )
        picked_parts.append(sub)
    picked = pd.concat(picked_parts, axis=0)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptive-results", default=DEFAULT_ADAPTIVE_RESULTS)
    ap.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    ensure_dir(args.output_dir)

    adaptive = pd.read_csv(args.adaptive_results)
    required_cols = {"scene_file", "rank", "selected_model", "selected_row_index", "caption_correct"}
    missing = required_cols - set(adaptive.columns)
    if missing:
        raise ValueError(f"adaptive results missing columns: {sorted(missing)}")

    rank_only_rows: List[Dict[str, object]] = []
    scene_rows: List[Dict[str, object]] = []

    by_rank = defaultdict(lambda: {"scenes": 0, "correct": 0, "wrong": 0, "wrong_with_clean": 0})

    for _, row in adaptive.iterrows():
        scene_file = row["scene_file"]
        scene_path = os.path.join(args.source_dir, scene_file)
        if not os.path.exists(scene_path):
            raise FileNotFoundError(scene_path)

        selected_rank = int(row["rank"])
        selected_model = str(row["selected_model"])
        selected_row_index = int(row["selected_row_index"])
        selected_correct = int(row["caption_correct"])

        cand = load_scene_candidates(scene_path)

        cand_records: List[Dict[str, object]] = []
        for idx, crow in cand.iterrows():
            parsed = parse_hallucination_check(crow.get("hallucination_check"))
            if parsed is None:
                continue
            h, t = parsed
            cand_records.append(
                {
                    "row_index": int(idx),
                    "model": str(crow["Model"]),
                    "hallu_sentences": int(h),
                    "total_sentences": int(t),
                    "hallu_rate": float(h / t),
                    "caption_correct": int(h == 0),
                }
            )

        if not cand_records:
            continue

        cand_df = pd.DataFrame(cand_records)
        min_hallu_sent = int(cand_df["hallu_sentences"].min())
        min_hallu_rate = float(cand_df["hallu_rate"].min())
        clean_count = int((cand_df["caption_correct"] == 1).sum())
        has_clean = int(clean_count > 0)

        selected_row = cand_df[cand_df["row_index"] == selected_row_index]
        if selected_row.empty:
            selected_h = None
            selected_t = None
            selected_rate = None
            selected_is_min_sent = None
            selected_is_min_rate = None
            regret_sent = None
            regret_rate = None
        else:
            selected_h = int(selected_row.iloc[0]["hallu_sentences"])
            selected_t = int(selected_row.iloc[0]["total_sentences"])
            selected_rate = float(selected_row.iloc[0]["hallu_rate"])
            selected_is_min_sent = int(selected_h == min_hallu_sent)
            selected_is_min_rate = int(abs(selected_rate - min_hallu_rate) < 1e-12)
            regret_sent = int(selected_h - min_hallu_sent)
            regret_rate = float(selected_rate - min_hallu_rate)

        clean_selected_model_count = int(
            ((cand_df["model"] == selected_model) & (cand_df["caption_correct"] == 1)).sum()
        )
        clean_other_models_count = int(
            ((cand_df["model"] != selected_model) & (cand_df["caption_correct"] == 1)).sum()
        )

        wrong_with_clean = int((selected_correct == 0) and (has_clean == 1))
        wrong_with_clean_same_model = int((selected_correct == 0) and (clean_selected_model_count > 0))
        wrong_with_clean_other_model = int((selected_correct == 0) and (clean_other_models_count > 0))

        rank_only_rows.append(
            {
                "scene_file": scene_file,
                "rank": selected_rank,
            }
        )

        scene_rows.append(
            {
                "scene_file": scene_file,
                "selected_rank": selected_rank,
                "selected_model": selected_model,
                "selected_row_index": selected_row_index,
                "selected_caption_correct": selected_correct,
                "selected_hallu_sentences": selected_h,
                "selected_total_sentences": selected_t,
                "selected_hallu_rate": selected_rate,
                "candidate_count": int(len(cand_df)),
                "clean_candidate_count": clean_count,
                "has_clean_candidate": has_clean,
                "min_hallu_sentences_in_candidates": min_hallu_sent,
                "min_hallu_rate_in_candidates": min_hallu_rate,
                "selected_is_min_hallu_sentences": selected_is_min_sent,
                "selected_is_min_hallu_rate": selected_is_min_rate,
                "regret_hallu_sentences_vs_best": regret_sent,
                "regret_hallu_rate_vs_best": regret_rate,
                "clean_candidates_in_selected_model": clean_selected_model_count,
                "clean_candidates_in_other_models": clean_other_models_count,
                "wrong_but_clean_exists": wrong_with_clean,
                "wrong_but_clean_exists_in_selected_model": wrong_with_clean_same_model,
                "wrong_but_clean_exists_in_other_models": wrong_with_clean_other_model,
            }
        )

        by_rank[selected_rank]["scenes"] += 1
        by_rank[selected_rank]["correct"] += selected_correct
        by_rank[selected_rank]["wrong"] += (1 - selected_correct)
        by_rank[selected_rank]["wrong_with_clean"] += wrong_with_clean

    rank_only_df = pd.DataFrame(rank_only_rows).sort_values("scene_file")
    scene_df = pd.DataFrame(scene_rows).sort_values("scene_file")

    rank_summary_rows: List[Dict[str, object]] = []
    for rk in sorted(by_rank.keys()):
        s = by_rank[rk]
        rank_summary_rows.append(
            {
                "rank": rk,
                "scene_count": s["scenes"],
                "caption_accuracy": (s["correct"] / s["scenes"]) if s["scenes"] else None,
                "wrong_count": s["wrong"],
                "wrong_with_clean_exists_count": s["wrong_with_clean"],
                "wrong_with_clean_exists_ratio": (
                    s["wrong_with_clean"] / s["wrong"] if s["wrong"] else None
                ),
            }
        )
    rank_summary_df = pd.DataFrame(rank_summary_rows)

    overall_rows = [
        {
            "metric": "scene_count",
            "value": int(len(scene_df)),
        },
        {
            "metric": "caption_accuracy",
            "value": float(scene_df["selected_caption_correct"].mean()) if len(scene_df) else None,
        },
        {
            "metric": "wrong_count",
            "value": int((scene_df["selected_caption_correct"] == 0).sum()) if len(scene_df) else 0,
        },
        {
            "metric": "wrong_but_clean_exists_count",
            "value": int(scene_df["wrong_but_clean_exists"].sum()) if len(scene_df) else 0,
        },
        {
            "metric": "wrong_but_clean_exists_ratio_among_wrong",
            "value": (
                float(scene_df["wrong_but_clean_exists"].sum())
                / float((scene_df["selected_caption_correct"] == 0).sum())
                if (scene_df["selected_caption_correct"] == 0).sum()
                else None
            ),
        },
        {
            "metric": "selected_is_min_hallu_sentences_ratio",
            "value": float(scene_df["selected_is_min_hallu_sentences"].mean())
            if len(scene_df)
            else None,
        },
        {
            "metric": "avg_regret_hallu_sentences_vs_best",
            "value": float(scene_df["regret_hallu_sentences_vs_best"].mean())
            if len(scene_df)
            else None,
        },
        {
            "metric": "avg_regret_hallu_rate_vs_best",
            "value": float(scene_df["regret_hallu_rate_vs_best"].mean())
            if len(scene_df)
            else None,
        },
    ]
    overall_df = pd.DataFrame(overall_rows)

    rank_only_path = os.path.join(args.output_dir, "dynamic_rank_per_scene.csv")
    scene_path = os.path.join(args.output_dir, "dynamic_rank_scene_diagnostics.csv")
    rank_summary_path = os.path.join(args.output_dir, "dynamic_rank_rank_summary.csv")
    overall_path = os.path.join(args.output_dir, "dynamic_rank_overall_summary.csv")

    rank_only_df.to_csv(rank_only_path, index=False)
    scene_df.to_csv(scene_path, index=False)
    rank_summary_df.to_csv(rank_summary_path, index=False)
    overall_df.to_csv(overall_path, index=False)

    print(f"Wrote: {rank_only_path}")
    print(f"Wrote: {scene_path}")
    print(f"Wrote: {rank_summary_path}")
    print(f"Wrote: {overall_path}")


if __name__ == "__main__":
    main()
