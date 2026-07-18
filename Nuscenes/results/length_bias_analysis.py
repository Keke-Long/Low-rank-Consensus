#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import re
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _clean_for_word_count(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    cleaned = _clean_for_word_count(text)
    if not cleaned:
        return 0
    return sum(1 for tok in cleaned.split(" ") if tok)


def char_count(text: str) -> int:
    cleaned = _clean_for_word_count(text)
    return len(cleaned)


def parse_gt_hallucination_score(hallucination_check: object) -> Optional[float]:
    """
    Convert `hallucination_check` to a numeric score in [0, 1].
    Supports values like: "1:Yes, 2:No, 3:Yes" -> yes_ratio = 2/3.
    Returns None if unparsable / empty.
    """
    if hallucination_check is None:
        return None
    if isinstance(hallucination_check, float) and np.isnan(hallucination_check):
        return None
    s = str(hallucination_check).strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return None
    yes = sum(1 for p in parts if "Yes" in p)
    return yes / len(parts)


def build_caption_df(input_dir: str, residual_col: str) -> Tuple[pd.DataFrame, List[str]]:
    paths = sorted(glob(os.path.join(input_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No .csv files found under: {input_dir}")

    frames: List[pd.DataFrame] = []
    skipped: List[str] = []
    for path in paths:
        scene_id = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)

        required_missing = []
        if "Caption" not in df.columns:
            required_missing.append("Caption")
        if residual_col not in df.columns:
            required_missing.append(residual_col)
        if "hallucination_check" not in df.columns:
            required_missing.append("hallucination_check")
        if required_missing:
            skipped.append(os.path.basename(path))
            continue

        df = df.copy()
        df["scene_id"] = scene_id
        df["caption_id"] = np.arange(len(df), dtype=int)
        df["caption_text"] = df["Caption"].astype(str)
        df["residual_score"] = pd.to_numeric(df[residual_col], errors="coerce")
        df["gt_hallucination_score"] = df["hallucination_check"].apply(parse_gt_hallucination_score)
        df["caption_length"] = df["caption_text"].apply(word_count).astype(int)
        df["char_length"] = df["caption_text"].apply(char_count).astype(int)
        frames.append(
            df[
                [
                    "scene_id",
                    "caption_id",
                    "caption_text",
                    "residual_score",
                    "gt_hallucination_score",
                    "caption_length",
                    "char_length",
                ]
            ]
        )

    all_df = pd.concat(frames, ignore_index=True)
    return all_df, skipped


@dataclass(frozen=True)
class SelectionInfo:
    tie_scenes: int
    eligible_scenes: int
    ineligible_scenes: int


def add_selected_flag(df: pd.DataFrame) -> Tuple[pd.DataFrame, SelectionInfo]:
    """
    Add `selected_flag` based on per-scene min residual_score.
    Tie-breaking: choose the first row after sorting by (residual_score asc, caption_id asc).
    Scenes where no finite residual_score exists are marked ineligible (all selected_flag=0).
    """
    df = df.copy()
    df["selected_flag"] = 0

    tie_scenes = 0
    eligible_scenes = 0
    ineligible_scenes = 0

    for scene_id, g in df.groupby("scene_id", sort=False):
        g_valid = g[np.isfinite(g["residual_score"].to_numpy())]
        if g_valid.empty:
            ineligible_scenes += 1
            continue

        eligible_scenes += 1
        min_val = g_valid["residual_score"].min()
        tied = g_valid[g_valid["residual_score"] == min_val]
        if len(tied) > 1:
            tie_scenes += 1

        chosen = tied.sort_values(["residual_score", "caption_id"], ascending=[True, True]).iloc[0]
        df.loc[(df["scene_id"] == scene_id) & (df["caption_id"] == chosen["caption_id"]), "selected_flag"] = 1

    return df, SelectionInfo(tie_scenes=tie_scenes, eligible_scenes=eligible_scenes, ineligible_scenes=ineligible_scenes)


@dataclass(frozen=True)
class AnalysisAResult:
    scenes_used: int
    scenes_skipped_singleton: int
    scenes_skipped_ineligible: int
    selected_is_shortest_pct: float
    selected_in_shortest25_pct: float
    mean_selected_length_rank_percentile: float


def analysis_a(df: pd.DataFrame) -> AnalysisAResult:
    """
    Per-scene length-rank stats for the selected caption.
    Candidates are restricted to rows with finite residual_score (so selection is defined).
    Scenes with only 1 eligible caption are skipped.
    """
    scenes_used = 0
    scenes_skipped_singleton = 0
    scenes_skipped_ineligible = 0

    selected_is_shortest: List[int] = []
    selected_in_shortest25: List[int] = []
    selected_percentiles: List[float] = []

    for scene_id, g in df.groupby("scene_id", sort=False):
        g = g[np.isfinite(g["residual_score"].to_numpy())].copy()
        n = len(g)
        if n == 0:
            scenes_skipped_ineligible += 1
            continue
        if n == 1:
            scenes_skipped_singleton += 1
            continue
        g["length_rank"] = g["caption_length"].rank(method="min", ascending=True).astype(int)

        sel = g[g["selected_flag"] == 1]
        if sel.empty:
            # Shouldn't happen if selection is built from same eligibility filter; treat as ineligible.
            scenes_skipped_ineligible += 1
            continue

        scenes_used += 1

        sel_rank = int(sel.iloc[0]["length_rank"])
        is_shortest = 1 if sel_rank == 1 else 0
        threshold = int(math.ceil(0.25 * n))
        in_short25 = 1 if sel_rank <= max(1, threshold) else 0
        percentile = (sel_rank - 1) / (n - 1)

        selected_is_shortest.append(is_shortest)
        selected_in_shortest25.append(in_short25)
        selected_percentiles.append(float(percentile))

    def _pct(x: List[int]) -> float:
        return float(np.mean(x) * 100.0) if x else float("nan")

    return AnalysisAResult(
        scenes_used=scenes_used,
        scenes_skipped_singleton=scenes_skipped_singleton,
        scenes_skipped_ineligible=scenes_skipped_ineligible,
        selected_is_shortest_pct=_pct(selected_is_shortest),
        selected_in_shortest25_pct=_pct(selected_in_shortest25),
        mean_selected_length_rank_percentile=float(np.mean(selected_percentiles)) if selected_percentiles else float("nan"),
    )


@dataclass(frozen=True)
class CorrResult:
    n: int
    spearman_rho: float
    spearman_p: float
    pearson_r: float
    pearson_p: float


def analysis_b(df: pd.DataFrame) -> CorrResult:
    valid = df[np.isfinite(df["residual_score"].to_numpy())].copy()
    valid = valid[valid["caption_length"].notna()]
    x = valid["caption_length"].to_numpy(dtype=float)
    y = valid["residual_score"].to_numpy(dtype=float)
    if len(x) < 3:
        return CorrResult(
            n=int(len(x)),
            spearman_rho=float("nan"),
            spearman_p=float("nan"),
            pearson_r=float("nan"),
            pearson_p=float("nan"),
        )
    rho, p_s = spearmanr(x, y)
    r, p_p = pearsonr(x, y)
    return CorrResult(n=int(len(x)), spearman_rho=float(rho), spearman_p=float(p_s), pearson_r=float(r), pearson_p=float(p_p))


@dataclass(frozen=True)
class AnalysisCResult:
    gt_threshold: float
    used_strict_zero: bool
    corr: CorrResult


def analysis_c(df: pd.DataFrame, gt_threshold: float = 0.0, fallback_threshold: float = 0.05) -> AnalysisCResult:
    """
    Control for ground-truth hallucination: restrict to captions with gt_hallucination_score <= threshold.
    Default attempts strict == 0 via threshold=0.0; if sample size < 3, fall back to <= fallback_threshold.
    """
    valid = df[np.isfinite(df["residual_score"].to_numpy())].copy()
    valid = valid[valid["gt_hallucination_score"].notna()]

    strict = valid[valid["gt_hallucination_score"] == 0.0]
    if len(strict) >= 3:
        corr = analysis_b(strict)
        return AnalysisCResult(gt_threshold=0.0, used_strict_zero=True, corr=corr)

    relaxed = valid[valid["gt_hallucination_score"] <= float(fallback_threshold)]
    corr = analysis_b(relaxed) if len(relaxed) >= 3 else CorrResult(
        n=int(len(relaxed)),
        spearman_rho=float("nan"),
        spearman_p=float("nan"),
        pearson_r=float("nan"),
        pearson_p=float("nan"),
    )
    return AnalysisCResult(gt_threshold=float(fallback_threshold), used_strict_zero=False, corr=corr)


def to_markdown_table(rows: List[Tuple[str, str, str]]) -> str:
    # rows: (section, metric, value/notes)
    header = ["Section", "Metric", "Value / Notes"]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for sec, metric, val in rows:
        out.append(f"| {sec} | {metric} | {val} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Length Bias Analysis for low-rank residual selection.")
    ap.add_argument("--input_dir", default="Nuscenes/results_caption", help="Directory of per-scene CSV files.")
    ap.add_argument(
        "--residual_col",
        default="ResidualError_r=5",
        help="Which residual column to use as residual_score (must exist in each CSV).",
    )
    ap.add_argument(
        "--gt_fallback_threshold",
        type=float,
        default=0.05,
        help="If gt_hallucination_score==0 subset is too small (<3), use gt_hallucination_score <= this threshold.",
    )
    args = ap.parse_args()

    df, skipped_files = build_caption_df(args.input_dir, args.residual_col)
    df, sel_info = add_selected_flag(df)

    a = analysis_a(df)
    b = analysis_b(df)
    c = analysis_c(df, gt_threshold=0.0, fallback_threshold=args.gt_fallback_threshold)

    # Summary table (single table output).
    rows: List[Tuple[str, str, str]] = []
    rows.append(("Setup", "Input dir", f"`{args.input_dir}`"))
    rows.append(("Setup", "residual_score column", f"`{args.residual_col}`"))
    rows.append(("Setup", "Skipped files (missing cols)", f"{len(skipped_files):,}" + (f" (e.g. `{skipped_files[0]}`)" if skipped_files else "")))
    rows.append(("Setup", "Captions (total)", f"{len(df):,}"))
    rows.append(("Setup", "Scenes (total)", f"{df['scene_id'].nunique():,}"))
    rows.append(
        (
            "Setup",
            "selected_flag construction",
            "Per `scene_id`, pick min `residual_score`; ties -> first by (`residual_score`, `caption_id`).",
        )
    )
    rows.append(("Setup", "Scenes w/ residual ties", f"{sel_info.tie_scenes:,}"))
    rows.append(("Setup", "Scenes w/ no finite residual", f"{sel_info.ineligible_scenes:,}"))

    rows.append(("A", "Scenes used (n>=2 eligible)", f"{a.scenes_used:,}"))
    rows.append(("A", "Scenes skipped (only 1 eligible)", f"{a.scenes_skipped_singleton:,}"))
    rows.append(("A", "Scenes skipped (0 eligible)", f"{a.scenes_skipped_ineligible:,}"))
    rows.append(("A", "Selected is shortest (%)", f"{a.selected_is_shortest_pct:.2f}"))
    rows.append(("A", "Selected in shortest 25% (%)", f"{a.selected_in_shortest25_pct:.2f}"))
    rows.append(("A", "Mean selected length rank percentile", f"{a.mean_selected_length_rank_percentile:.4f}"))

    rows.append(("B", "Captions used (finite residual)", f"{b.n:,}"))
    rows.append(("B", "Spearman corr(length, residual) (p)", f"{b.spearman_rho:.4f} (p={b.spearman_p:.3g})"))
    rows.append(("B", "Pearson corr(length, residual) (p)", f"{b.pearson_r:.4f} (p={b.pearson_p:.3g})"))

    gt_note = (
        "gt_hallucination_score == 0"
        if c.used_strict_zero
        else f"gt_hallucination_score <= {c.gt_threshold:g} (fallback; strict==0 had <3 samples)"
    )
    rows.append(("C", "GT filter", gt_note))
    rows.append(("C", "Captions used (after GT filter)", f"{c.corr.n:,}"))
    rows.append(("C", "Spearman corr(length, residual) (p)", f"{c.corr.spearman_rho:.4f} (p={c.corr.spearman_p:.3g})"))
    rows.append(("C", "Pearson corr(length, residual) (p)", f"{c.corr.pearson_r:.4f} (p={c.corr.pearson_p:.3g})"))

    print(to_markdown_table(rows))


if __name__ == "__main__":
    main()
