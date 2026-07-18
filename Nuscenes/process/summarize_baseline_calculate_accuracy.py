"""
Compute sentence-level hallucination accuracy for summarization baseline outputs.

Accuracy is defined as: 1 - (hallucinated_sentences / total_sentences),
where hallucination_check is formatted like: "1:Yes, 2:No, 3:Yes".
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import pandas as pd


def parse_hallucination_check(hallucination_check_str):
    s = "" if hallucination_check_str is None else str(hallucination_check_str).strip()
    if not s or s.lower() == "nan":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return None
    total = 0
    hallucinated = 0
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()

    stats = defaultdict(lambda: {"captions": 0, "hallucinated_sentences": 0, "total_sentences": 0})

    csv_files = sorted([f for f in os.listdir(args.results_dir) if f.endswith("_final.csv")])
    for f in csv_files:
        df = pd.read_csv(os.path.join(args.results_dir, f))
        for _, row in df.iterrows():
            model = row.get("Model", "unknown")
            parsed = parse_hallucination_check(row.get("hallucination_check", ""))
            if parsed is None:
                continue
            h, t = parsed
            stats[model]["captions"] += 1
            stats[model]["hallucinated_sentences"] += h
            stats[model]["total_sentences"] += t

    for model, s in stats.items():
        total = s["total_sentences"]
        hallu = s["hallucinated_sentences"]
        acc = 1.0 - (hallu / total) if total else None
        print(f"{model}\tcaptions={s['captions']}\tsentences={total}\thallu={hallu}\taccuracy={acc}")


if __name__ == "__main__":
    main()

