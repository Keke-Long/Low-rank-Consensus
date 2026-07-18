#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer


def compute_embeddings(captions: list[str]) -> np.ndarray:
    model = SentenceTransformer("intfloat/e5-large-v2")
    return model.encode(captions, convert_to_numpy=True)


def load_multi_captions(csv_path: Path) -> tuple[list[str], list[str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    models = ["deepseek-vl2-tiny", "Qwen2-VL-7B-Instruct", "llava-1.5-7b-hf"]
    all_captions: list[str] = []
    caption_labels: list[str] = []

    for model in models:
        model_rows = [row for row in rows if row.get("Model") == model]
        if not model_rows:
            continue
        sample_size = 4 if model == "deepseek-vl2-tiny" else 3
        captions = [row.get("Caption", "") for row in model_rows[:sample_size]]
        all_captions.extend(captions)
        caption_labels.extend([f"{model.split('-')[0]}_{i+1}" for i in range(len(captions))])

    if not all_captions:
        raise ValueError(f"No captions found for expected models in {csv_path}")

    return all_captions, caption_labels


def information_decomposition(
    embeddings: np.ndarray, *, threshold: float
) -> tuple[list[float], list[float], list[float], int]:
    x = embeddings.astype(np.float64)
    u, s, vt = np.linalg.svd(x, full_matrices=False)

    max_rank = min(x.shape)
    ranks = range(1, max_rank + 1)

    fro_low: list[float] = []
    fro_res: list[float] = []
    cum_var: list[float] = []

    total_energy = float(np.sum(s**2))

    for rank in ranks:
        u_r = u[:, :rank]
        s_r = s[:rank]
        vt_r = vt[:rank, :]

        x_low = u_r @ np.diag(s_r) @ vt_r
        x_res = x - x_low

        fro_low.append(float(np.linalg.norm(x_low, "fro")))
        fro_res.append(float(np.linalg.norm(x_res, "fro")))
        cum_var.append(float(np.sum(s_r**2) / total_energy))

    k_star = next((i + 1 for i, v in enumerate(cum_var) if v >= threshold), max_rank)
    return fro_low, fro_res, cum_var, k_star


def plot_information_decomposition(
    *,
    ranks: list[int],
    fro_low: list[float],
    fro_res: list[float],
    cum_var: list[float],
    out_path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(ranks, fro_low, "b-o", linewidth=1, markersize=5, label="Low-rank Part")
    plt.plot(ranks, fro_res, "r-s", linewidth=1, markersize=5, label="Residual Part")
    original_norm = float(np.sqrt((fro_low[-1] ** 2) + (fro_res[-1] ** 2)))
    plt.axhline(
        y=original_norm,
        color="black",
        linestyle="--",
        alpha=0.7,
        label="Original Matrix",
    )

    plt.xlabel("Number of Components (Rank)", fontsize=11)
    plt.ylabel("Information Content (Frobenius Norm)", fontsize=11)
    plt.title("Information Distribution", fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(ranks, cum_var, "g-o", linewidth=1, markersize=5)
    plt.axhline(y=0.8, color="red", linestyle="--", alpha=0.7, label="80% Threshold")
    plt.axhline(y=0.9, color="orange", linestyle="--", alpha=0.7, label="90% Threshold")

    plt.xlabel("Number of Components (Rank)", fontsize=11)
    plt.ylabel("Cumulative Explained Variance Ratio", fontsize=11)
    plt.title("Cumulative Information Capture", fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    if title:
        fig.suptitle(title, fontsize=11)

    plt.tight_layout(pad=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot information decomposition (Frobenius norms + cumulative explained variance) to choose rank."
    )
    parser.add_argument("--csv", required=True, help="Input CSV under Nuscenes/results_caption/")
    parser.add_argument(
        "--out",
        required=True,
        help="Output image path (e.g. Nuscenes/results/1.1 check_rank/information_decomposition_*.jpg)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Explained variance threshold used only for printing selected rank.",
    )
    parser.add_argument("--title", default="", help="Optional title suffix.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    captions, _labels = load_multi_captions(csv_path)
    embeddings = compute_embeddings(captions)
    fro_low, fro_res, cum_var, k_star = information_decomposition(embeddings, threshold=float(args.threshold))

    ranks = list(range(1, len(cum_var) + 1))
    plot_information_decomposition(
        ranks=ranks,
        fro_low=fro_low,
        fro_res=fro_res,
        cum_var=cum_var,
        out_path=out_path,
        title=args.title,
    )

    print(f"csv={csv_path}")
    print(f"threshold={args.threshold}")
    print(f"selected_rank={k_star}")
    print(f"out={out_path}")


if __name__ == "__main__":
    main()
