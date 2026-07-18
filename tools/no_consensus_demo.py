#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD


@dataclass(frozen=True)
class Row:
    timestamp: str
    model: str
    caption: str


def write_synthetic_csv(csv_path: Path) -> None:
    models = [
        "Qwen2-VL-7B-Instruct",
        "deepseek-vl2-tiny",
        "llava-1.5-7b-hf",
    ]

    # Intentionally "no consensus": logically coherent captions that describe plausible scenes,
    # but are unrelated to the actual nuScenes images (hallucination scenario: internally
    # consistent text, externally mismatched content).
    captions = [
        "Inside a quiet supermarket, a cashier scans groceries while two customers wait with baskets; clear price tags hang above produce, and a child compares cereal boxes near the checkout.",
        "In a small neighborhood café, a barista steams milk for cappuccinos as laptop users type at wooden tables; sunlight falls through the window, and a chalkboard menu lists pastries and tea.",
        "At an airport gate, passengers line up for boarding with carry-on bags; a display shows the flight number and departure time, and staff check passports beside a roped queue.",
        "In a school classroom, students sit at desks facing a whiteboard covered with math problems; the teacher holds a marker, and backpacks rest under chairs beside neatly stacked notebooks.",
        "Along a pedestrian shopping street, people stroll past storefronts and street lamps; a musician plays guitar near a bench, and cyclists walk their bikes while chatting in the evening.",
        "Inside a public library, patrons browse tall shelves and read at long tables; a librarian stamps returns, and a notice board advertises children’s story hour and computer reservations.",
        "In a hospital reception area, a nurse calls the next patient while others sit quietly; a digital sign lists clinic rooms, and a vending machine hums beside the corridor.",
        "In an office meeting room, colleagues review a presentation on a wall screen; coffee cups sit on the table, and one person takes notes while another points to a chart.",
        "In a museum gallery, visitors walk slowly between framed photographs; the lighting is soft, an audio guide speaks in headphones, and a guard stands near a rope barrier.",
        "In a hotel lobby, a receptionist checks in a couple with suitcases; fresh flowers sit on a marble counter, and a concierge arranges a taxi beside a map of the city.",
        "In a busy train station, commuters read timetables and wait on the platform; a conductor whistles, doors open, and an announcement echoes as people step aboard with umbrellas.",
        "At a weekend farmers market, vendors arrange vegetables and jars of honey; shoppers sample fruit, a dog waits on a leash, and paper signs show prices in neat handwriting.",
        "In a university lecture hall, students watch slides on an overhead projector; the professor speaks into a microphone, and rows of seats fill as notebooks open and pens scribble.",
        "Inside a gym, a trainer demonstrates a stretching routine near mirrored walls; treadmills run in the background, and a small group follows along while towels and water bottles rest nearby.",
        "In a restaurant kitchen, cooks chop vegetables and call orders over sizzling pans; stainless counters are clean, a timer beeps, and plated dishes are lined up under heat lamps.",
        "At a hair salon, a stylist trims a client’s hair while another person waits with a magazine; mirrors reflect bright lights, and bottles of shampoo are arranged by the sink.",
        "In a daycare playroom, toddlers stack blocks and draw with crayons on a low table; a caregiver reads aloud, and colorful toys are sorted into labeled bins along the wall.",
        "Inside a movie theater, the lights dim as trailers begin on a large screen; people settle into seats with popcorn, and an exit sign glows softly near the aisle.",
        "In a city park, families picnic on grass beside a pond; children feed ducks, joggers pass on a paved path, and a maintenance worker empties bins near the playground.",
        "At a construction site entrance, workers wear hard hats and check a schedule board; cones mark a safe walkway, a supervisor speaks on a radio, and tools are stored in a locked trailer.",
        "In a science lab, students wear goggles and measure liquids in graduated cylinders; a teacher monitors safety, and labeled beakers sit on a tray beside a microscope and worksheets.",
        "Inside a courthouse hallway, lawyers review documents while clients wait on benches; signs point to courtrooms, security officers stand by metal detectors, and footsteps echo on stone floors.",
        "In a small grocery warehouse, staff sort boxes onto shelves with a hand truck; inventory labels face outward, a clipboard tracks counts, and fluorescent lights buzz overhead.",
        "At a community center, volunteers set up folding chairs for an evening workshop; a projector is tested, name tags are laid out, and a pot of coffee warms on a side table.",
        "In a bookstore, readers browse new releases and ask for recommendations; the cashier wraps a gift, and a poster announces an author talk scheduled for next Thursday evening.",
        "Inside a coffee roasting shop, burlap sacks of beans are stacked neatly; the roaster turns slowly, a rich aroma fills the room, and staff record batch notes on a tablet.",
        "In a music rehearsal studio, a drummer counts in while others tune instruments; sound panels cover the walls, cables run to an amp, and sheet music is clipped to stands.",
        "At a seaside aquarium, visitors watch fish glide behind thick glass; children press hands to the tank, staff answer questions, and a sign explains the feeding schedule.",
        "In a residential kitchen, a family prepares dinner together; someone washes rice, another chops herbs, and a pot simmers on the stove while a timer counts down quietly.",
        "In a quiet art classroom, students paint landscapes at easels with brushes and palettes; the instructor offers feedback, and drying racks hold finished canvases near the door.",
    ]

    start = datetime(2024, 1, 1, 10, 0, 0)
    rows: list[Row] = []
    for idx, caption in enumerate(captions):
        model = models[idx // 10]
        ts = start + timedelta(minutes=idx)
        rows.append(Row(timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"), model=model, caption=caption))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Timestamp", "Model", "Caption"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"Timestamp": r.timestamp, "Model": r.model, "Caption": r.caption})


def compute_embeddings(captions: list[str]) -> np.ndarray:
    model = SentenceTransformer("intfloat/e5-large-v2")
    return model.encode(captions, convert_to_numpy=True)


def low_rank_approximation(embeddings: np.ndarray, rank: int) -> np.ndarray:
    if rank <= 0:
        return np.zeros_like(embeddings, dtype=np.float64)
    max_rank = min(embeddings.shape[0] - 1, embeddings.shape[1], rank)
    svd = TruncatedSVD(n_components=max_rank, random_state=42)
    low_rank_matrix = svd.fit_transform(embeddings)
    return svd.inverse_transform(low_rank_matrix)


def plot_svd_comparison(embeddings: np.ndarray, rank: int, out_path: Path) -> None:
    _, s_orig, _ = np.linalg.svd(embeddings, full_matrices=False)
    m_r = low_rank_approximation(embeddings, rank)
    _, s_r, _ = np.linalg.svd(m_r, full_matrices=False)

    x = np.arange(len(s_orig))
    plt.figure(figsize=(3.5, 3))

    consensus_patch = plt.fill_between(
        x, 0, s_r, color="lightblue", alpha=0.5, label="Consensus"
    )
    halluc_patch = plt.fill_between(
        x, s_r, s_orig, color="orange", alpha=0.4, label="Hallucination"
    )

    orig_line, = plt.plot(
        x,
        s_orig,
        color="black",
        linestyle="-",
        linewidth=1,
        marker="o",
        markersize=5,
        markerfacecolor="none",
        markeredgecolor="black",
        label=r"Original Matrix $M$",
    )

    lowrank_line, = plt.plot(
        x,
        s_r,
        color="lightseagreen",
        linestyle="--",
        linewidth=1,
        marker="s",
        markersize=3,
        markerfacecolor="lightseagreen",
        markeredgecolor="lightseagreen",
        label=r"Low-rank Matrix $\hat{M}$",
    )

    plt.xlabel("Singular Value Index")
    plt.ylabel("Amplitude")
    plt.legend(
        handles=[orig_line, lowrank_line, consensus_patch, halluc_patch],
        loc="upper right",
        frameon=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_captions_for_multi_selection(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Same selection pattern as `Nuscenes/results/1.1 check_rank/visualization_0_svd_comparison.py`
    selected_indices = [1, 2, 3, 11, 12, 13, 21, 22, 23, 24]
    return [row["Caption"] for i, row in enumerate(rows) if i in selected_indices]


def load_captions_for_matrix_comparison(csv_path: Path) -> tuple[list[str], list[str]]:
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

    return all_captions, caption_labels


def load_all_captions(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return [row.get("Caption", "") for row in rows]


def plot_matrix_comparison(embeddings: np.ndarray, caption_labels: list[str], rank: int, out_path: Path) -> None:
    x = embeddings.astype(np.float64)
    u, s, vt = np.linalg.svd(x, full_matrices=False)

    u_r = u[:, :rank]
    s_r = s[:rank]
    vt_r = vt[:rank, :]

    x_reconstructed = u_r @ np.diag(s_r) @ vt_r
    residual = x - x_reconstructed

    x_display = x[:, :40]
    x_recon_display = x_reconstructed[:, :40]
    residual_display = residual[:, :40]

    all_data = np.concatenate(
        [x_display.flatten(), x_recon_display.flatten(), residual_display.flatten()]
    )
    vmin, vmax = np.percentile(all_data, [2, 98])

    fig, axes = plt.subplots(1, 3, figsize=(14, 3))

    im1 = axes[0].imshow(x_display, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
    axes[0].set_title("Original Matrix (first 40 dims)", fontsize=11)
    axes[0].set_ylabel("Captions", fontsize=11)
    axes[0].set_xlabel("Feature Dimensions", fontsize=11)
    axes[0].set_yticks(range(len(caption_labels)))
    axes[0].set_yticklabels(caption_labels, fontsize=9)
    axes[0].set_xticks(range(0, 40, 10))

    axes[1].imshow(x_recon_display, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
    axes[1].set_title("Low-rank Matrix", fontsize=11)
    axes[1].set_ylabel("Captions", fontsize=11)
    axes[1].set_xlabel("Feature Dimensions", fontsize=11)
    axes[1].set_yticks(range(len(caption_labels)))
    axes[1].set_yticklabels(caption_labels, fontsize=9)
    axes[1].set_xticks(range(0, 40, 10))

    axes[2].imshow(residual_display, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
    axes[2].set_title("Residual Matrix", fontsize=11)
    axes[2].set_ylabel("Captions", fontsize=11)
    axes[2].set_xlabel("Feature Dimensions", fontsize=11)
    axes[2].set_yticks(range(len(caption_labels)))
    axes[2].set_yticklabels(caption_labels, fontsize=9)
    axes[2].set_xticks(range(0, 40, 10))

    fig.subplots_adjust(wspace=0.7, right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label("Feature Value", rotation=270, labelpad=20, fontsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_information_decomposition(
    embeddings: np.ndarray, out_path: Path, variance_threshold: float = 0.8
) -> None:
    """
    Visualize low-rank vs residual information and cumulative explained variance.
    Uses Frobenius norms for information content and singular values for variance.
    """
    x = embeddings.astype(np.float64)
    _, s, _ = np.linalg.svd(x, full_matrices=False)

    max_rank = min(x.shape)
    ranks = np.arange(1, max_rank + 1)

    fro_lowrank: list[float] = []
    fro_residual: list[float] = []
    cumulative_variance: list[float] = []

    u, _, vt = np.linalg.svd(x, full_matrices=False)
    total_var = float(np.sum(s**2))

    for rank in ranks:
        u_r = u[:, :rank]
        s_r = s[:rank]
        vt_r = vt[:rank, :]
        x_lowrank = u_r @ np.diag(s_r) @ vt_r
        x_residual = x - x_lowrank

        fro_lowrank.append(float(np.linalg.norm(x_lowrank, "fro")))
        fro_residual.append(float(np.linalg.norm(x_residual, "fro")))
        cumulative_variance.append(float(np.sum(s_r**2) / total_var))

    # Match layout/labels with `Nuscenes/results/1.1 check_rank/information_decomposition.jpg`.
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(ranks, fro_lowrank, "b-o", linewidth=1, markersize=5, label="Low-rank Part")
    plt.plot(ranks, fro_residual, "r-s", linewidth=1, markersize=5, label="Residual Part")
    plt.axhline(
        y=float(np.linalg.norm(x, "fro")),
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
    plt.plot(ranks, cumulative_variance, "g-o", linewidth=1, markersize=5)
    plt.axhline(y=variance_threshold, color="red", linestyle="--", alpha=0.7, label="80% Threshold")
    plt.axhline(y=0.9, color="orange", linestyle="--", alpha=0.7, label="90% Threshold")

    plt.xlabel("Number of Components (Rank)", fontsize=11)
    plt.ylabel("Cumulative Explained Variance Ratio", fontsize=11)
    plt.title("Cumulative Information Capture", fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    plt.tight_layout(pad=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a synthetic 'no consensus' captions CSV (internally coherent but externally mismatched) "
            "and regenerate the associated analysis figures."
        )
    )
    parser.add_argument(
        "--csv-out",
        default="Nuscenes/results_caption/0000_synthetic_no_consensus.csv",
        help="Where to write the synthetic CSV (results_caption folder logic).",
    )
    parser.add_argument(
        "--skip-csv-write",
        action="store_true",
        help="Do not overwrite the CSV; read captions from the existing file at --csv-out.",
    )
    parser.add_argument(
        "--out-dir",
        default="Nuscenes/results/synthetic_no_consensus",
        help="Directory where analysis figures are written.",
    )
    parser.add_argument(
        "--ranks",
        default="0,1,5",
        help="Comma-separated ranks to generate for SVD and matrix comparison plots (e.g. '0,1,5').",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.8,
        help="Explained-variance threshold for the information decomposition plot.",
    )
    parser.add_argument(
        "--info-scope",
        choices=["all", "matrix"],
        default="matrix",
        help="Captions used for information decomposition: all rows, or the same subset as matrix comparison.",
    )
    args = parser.parse_args()

    csv_out = Path(args.csv_out)
    out_dir = Path(args.out_dir)
    ranks = [int(r.strip()) for r in str(args.ranks).split(",") if r.strip() != ""]

    if not args.skip_csv_write:
        write_synthetic_csv(csv_out)
    captions = load_captions_for_multi_selection(csv_out)
    embeddings = compute_embeddings(captions)
    for rank in ranks:
        plot_svd_comparison(
            embeddings,
            rank=rank,
            out_path=out_dir / f"svd_comparison_rank={rank} synthetic_no_consensus.png",
        )

    matrix_captions, matrix_labels = load_captions_for_matrix_comparison(csv_out)
    matrix_embeddings = compute_embeddings(matrix_captions)
    for rank in ranks:
        plot_matrix_comparison(
            matrix_embeddings,
            caption_labels=matrix_labels,
            rank=rank,
            out_path=out_dir / f"matrix_comparison_rank={rank} synthetic_no_consensus.jpg",
        )

    if args.info_scope == "all":
        info_captions = load_all_captions(csv_out)
        info_embeddings = compute_embeddings(info_captions)
    else:
        info_embeddings = matrix_embeddings

    plot_information_decomposition(
        info_embeddings,
        out_path=out_dir / "information_decomposition_synthetic_no_consensus_80.jpg",
        variance_threshold=float(args.variance_threshold),
    )

    print(f"Wrote synthetic CSV: {csv_out}")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
