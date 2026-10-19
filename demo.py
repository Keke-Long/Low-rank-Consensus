"""Select a caption by low-rank consensus, without ground-truth labels.

Run ``python demo.py`` after ``pip install numpy sentence-transformers``.
The first run downloads intfloat/e5-large-v2 unless it is already cached.
The ten captions below are illustrative examples, not benchmark data.
"""

from __future__ import annotations

import numpy as np


ENCODER_ID = "intfloat/e5-large-v2"
GATE = 4.75
MAX_RANK = 3
CAPTIONS = [
    "A roadwork area has traffic cones, workers, a truck, and a red car.",
    "Workers stand near traffic cones and construction equipment on the road.",
    "A red car passes a construction zone with workers and a parked truck.",
    "Traffic cones mark a roadwork site where workers and vehicles are present.",
    "A truck and construction equipment occupy part of a road lined with cones.",
    "Road workers stand beside a truck while a red car is nearby.",
    "Cones separate a work area from traffic beside buildings and trees.",
    "Construction workers and a truck are visible near a red car on the road.",
    "A red car approaches roadwork while a cyclist rides between the cones.",
    "Workers repair the road as a blue bus turns past a traffic officer.",
]


def choose_rank_by_gated_spectral_gap(singular_values, gate=GATE, max_rank=MAX_RANK):
    """Apply the experiment's gap rule, then cap the selected rank.

    Test the first gap against the gate. Otherwise choose the largest gap
    for ranks 2 through 4 (when available), before applying ``max_rank``.
    """
    if len(singular_values) < 2:
        return 1, []

    k = min(4, len(singular_values) - 1)
    gaps = [
        float(singular_values[i] / max(singular_values[i + 1], 1e-12))
        for i in range(k)
    ]
    if gaps[0] >= gate or k <= 1:
        rank = 1
    else:
        rank = int(np.argmax(gaps[1:]) + 2)

    rank = max(1, min(rank, max_rank, len(singular_values) - 1))
    return rank, gaps


def select_caption(captions, encoder=None):
    """Return the caption with the smallest low-rank reconstruction residual.

    ``captions`` must contain at least two nonblank strings for one scene.
    An optional SentenceTransformer-compatible ``encoder`` can be reused
    across calls. Embeddings remain uncentered and unnormalized, matching
    the main experiment. Exact residual ties select the first input caption.

    Lower residual means stronger agreement with the candidate pool; it is
    not a probability of correctness or a guarantee of factual accuracy.
    """
    if isinstance(captions, (str, bytes)):
        raise ValueError("Pass a collection of captions, not a single string.")
    try:
        captions = list(captions)
    except TypeError as exc:
        raise ValueError("Captions must be an iterable of strings.") from exc
    if len(captions) < 2:
        raise ValueError("Provide at least two captions for the same scene.")
    if any(not isinstance(text, str) or not text.strip() for text in captions):
        raise ValueError("Every caption must be a nonblank string.")

    if encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install the demo dependencies: pip install numpy sentence-transformers"
            ) from exc
        encoder = SentenceTransformer(ENCODER_ID, device="cpu")

    embeddings = np.asarray(
        encoder.encode(
            captions,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
    )
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(captions)
        or embeddings.shape[1] < 2
    ):
        raise ValueError("Encoder must return one row per caption and at least two columns.")
    if not np.issubdtype(embeddings.dtype, np.number) or np.iscomplexobj(embeddings):
        raise ValueError("Encoder embeddings must contain real numeric values.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Encoder embeddings must not contain NaN or infinity.")
    if not np.any(embeddings):
        raise ValueError("Encoder returned an all-zero matrix; consensus is undefined.")

    # 1. Find the shared low-rank structure using a single SVD.
    U, singular_values, Vt = np.linalg.svd(embeddings, full_matrices=False)
    rank, _ = choose_rank_by_gated_spectral_gap(singular_values)
    reconstructed = U[:, :rank] @ np.diag(singular_values[:rank]) @ Vt[:rank, :]

    # 2. Score each caption by its L2 residual and select the smallest score.
    residuals = np.linalg.norm(embeddings - reconstructed, axis=1)
    best_index = int(np.argmin(residuals))
    return {
        "best_caption": captions[best_index],
        "best_index": best_index,
        "rank": rank,
        "residuals": residuals.tolist(),
    }


def main():
    """Run the illustrative ten-caption example on CPU."""
    print("Illustrative captions only; no ground-truth labels or benchmark evaluation.")
    try:
        result = select_caption(CAPTIONS)
    except (ImportError, ValueError, np.linalg.LinAlgError) as exc:
        raise SystemExit(str(exc)) from None

    print(f"\nSelected rank: {result['rank']}")
    print(f"Selected caption: {result['best_caption']}")
    print("\nResiduals in original input order (lower means closer to consensus):")
    for index, (caption, residual) in enumerate(zip(CAPTIONS, result["residuals"])):
        marker = " <-- selected" if index == result["best_index"] else ""
        print(f"{index + 1:2d}. {residual:.6f}{marker}  {caption}")


if __name__ == "__main__":
    main()
