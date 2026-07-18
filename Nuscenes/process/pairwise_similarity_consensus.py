"""
Pairwise Similarity Consensus baseline.

Given 10 candidate captions per scene (sampled across 3 models), compute sentence
embeddings with the same encoder used in the low-rank framework, then pick the
caption whose average cosine similarity to the other captions is the largest.

Outputs are written to: Nuscenes/baselines/pairwise_baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from sentence_transformers import SentenceTransformer
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Nuscenes.config import project_path


ENCODER_ID = "intfloat/e5-large-v2"
DEFAULT_INPUT_DIR = "./Nuscenes/results_caption"
DEFAULT_OUTPUT_DIR = str(project_path("Nuscenes", "baselines", "pairwise_baseline"))


@dataclass(frozen=True)
class CandidateCaption:
    scene_file: str
    scene_id: str
    sample_token: str
    row_index: int
    model: str
    caption: str
    hallucination_check: str


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    if s.lower() == "nan":
        return ""
    return s


def parse_scene_file_name(scene_file: str) -> Tuple[str, str]:
    stem = os.path.splitext(os.path.basename(scene_file))[0]
    if "_" not in stem:
        return stem, ""
    scene_id, sample_token = stem.split("_", 1)
    return scene_id, sample_token


def list_scene_csvs(input_dir: str, include_synthetic: bool) -> List[str]:
    csvs = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
    if include_synthetic:
        return csvs
    return [f for f in csvs if "synthetic" not in f.lower()]


def load_scene_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_candidates_for_scene(
    scene_file: str,
    rows: Sequence[Dict[str, str]],
    model_sample_counts: Dict[str, int],
) -> List[CandidateCaption]:
    scene_id, sample_token = parse_scene_file_name(scene_file)

    candidates: List[CandidateCaption] = []
    for model_name, k in model_sample_counts.items():
        picked = 0
        for idx, row in enumerate(rows):
            if row.get("Model") != model_name:
                continue
            candidates.append(
                CandidateCaption(
                    scene_file=scene_file,
                    scene_id=scene_id,
                    sample_token=sample_token,
                    row_index=idx,
                    model=model_name,
                    caption=_safe_str(row.get("Caption")),
                    hallucination_check=_safe_str(row.get("hallucination_check")),
                )
            )
            picked += 1
            if picked >= k:
                break
        if picked != k:
            raise ValueError(
                f"{scene_file}: expected {k} captions for model={model_name}, got {picked}"
            )

    if len(candidates) != sum(model_sample_counts.values()):
        raise RuntimeError(
            f"{scene_file}: internal error, expected {sum(model_sample_counts.values())} candidates, "
            f"got {len(candidates)}"
        )
    return candidates


def get_device(prefer_cuda: bool) -> str:
    if not prefer_cuda:
        return "cpu"
    if torch is None:
        return "cpu"
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    e = embeddings / norms
    return e @ e.T


def pick_by_pairwise_consensus(
    candidates: Sequence[CandidateCaption],
    model: SentenceTransformer,
) -> Tuple[int, np.ndarray, np.ndarray]:
    captions = [c.caption for c in candidates]
    embeddings = model.encode(
        captions,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    sim = cosine_similarity_matrix(np.asarray(embeddings, dtype=np.float32))

    # avg_{j!=i} sim(i,j) = (sum_j sim(i,j) - sim(i,i)) / (n-1)
    n = sim.shape[0]
    if n < 2:
        return 0, sim, np.zeros((n,), dtype=np.float32)
    row_sums = sim.sum(axis=1) - np.diag(sim)
    avg_sim = row_sums / float(n - 1)
    best_idx = int(np.argmax(avg_sim))
    return best_idx, sim, avg_sim


def parse_hallucination_check(hallucination_check_str: str) -> Optional[Tuple[int, int]]:
    """
    Returns (hallucinated_sentences, total_sentences) for strings like: "1:Yes, 2:No".
    """
    s = _safe_str(hallucination_check_str).strip()
    if not s:
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--limit-scenes", type=int, default=1)
    ap.add_argument("--include-synthetic", action="store_true")
    ap.add_argument("--prefer-cuda", action="store_true")
    ap.add_argument("--save-sim-matrix", action="store_true")
    ap.add_argument("--deepseek-k", type=int, default=4)
    ap.add_argument("--qwen-k", type=int, default=3)
    ap.add_argument("--llava-k", type=int, default=3)
    args = ap.parse_args()

    model_sample_counts = {
        "deepseek-vl2-tiny": args.deepseek_k,
        "Qwen2-VL-7B-Instruct": args.qwen_k,
        "llava-1.5-7b-hf": args.llava_k,
    }
    expected_total = sum(model_sample_counts.values())
    if expected_total != 10:
        raise ValueError(f"Expected 10 candidates total, got {expected_total} from {model_sample_counts}")

    input_dir = args.input_dir
    output_dir = args.output_dir
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "sim_matrices"))

    scene_files = list_scene_csvs(input_dir, include_synthetic=args.include_synthetic)
    if args.limit_scenes is not None and args.limit_scenes > 0:
        scene_files = scene_files[: args.limit_scenes]

    device = get_device(args.prefer_cuda)
    st_model = SentenceTransformer(ENCODER_ID, device=device)

    selected_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []

    total_sentences = 0
    hallucinated_sentences = 0
    missing_hallu = 0

    for scene_file in scene_files:
        csv_path = os.path.join(input_dir, scene_file)
        rows = load_scene_rows(csv_path)
        candidates = build_candidates_for_scene(scene_file, rows, model_sample_counts)

        best_idx, sim, avg_sim = pick_by_pairwise_consensus(candidates, st_model)
        picked = candidates[best_idx]

        # Per-candidate outputs
        for i, c in enumerate(candidates):
            candidate_rows.append(
                {
                    "scene_file": c.scene_file,
                    "scene_id": c.scene_id,
                    "sample_token": c.sample_token,
                    "candidate_i": i,
                    "row_index_in_scene_csv": c.row_index,
                    "model": c.model,
                    "avg_cosine_similarity": float(avg_sim[i]) if len(avg_sim) else None,
                    "is_selected": 1 if i == best_idx else 0,
                    "caption": c.caption,
                    "hallucination_check": c.hallucination_check,
                }
            )

        # Hallucination stats for selected caption
        parsed = parse_hallucination_check(picked.hallucination_check)
        if parsed is None:
            missing_hallu += 1
        else:
            h, t = parsed
            hallucinated_sentences += h
            total_sentences += t

        selected_rows.append(
            {
                "scene_file": picked.scene_file,
                "scene_id": picked.scene_id,
                "sample_token": picked.sample_token,
                "selected_candidate_i": best_idx,
                "row_index_in_scene_csv": picked.row_index,
                "selected_model": picked.model,
                "selected_avg_cosine_similarity": float(avg_sim[best_idx]) if len(avg_sim) else None,
                "selected_caption": picked.caption,
                "hallucination_check": picked.hallucination_check,
            }
        )

        if args.save_sim_matrix:
            sim_path = os.path.join(output_dir, "sim_matrices", f"{picked.scene_id}_{picked.sample_token}.json")
            with open(sim_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "scene_file": picked.scene_file,
                        "scene_id": picked.scene_id,
                        "sample_token": picked.sample_token,
                        "model_sample_counts": model_sample_counts,
                        "captions": [c.caption for c in candidates],
                        "avg_cosine_similarity": avg_sim.tolist(),
                        "similarity_matrix": sim.tolist(),
                        "selected_index": best_idx,
                    },
                    f,
                    ensure_ascii=False,
                )

    write_csv(
        os.path.join(output_dir, "selected_captions.csv"),
        fieldnames=list(selected_rows[0].keys()) if selected_rows else [],
        rows=selected_rows,
    )
    write_csv(
        os.path.join(output_dir, "candidate_scores.csv"),
        fieldnames=list(candidate_rows[0].keys()) if candidate_rows else [],
        rows=candidate_rows,
    )

    summary = {
        "encoder_id": ENCODER_ID,
        "device": device,
        "input_dir": os.path.abspath(input_dir),
        "output_dir": os.path.abspath(output_dir),
        "num_scenes_processed": len(scene_files),
        "model_sample_counts": model_sample_counts,
        "hallucination": {
            "hallucinated_sentences": hallucinated_sentences,
            "total_sentences": total_sentences,
            "sentence_hallucination_rate": (hallucinated_sentences / total_sentences) if total_sentences else None,
            "missing_hallucination_check_scenes": missing_hallu,
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
