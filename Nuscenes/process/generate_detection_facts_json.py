"""
Generate detection fact JSON files (visible objects) for NuScenes scenes.

This is used by hallucination check scripts which require:
  Nuscenes/results_detection_fact/{scene_idx}_{first_sample_token}.json

The JSON format matches existing files:
  {
    "scene_name": "...",
    "sample_token": "...",
    "visible_objects": [...]
  }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Set

from nuscenes.nuscenes import NuScenes
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Nuscenes.config import nuscenes_dir


NUSCENES_DIR = str(nuscenes_dir())
DEFAULT_OUTPUT_DIR = "./Nuscenes/results_detection_fact"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_visible_objects(nusc: NuScenes, sample_token: str) -> List[str]:
    sample = nusc.get("sample", sample_token)
    cam_name = "CAM_FRONT"
    _, boxes, _ = nusc.get_sample_data(sample["data"][cam_name])

    visible: Set[str] = set()
    for box in boxes:
        ann = nusc.get("sample_annotation", box.token)
        try:
            visibility = int(ann.get("visibility_token", 0))
        except Exception:
            visibility = 0
        if visibility >= 2:
            visible.add(box.name)
    return sorted(visible)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-scenes", type=int, default=300)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.output_dir)

    nusc = NuScenes(version="v1.0-trainval", dataroot=NUSCENES_DIR, verbose=False)

    processed = 0
    skipped = 0
    for scene in nusc.scene[: args.num_scenes]:
        scene_idx = scene["name"][6:]
        first_sample_token = scene["first_sample_token"]
        out_path = os.path.join(args.output_dir, f"{scene_idx}_{first_sample_token}.json")

        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        visible_objects = compute_visible_objects(nusc, first_sample_token)
        data = {
            "scene_name": scene["name"],
            "sample_token": first_sample_token,
            "visible_objects": visible_objects,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        processed += 1
        if processed % 25 == 0:
            print(f"processed={processed} (last={out_path})")

    print(f"Done. processed={processed}, skipped={skipped}, output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
