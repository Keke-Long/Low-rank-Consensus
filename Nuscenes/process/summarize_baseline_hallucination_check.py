"""
Run hallucination check for summarization baseline outputs.

Reads *_final.csv under a given results directory (each should contain columns:
Model, Caption, Time, hallucination_check) and fills hallucination_check using
the same strict per-sentence verifier as other baselines.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
import re

import pandas as pd
import torch
from nuscenes.nuscenes import NuScenes
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Nuscenes.config import model_offload_dir, nuscenes_dir


NUSCENES_DIR = str(nuscenes_dir())
nusc = NuScenes(version="v1.0-trainval", dataroot=NUSCENES_DIR, verbose=False)


def get_scene_front_image(sample_token: str) -> str | None:
    try:
        sample = nusc.get("sample", sample_token)
        cam_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        image_path = os.path.join(NUSCENES_DIR, cam_data["filename"])
        return image_path if os.path.exists(image_path) else None
    except KeyError:
        return None


def split_caption_into_sentences(caption: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", str(caption).strip())
    return [s.strip() for s in sentences if s and s.strip()]


def load_checker(device: str):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map=device,
        offload_folder=str(model_offload_dir()),
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return model, processor


def detect_hallucination(
    model,
    processor,
    sentence: str,
    detection_data: dict,
    image_path: str | None,
    device: str,
) -> str:
    try:
        image = Image.open(image_path).convert("RGB") if image_path else None
    except Exception:
        image = None

    prompt = f"""
You are analyzing an autonomous driving scene. Your task is to **strictly** verify whether the given sentence is consistent with both the detected objects and the provided image.

1. **Detected objects in the image**: {detection_data['visible_objects']}
2. **Sentence to verify**: {sentence}

**Rules for Hallucination Detection (Be Strict!)**:
- If the sentence mentions an object, but the object **is missing** from both the image and the detected objects → **Answer "Yes" (Hallucination detected).**
- If the sentence **describes the scene inaccurately** (e.g., wrong colors, shapes, or numbers of objects) → **Answer "Yes".**
- If the sentence **contradicts** the detected objects (e.g., saying "there is no truck" when a truck is detected) → **Answer "Yes".**
- If the sentence **describes something in the wrong position** (e.g., "a bus is on the left" but it's actually on the right) → **Answer "Yes".**
- If the sentence **describes a weather condition** (e.g., "it is raining"), but the image suggests otherwise → **Answer "Yes".**
- Otherwise, **Answer "No".**

**Final Answer:**
- Answer only with "Yes" or "No". **Do not include any additional text.**
"""

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if image_path and image is not None:
        messages[0]["content"].insert(0, {"type": "image", "image": image_path})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image] if image is not None else None,
        padding=True,
        return_tensors="pt",
    ).to(torch.device(device if device.startswith("cuda") else "cpu"))

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=3)
    output_texts = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    hallucination_result = output_texts[0].strip().split("\n")[-1]
    match = re.search(r"\b(Yes|No)\b", hallucination_result, re.IGNORECASE)
    hallucination_result = match.group(1) if match else "Unknown"

    del inputs, generated_ids, output_texts
    torch.cuda.empty_cache()
    gc.collect()
    return hallucination_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="Directory containing *_final.csv")
    ap.add_argument("--detection-dir", default="./Nuscenes/results_detection_fact")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    torch.cuda.empty_cache()
    gc.collect()

    model, processor = load_checker(args.device)

    csv_files = sorted([f for f in os.listdir(args.results_dir) if f.endswith("_final.csv")])
    for filename in csv_files:
        parts = filename.replace("_final.csv", "").split("_")
        scene_id = parts[0]
        sample_token = "_".join(parts[1:])

        detection_path = os.path.join(args.detection_dir, f"{scene_id}_{sample_token}.json")
        if not os.path.exists(detection_path):
            print(f"Skip {filename}: missing detection json")
            continue

        df = pd.read_csv(os.path.join(args.results_dir, filename))
        if "hallucination_check" not in df.columns:
            df["hallucination_check"] = None
        # Ensure we can safely assign strings (avoid dtype warnings/errors).
        df["hallucination_check"] = df["hallucination_check"].astype("object")

        missing_rows = df["hallucination_check"].isnull() | (df["hallucination_check"].astype(str).str.strip() == "")
        if not missing_rows.any():
            continue

        image_path = get_scene_front_image(sample_token)
        if not image_path:
            print(f"Skip {filename}: missing image for sample_token={sample_token}")
            continue

        with open(detection_path, "r") as f:
            detection_data = json.load(f)

        for row_idx in df[missing_rows].index:
            caption = df.at[row_idx, "Caption"]
            sentences = split_caption_into_sentences(caption)
            row_hallu = []
            for sent_idx, sentence in enumerate(sentences):
                res = detect_hallucination(model, processor, sentence, detection_data, image_path, args.device)
                row_hallu.append(f"{sent_idx+1}:{res}")
            df.at[row_idx, "hallucination_check"] = ", ".join(row_hallu)

        df = df.loc[:, ~df.columns.duplicated()].copy()
        df.to_csv(os.path.join(args.results_dir, filename), index=False)
        print(f"Checked {filename}")


if __name__ == "__main__":
    main()
