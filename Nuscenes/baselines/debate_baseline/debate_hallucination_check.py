import os
import sys
from pathlib import Path
import json
import pandas as pd
import torch
import gc
import re
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from nuscenes.nuscenes import NuScenes
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Nuscenes.config import model_offload_dir, nuscenes_dir

NUSCENES_DIR = str(nuscenes_dir())
nusc = NuScenes(version="v1.0-trainval", dataroot=NUSCENES_DIR, verbose=False)

def get_scene_front_image(sample_token):
    """直接通过 sample_token 找到 CAM_FRONT 图像路径"""
    try:
        sample = nusc.get("sample", sample_token)
        cam_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        image_path = os.path.join(NUSCENES_DIR, cam_data["filename"])

        if os.path.exists(image_path):
            print(f"Found image: {image_path}")
            return image_path
        else:
            print(f"Warning: Image file does not exist: {image_path}")
            return None
    except KeyError:
        print(f"Warning: sample_token {sample_token} not found or CAM_FRONT missing.")
        return None

torch.cuda.set_device(1)

# 清理 CUDA 和 Python 内存
torch.cuda.empty_cache()
gc.collect()

# 加载模型
def load_model_and_processor(model_id):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="cuda:1",
        offload_folder=str(model_offload_dir()),
        max_memory={1: "36GB"}
    )

    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor

model_name = "Qwen/Qwen2-VL-7B-Instruct"
model, processor = load_model_and_processor(model_name)

def split_caption_into_sentences(caption: str) -> list:
    """按 `.`, `?`, `!` 拆分 Caption 为多个句子"""
    sentences = re.split(r'(?<=[.!?])\s+', caption.strip())
    sentences = [s.strip() for s in sentences if s]
    return sentences

def detect_hallucination(sentence: str, detection_data: dict, image_path: str) -> str:
    """使用 LLM 进行单句幻觉检测，输入 Sentence + Detected Objects + Scene Image"""
    # 加载图片
    try:
        image = Image.open(image_path).convert("RGB") if image_path else None
    except Exception as e:
        print(f"Warning: Failed to load image {image_path}. Error: {e}")
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

    # 如果 image_path 存在，添加图片
    if image_path:
        messages[0]["content"].insert(0, {"type": "image", "image": image_path})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=[image] if image else None,
        padding=True,
        return_tensors="pt",
    ).to(torch.device("cuda:1"))

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=3)

    output_texts = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # 提取 LLM 最后一行
    hallucination_result = output_texts[0].strip().split("\n")[-1]

    # 匹配 `Yes` 或 `No`，防止其他字符干扰
    match = re.search(r'\b(Yes|No)\b', hallucination_result, re.IGNORECASE)
    hallucination_result = match.group(1) if match else "Unknown"

    # 清理 CUDA & Python 内存
    del inputs, generated_ids, output_texts
    torch.cuda.empty_cache()
    gc.collect()

    return hallucination_result

# 设置路径
ROUND2_RESULTS_DIR = "./Nuscenes/baselines/debate_baseline/results_round2/"
DETECTION_DIR = "./Nuscenes/results_detection_fact/"

# 遍历 round2 目录，处理所有 _final.csv 文件
for filename in sorted(os.listdir(ROUND2_RESULTS_DIR)):
    if filename.endswith("_final.csv"):
        # 从文件名提取 scene_idx 和 sample_token
        # 格式: {scene_idx}_{sample_token}_final.csv
        parts = filename.replace('_final.csv', '').split('_')
        scene_idx_str = parts[0]
        sample_token = '_'.join(parts[1:])
        
        print(f"\n{'='*60}")
        print(f"Processing scene {scene_idx_str}")
        print(f"{'='*60}")

        csv_path = os.path.join(ROUND2_RESULTS_DIR, filename)
        detection_path = os.path.join(DETECTION_DIR, f"{scene_idx_str}_{sample_token}.json")

        if not os.path.exists(detection_path):
            print(f"Warning: No detection data found at {detection_path}, skipping...")
            continue

        # 读取 CSV 文件
        df = pd.read_csv(csv_path)

        # 检查 hallucination_check 列是否存在
        if "hallucination_check" in df.columns:
            missing_rows = df["hallucination_check"].isnull()
            if not missing_rows.any():
                print(f"File {filename} already processed, skipping...")
                continue
        else:
            # 如果不存在，创建该列
            df["hallucination_check"] = None
            missing_rows = df["hallucination_check"].isnull()

        # 获取场景的 CAM_FRONT 图片
        image_path = get_scene_front_image(sample_token)
        
        if not image_path:
            print(f"Warning: No image found for {sample_token}, skipping...")
            continue

        # 加载检测数据
        with open(detection_path, 'r') as f:
            detection_data = json.load(f)

        # 仅处理 hallucination_check 为空的行
        for row_idx in df[missing_rows].index:
            model_name = df.at[row_idx, "Model"]
            caption = df.at[row_idx, "Caption"]
            
            print(f"\nChecking {model_name}:")
            print(f"Caption: {caption[:100]}...")
            
            sentences = split_caption_into_sentences(caption)

            row_hallucinations = []
            for sent_idx, sentence in enumerate(sentences):
                hallucination_result = detect_hallucination(sentence, detection_data, image_path)
                row_hallucinations.append(f"{sent_idx + 1}:{hallucination_result}")
                print(f"  Sentence {sent_idx + 1}: {hallucination_result}")

            df.at[row_idx, "hallucination_check"] = ", ".join(row_hallucinations)

        # 确保没有重复列
        df = df.loc[:, ~df.columns.duplicated()].copy()
        
        # 保存处理后的文件
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved results to {filename}")

print("\n" + "="*60)
print("Hallucination check completed for all debate results!")
print("="*60)
