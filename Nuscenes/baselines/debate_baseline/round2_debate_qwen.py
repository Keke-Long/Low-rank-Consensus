import os
import sys
from pathlib import Path
import torch
import csv
import time
import cv2
import io
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from nuscenes.nuscenes import NuScenes
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Nuscenes.config import nuscenes_dir

torch.cuda.set_device(1)
device = torch.device("cuda:1")

NUSCENES_DIR = str(nuscenes_dir())
nusc = NuScenes(version="v1.0-trainval", dataroot=NUSCENES_DIR, verbose=False)

ROUND1_INPUT_DIR = "./Nuscenes/baselines/debate_baseline/results_round1/"
ROUND2_OUTPUT_DIR = "./Nuscenes/baselines/debate_baseline/results_round2/"
os.makedirs(ROUND2_OUTPUT_DIR, exist_ok=True)

def load_model_and_processor(model_id):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="cuda:1",
    )
    min_pixels = 64 * 28 * 28
    max_pixels = 512 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    return model, processor

def get_image_path_from_csv(csv_filename):
    """从CSV文件名提取scene信息，获取图片路径"""
    # CSV格式: {scene_idx}_{sample_token}_debate.csv
    parts = csv_filename.replace('_debate.csv', '').split('_')
    scene_idx = parts[0]
    sample_token = '_'.join(parts[1:])
    
    # 获取图片路径
    sample = nusc.get('sample', sample_token)
    cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    image_path = os.path.join(NUSCENES_DIR, cam_data['filename'])
    
    return image_path

def compress_image(image_path):
    """读取并压缩图片，返回 BytesIO 对象"""
    img = cv2.imread(image_path)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    _, compressed_img = cv2.imencode(".jpg", img, encode_param)
    return io.BytesIO(compressed_img.tobytes())

def read_round1_refined_captions(csv_path):
    """读取round1的refined captions"""
    captions = {
        'deepseek-vl2-tiny': None,
        'Qwen2-VL-7B-Instruct': None,
        'llava-1.5-7b-hf': None
    }
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['Model']
            stage = row['Stage']
            # 只读取round1的结果
            if stage == 'round1' and model in captions:
                captions[model] = row['Caption']
    
    return captions

def create_round2_prompt(round1_caption, other_refined_1, other_refined_2):
    """构造Round 2 prompt"""
    return f"""You are in the second round of evaluating a driving scene description.

Your Round 1 Refined Description: {round1_caption}

Round 1 Refined Descriptions from Other Models:
Model 2: {other_refined_1}
Model 3: {other_refined_2}

Task: This is the final round of refinement. After seeing the updated descriptions from other models:
1. Check if there are still any remaining inconsistencies in your description
2. Verify that you haven't introduced new errors in Round 1
3. Ensure your description aligns with the emerging consensus

Instructions:
* Provide your FINAL description that is as factually accurate as possible
* Remove any remaining hallucinated content
* If your Round 1 description is already optimal, you may keep it unchanged

Output Format: Provide your final description directly, without explanation."""

def call_qwen_for_debate(model, processor, image_data, prompt_text):
    """使用Qwen模型进行debate（需要图片输入）"""
    
    # 记录开始时间
    start_time = time.time()
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_data)},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True,
        top_p=0.9
    )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    
    # 计算耗时
    elapsed_time = time.time() - start_time
    
    return output_text, elapsed_time

def main():
    model_name = "Qwen2-VL-7B-Instruct"
    model, processor = load_model_and_processor(f"Qwen/{model_name}")
    
    # 遍历所有round1结果文件
    csv_files = sorted([f for f in os.listdir(ROUND1_INPUT_DIR) if f.endswith('_debate.csv')])
    
    for csv_file in csv_files:
        scene_id = csv_file.replace('_debate.csv', '')
        
        # 获取图片路径
        try:
            image_path = get_image_path_from_csv(csv_file)
        except Exception as e:
            print(f"Error getting image path for {csv_file}: {e}")
            continue
        
        # 读取并压缩图片
        compressed_image = compress_image(image_path)
        
        # 读取round1的refined captions
        round1_csv = os.path.join(ROUND1_INPUT_DIR, csv_file)
        
        captions = read_round1_refined_captions(round1_csv)
        
        # 检查是否所有模型都有round1结果
        if None in captions.values():
            print(f"Skipping {scene_id}: missing round1 captions")
            continue
        
        # 检查round2结果文件
        round2_csv = os.path.join(ROUND2_OUTPUT_DIR, f"{scene_id}_final.csv")
        
        # 读取已存在的行（如果文件存在）
        file_exists = os.path.exists(round2_csv)
        all_rows = []
        if file_exists:
            with open(round2_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 保留所有非Qwen的行
                    if row['Model'] != model_name:
                        all_rows.append(row)
        
        # 创建prompt (Qwen看deepseek和llava的round1结果)
        prompt = create_round2_prompt(
            captions[model_name],
            captions['deepseek-vl2-tiny'],
            captions['llava-1.5-7b-hf']
        )
        
        # 使用Qwen模型进行round2 debate（带图片输入）
        final_caption, elapsed_time = call_qwen_for_debate(
            model, processor, compressed_image, prompt
        )
        
        # 保存结果
        if file_exists and len(all_rows) > 0:
            # 文件已存在，添加新结果并重写
            all_rows.append({
                'Model': model_name,
                'Caption': final_caption,
                'Time': f"{elapsed_time:.2f}"
            })
            
            with open(round2_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['Model', 'Caption', 'Time'])
                writer.writeheader()
                writer.writerows(all_rows)
        else:
            # 新文件
            with open(round2_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Model', 'Caption', 'Time'])
                writer.writerow([model_name, final_caption, f"{elapsed_time:.2f}"])
        
        print(f"Processed \033[1;33m{scene_id}\033[0m: {model_name} round2 done (time: {elapsed_time:.2f}s)")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
