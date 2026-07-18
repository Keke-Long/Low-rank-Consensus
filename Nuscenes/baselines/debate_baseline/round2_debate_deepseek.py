import os
import sys
from pathlib import Path
import torch
import csv
import time
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images
from nuscenes.nuscenes import NuScenes
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Nuscenes.config import nuscenes_dir

import transformers
transformers.logging.set_verbosity_error()

NUSCENES_DIR = str(nuscenes_dir())
nusc = NuScenes(version="v1.0-trainval", dataroot=NUSCENES_DIR, verbose=False)

ROUND1_INPUT_DIR = "./Nuscenes/baselines/debate_baseline/results_round1/"
ROUND2_OUTPUT_DIR = "./Nuscenes/baselines/debate_baseline/results_round2/"
os.makedirs(ROUND2_OUTPUT_DIR, exist_ok=True)

def load_model_and_processor(model_path):
    processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).cuda().eval()
    return model, processor, tokenizer

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

def call_deepseek_for_debate(model, processor, tokenizer, image_path, prompt_text):
    """使用Deepseek模型进行debate（需要图片输入）"""
    
    # 记录开始时间
    start_time = time.time()
    
    # 构造多模态对话
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image>\n{prompt_text}",
            "images": [image_path],
        },
        {
            "role": "<|Assistant|>",
            "content": ""
        }
    ]
    
    pil_images = load_pil_images(conversation)

    # 处理输入
    inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(model.device)

    # 调用prepare_inputs_embeds先做图像编码，再 generate
    inputs_embeds = model.prepare_inputs_embeds(**inputs)
    outputs = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True,
        top_p=0.9
    )

    # 解码输出
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
    # 计算耗时
    elapsed_time = time.time() - start_time
    
    return answer.strip(), elapsed_time

def main():
    model_id = "deepseek-vl2-tiny"
    model, processor, tokenizer = load_model_and_processor(f"deepseek-ai/{model_id}")
    
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
                    # 保留所有非deepseek的行
                    if row['Model'] != model_id:
                        all_rows.append(row)
        
        # 创建prompt (deepseek看Qwen和llava的round1结果)
        prompt = create_round2_prompt(
            captions[model_id],
            captions['Qwen2-VL-7B-Instruct'],
            captions['llava-1.5-7b-hf']
        )
        
        # 使用Deepseek模型进行round2 debate（带图片输入）
        final_caption, elapsed_time = call_deepseek_for_debate(
            model, processor, tokenizer, image_path, prompt
        )
        
        # 保存结果
        if file_exists and len(all_rows) > 0:
            # 文件已存在，添加新结果并重写
            all_rows.append({
                'Model': model_id,
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
                writer.writerow([model_id, final_caption, f"{elapsed_time:.2f}"])
        
        print(f"Processed \033[1;33m{scene_id}\033[0m: {model_id} round2 done (time: {elapsed_time:.2f}s)")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
