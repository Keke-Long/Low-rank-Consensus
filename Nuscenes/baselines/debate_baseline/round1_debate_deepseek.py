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

CAPTION_INPUT_DIR = "./Nuscenes/results_caption/"
DEBATE_OUTPUT_DIR = "./Nuscenes/baselines/debate_baseline/results_round1/"
os.makedirs(DEBATE_OUTPUT_DIR, exist_ok=True)

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
    # CSV格式: {scene_idx}_{sample_token}.csv
    parts = csv_filename.replace('.csv', '').split('_')
    scene_idx = parts[0]
    sample_token = '_'.join(parts[1:])
    
    # 获取图片路径
    sample = nusc.get('sample', sample_token)
    cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    image_path = os.path.join(NUSCENES_DIR, cam_data['filename'])
    
    return image_path

def read_last_captions(csv_path):
    """读取三个模型的最后一个caption"""
    captions = {
        'deepseek-vl2-tiny': None,
        'Qwen2-VL-7B-Instruct': None,
        'llava-1.5-7b-hf': None
    }
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['Model']
            if model in captions:
                captions[model] = row['Caption']
    
    return captions

def create_round1_prompt(original_caption, other_caption_1, other_caption_2):
    """构造Round 1 prompt"""
    return f"""You are evaluating a driving scene description for potential hallucinations.

Your Original Description: {original_caption}

Descriptions from Other Vision-Language Models:
Model 2: {other_caption_1}
Model 3: {other_caption_2}

Task: Compare your description with the other two models' descriptions. Check if your description contains:
1. Objects that are NOT mentioned by the other models (potential hallucination)
2. Incorrect spatial relationships or positions
3. Wrong attributes (color, size, type) that contradict the other descriptions
4. Any details that seem inconsistent with the consensus

Instructions:
* If you find potential hallucinations in your description, provide a CORRECTED version that aligns better with the consensus.
* If your description is accurate and the inconsistencies are in the other descriptions, keep your original description.
* Focus on factual accuracy, not stylistic differences.

Output Format: Provide your refined description directly, without explanation."""

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
    
    # 遍历所有已有的caption CSV文件
    csv_files = sorted([f for f in os.listdir(CAPTION_INPUT_DIR) if f.endswith('.csv')])
    
    for csv_file in csv_files:
        scene_id = csv_file.replace('.csv', '')
        
        # 获取图片路径
        try:
            image_path = get_image_path_from_csv(csv_file)
        except Exception as e:
            print(f"Error getting image path for {csv_file}: {e}")
            continue
        
        # 读取原始captions (每个模型的最后一个)
        original_csv = os.path.join(CAPTION_INPUT_DIR, csv_file)
        
        captions = read_last_captions(original_csv)
        
        # 检查是否所有模型都有caption
        if None in captions.values():
            print(f"Skipping {scene_id}: missing captions")
            continue
        
        # 检查debate结果文件
        debate_csv = os.path.join(DEBATE_OUTPUT_DIR, f"{scene_id}_debate.csv")
        
        # 读取已存在的行（如果文件存在）
        file_exists = os.path.exists(debate_csv)
        all_rows = []
        if file_exists:
            with open(debate_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 保留所有非deepseek-round1的行
                    if not (row['Model'] == model_id and row['Stage'] == 'round1'):
                        all_rows.append(row)
        
        # 创建prompt (deepseek看Qwen和llava的结果)
        prompt = create_round1_prompt(
            captions[model_id],
            captions['Qwen2-VL-7B-Instruct'],
            captions['llava-1.5-7b-hf']
        )
        
        # 使用Deepseek模型进行debate（带图片输入）
        refined_caption, elapsed_time = call_deepseek_for_debate(
            model, processor, tokenizer, image_path, prompt
        )
        
        # 保存结果
        if file_exists and len(all_rows) > 0:
            # 文件已存在，添加新结果并重写
            all_rows.append({
                'Model': model_id,
                'Stage': 'round1',
                'Caption': refined_caption,
                'Time': f"{elapsed_time:.2f}"
            })
            
            with open(debate_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['Model', 'Stage', 'Caption', 'Time'])
                writer.writeheader()
                writer.writerows(all_rows)
        else:
            # 新文件，写入original + round1
            with open(debate_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Model', 'Stage', 'Caption', 'Time'])
                # 写入original captions (没有时间记录)
                for model_name, caption in captions.items():
                    writer.writerow([model_name, 'original', caption, ''])
                # 写入round1 refined caption (有时间记录)
                writer.writerow([model_id, 'round1', refined_caption, f"{elapsed_time:.2f}"])
        
        print(f"Processed \033[1;33m{scene_id}\033[0m: {model_id} round1 done (time: {elapsed_time:.2f}s)")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
