import os
import torch
import csv
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

torch.cuda.set_device(1)
device = torch.device("cuda:1")

CAPTION_INPUT_DIR = "./Nuscenes/results_caption/"
DEBATE_OUTPUT_DIR = "./Nuscenes/baselines/debate_baseline/results_round1/"
os.makedirs(DEBATE_OUTPUT_DIR, exist_ok=True)

def load_model_and_processor(model_id):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="cuda:1",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

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

def call_qwen_for_debate(model, processor, prompt_text):
    """使用Qwen模型进行debate（纯文本输入，不需要图片）"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text],
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
    
    return output_text

def check_existing_round1_result(debate_csv, model_name):
    """检查round1结果是否已存在"""
    if not os.path.exists(debate_csv):
        return False
    
    with open(debate_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Model'] == model_name and row['Stage'] == 'round1':
                return True
    return False

def main():
    model_name = "Qwen2-VL-7B-Instruct"
    model, processor = load_model_and_processor(f"Qwen/{model_name}")
    
    # 遍历所有已有的caption CSV文件
    csv_files = sorted([f for f in os.listdir(CAPTION_INPUT_DIR) if f.endswith('.csv')])
    
    for csv_file in csv_files:
        scene_id = csv_file.replace('.csv', '')
        
        # 读取原始captions (每个模型的第一个)
        original_csv = os.path.join(CAPTION_INPUT_DIR, csv_file)
        
        captions = read_last_captions(original_csv)
        
        # 检查是否所有模型都有caption
        if None in captions.values():
            print(f"Skipping {scene_id}: missing captions")
            continue
        
        # 检查debate结果文件
        debate_csv = os.path.join(DEBATE_OUTPUT_DIR, f"{scene_id}_debate.csv")
        
        # 如果已经生成过round1结果，跳过
        if check_existing_round1_result(debate_csv, model_name):
            print(f"\033[1;32mSkipping {scene_id}: {model_name} round1 already exists\033[0m")
            continue
        
        # 创建prompt (Qwen看deepseek和llava的结果)
        prompt = create_round1_prompt(
            captions[model_name],
            captions['deepseek-vl2-tiny'],
            captions['llava-1.5-7b-hf']
        )
        
        # 使用Qwen模型进行debate
        refined_caption = call_qwen_for_debate(model, processor, prompt)
        
        # 保存结果
        file_exists = os.path.exists(debate_csv)
        with open(debate_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Model', 'Stage', 'Caption'])
                # 写入original captions
                for model_id, caption in captions.items():
                    writer.writerow([model_id, 'original', caption])
            
            # 写入round1 refined caption
            writer.writerow([model_name, 'round1', refined_caption])
        
        print(f"Processed \033[1;33m{scene_id}\033[0m: {model_name} round1 done")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()