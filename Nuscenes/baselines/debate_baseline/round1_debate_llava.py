import os
import torch
import csv
from transformers import AutoProcessor, LlavaForConditionalGeneration

torch.cuda.set_device(0)
device = torch.device("cuda:0")

CAPTION_INPUT_DIR = "./Nuscenes/results_caption/"
DEBATE_OUTPUT_DIR = "./Nuscenes/baselines/debate_baseline/results_round1/"
os.makedirs(DEBATE_OUTPUT_DIR, exist_ok=True)

def load_model_and_processor(model_name):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(model_name)
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
                captions[model] = row['Caption']  # 去掉 is None 判断，这样会一直覆盖，最后保留的就是最后一个
    
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

def call_llava_for_debate(model, processor, prompt_text):
    """使用LLaVA模型进行debate（纯文本输入，不需要图片）"""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
            ]
        }
    ]
    
    # 生成符合 LLaVA 格式的 prompt
    prompt = processor.apply_chat_template(
        conversation, 
        add_generation_prompt=True,
        tokenize=False
    )
    
    # 处理输入（纯文本，不需要图片）
    inputs = processor(
        text=prompt,
        return_tensors="pt"
    ).to(device, torch.float16)

    # 生成描述
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True,
        top_k=100,
        top_p=0.9,
    )
    
    # 解码输出
    output_text = processor.decode(
        generated_ids[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
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
    model_name = "llava-1.5-7b-hf"
    model, processor = load_model_and_processor(f"llava-hf/{model_name}")
    
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
        
        # 创建prompt (llava看deepseek和Qwen的结果)
        prompt = create_round1_prompt(
            captions[model_name],
            captions['deepseek-vl2-tiny'],
            captions['Qwen2-VL-7B-Instruct']
        )
        
        # 使用LLaVA模型进行debate
        refined_caption = call_llava_for_debate(model, processor, prompt)
        
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