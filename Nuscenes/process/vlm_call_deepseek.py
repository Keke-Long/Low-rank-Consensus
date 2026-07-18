import os
import torch
import cv2
import csv
import argparse
import sys
from pathlib import Path
from PIL import Image
from datetime import datetime
from nuscenes.nuscenes import NuScenes
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Nuscenes.config import nuscenes_dir

from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

import transformers
transformers.logging.set_verbosity_error()

# # 设置 CUDA 设备
# torch.cuda.set_device(1)
# device = torch.device("cuda:1")

# 设定 NuScenes 数据路径
NUSCENES_DIR = str(nuscenes_dir())
nusc = NuScenes(version="v1.0-trainval", dataroot=NUSCENES_DIR, verbose=False)

# 设置输出 CSV 存储路径
CAPTION_OUTPUT_DIR = "./Nuscenes/results_caption/"
os.makedirs(CAPTION_OUTPUT_DIR, exist_ok=True)

# 设置临时图片保存路径
TEMP_IMAGE_DIR = "./temp_images/"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)


def load_model_and_processor(model_path):
    processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 
    ).cuda().eval()
    return model, processor, tokenizer


def compress_image(image_path, save_path):
    """读取图片并保存到 TEMP_IMAGE_DIR，同时返回 PIL.Image 对象"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
    # img = cv2.resize(img, (640, 360))  # 如需缩小尺寸，可以解开这行代码

    pil_img = Image.fromarray(img)
    pil_img.save(save_path)
    
    return pil_img


def generate_response(model, processor, tokenizer, img_path):

    system_prompt = (
        "You are an autonomous‑driving captioning assistant, you need to generate the caption of the image to help understanding the traffic scenario "
    )

    # 构造多模态对话
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\n<|ref|> Output several short sentences (<30 words) to describe the other road users, especially vehicles.<|/ref|>.",
            "images": [img_path], 
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
        system_prompt=system_prompt,
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
        max_new_tokens=512,
        temperature=0.4,
        do_sample=True,
        top_p=0.9
    )

    # 解码输出
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer


def count_existing_rows(csv_filename):
    """检查 CSV 文件中已有的描述数量"""
    if not os.path.isfile(csv_filename):
        return 0
    with open(csv_filename, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1  # 减去表头行


def count_existing_rows_for_model_and_temp(csv_filename, model_id, temperature):
    """检查 CSV 文件中当前模型+temperature 的记录数量"""
    if not os.path.isfile(csv_filename):
        return 0
    with open(csv_filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return 0
        if "temperature" not in reader.fieldnames:
            return sum(1 for row in reader if row.get("Model") == model_id)
        target_temp = str(float(temperature))
        return sum(
            1
            for row in reader
            if row.get("Model") == model_id
            and str(float(row.get("temperature", 0.0))) == target_temp
        )
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--target-captions", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=CAPTION_OUTPUT_DIR)
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-count", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_id = "deepseek-vl2-tiny"
    model, processor,tokenizer = load_model_and_processor(f"deepseek-ai/{model_id}")

    scene_end = args.scene_start + args.scene_count
    for scene in nusc.scene[args.scene_start:scene_end]:
        scene_idx = scene['name'][6:]  # 去掉前6个字符，只保留数字部分
        first_sample_token = scene['first_sample_token']
        sample = nusc.get('sample', first_sample_token)
        cam_name = "CAM_FRONT"
        cam_data = nusc.get('sample_data', sample['data'][cam_name])
        image_path = os.path.join(NUSCENES_DIR, cam_data['filename'])

        csv_filename = os.path.join(args.output_dir, f"{scene_idx}_{first_sample_token}.csv")

        # 设置 **图片保存路径**
        temp_image_path = os.path.join(TEMP_IMAGE_DIR, f"{scene_idx}_{first_sample_token}.jpg") 

        existing_rows = count_existing_rows_for_model_and_temp(
            csv_filename, model_id, args.temperature
        )

        # 如果已有足够条数，就跳过
        if existing_rows >= args.target_captions:
            print(
                f"\033[1;32mSkipping {first_sample_token}, already has "
                f"{args.target_captions} captions for model {model_id}.\033[0m"
            )
            continue

        num_needed = args.target_captions - existing_rows
        print(f"Processing scene idx={scene_idx}, token={first_sample_token}, needed={num_needed} ...")

        # 读取 PIL.Image
        # pil_image = compress_image(image_path, temp_image_path)

        # 生成描述
        with open(csv_filename, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            if existing_rows == 0:
                writer.writerow(["Model", "Timestamp", "Caption", "temperature"])

            for _ in range(num_needed):
                caption = generate_response(
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    img_path = image_path
                )
                timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
                writer.writerow([model_id, timestamp, caption, args.temperature])
                print(f"Processed \033[1;33m{scene_idx}\033[0m: {caption}")
        
        # 清理 CUDA 缓存
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
