import os
import torch
import cv2
import io
import csv
import argparse
import sys
from pathlib import Path
from PIL import Image
from datetime import datetime
from nuscenes.nuscenes import NuScenes
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Nuscenes.config import nuscenes_dir
from transformers import AutoProcessor, LlavaForConditionalGeneration

# 设置 CUDA 设备
torch.cuda.set_device(0)
device = torch.device("cuda:0")

# 路径
NUSCENES_DIR = str(nuscenes_dir())
nusc = NuScenes(version="v1.0-trainval", dataroot=NUSCENES_DIR, verbose=False)

CAPTION_OUTPUT_DIR = "./Nuscenes/results_caption/"


def load_model_and_processor(model_name):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def compress_image(image_path):
    """读取并压缩图片，返回 PIL.Image 对象"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
    # img = cv2.resize(img, (640, 360))  # 降低分辨率
    return Image.fromarray(img)  # 直接返回 PIL 图像


def generate_response(model, processor, image, text_query, generation_args):
    """使用 LLaVA 模型生成响应"""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_query},
                {"type": "image"},
            ]
        }
    ]
    
    # 生成符合 LLaVA 格式的 prompt
    prompt = processor.apply_chat_template(
        conversation, 
        add_generation_prompt=True,
        tokenize=False
    )
    
    # 处理输入
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(device, torch.float16)

    # 生成描述
    generated_ids = model.generate(
        **inputs,
        **generation_args
    )
    
    # 解码输出
    return processor.decode(
        generated_ids[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    ).strip()


def count_existing_rows(csv_filename):
    """检查 CSV 文件中已有的描述数量"""
    if not os.path.isfile(csv_filename):
        return 0
    with open(csv_filename, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1  # 减去表头行


def count_existing_rows_for_model_and_temp(csv_filename, model_name, temperature):
    """检查 CSV 文件中当前模型+temperature 的记录数量"""
    if not os.path.isfile(csv_filename):
        return 0
    with open(csv_filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return 0       
        if "temperature" not in reader.fieldnames:
            return sum(1 for row in reader if row.get("Model") == model_name)
        target_temp = str(float(temperature))
        return sum(
            1
            for row in reader
            if row.get("Model") == model_name
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

    model_name = "llava-1.5-7b-hf"
    model, processor = load_model_and_processor(f"llava-hf/{model_name}")

    text_query = (
        #"This is an image captured from the front camera of ego vehicle. "
        "Please describe all the other vehicles in the scene. "
        "Do not describe pedestrians, bicycles, weather conditions, buildings, or any unrelated details. "
        "Provide a clear and concise description within 50 words."
    )

    generation_args = {
        "max_new_tokens": 100,
        "temperature": args.temperature,
        "do_sample": True,
        "top_k": 100,
        "top_p": 0.8,
    }

    scene_end = args.scene_start + args.scene_count
    for scene in nusc.scene[args.scene_start:scene_end]:
        scene_idx = scene['name'][6:]  # 去掉前6个字符，只保留数字部分
        first_sample_token = scene['first_sample_token']
        sample = nusc.get('sample', first_sample_token)
        cam_name = "CAM_FRONT"
        cam_data = nusc.get('sample_data', sample['data'][cam_name])
        image_path = os.path.join(NUSCENES_DIR, cam_data['filename'])

        # 读取 CSV，检查已有的行数
        csv_filename = os.path.join(args.output_dir, f"{scene_idx}_{first_sample_token}.csv")
        existing_rows = count_existing_rows_for_model_and_temp(
            csv_filename, model_name, args.temperature
        )

        # 如果已有足够条数，就跳过
        if existing_rows >= args.target_captions:
            print(
                f"\033[1;32mSkipping {first_sample_token}, already has "
                f"{args.target_captions} captions.\033[0m"
            )
            continue

        # 需要补充的描述数量
        num_needed = args.target_captions - existing_rows

        # 读取并压缩图片
        pil_image = compress_image(image_path)

        # 生成描述
        with open(csv_filename, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            if existing_rows == 0:
                writer.writerow(["Model", "Timestamp", "Caption", "temperature"])

            for _ in range(num_needed):
                caption = generate_response(
                    model=model,
                    processor=processor,
                    image=pil_image,
                    text_query=text_query,
                    generation_args=generation_args
                )
                timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
                writer.writerow([model_name, timestamp, caption, args.temperature])
                print(f"Processed \033[1;33m{scene_idx}\033[0m: {caption}")

        # 清理 CUDA 缓存
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
