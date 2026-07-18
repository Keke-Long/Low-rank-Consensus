import os
import torch
import cv2
import io
import csv
import argparse
import sys
from pathlib import Path
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datetime import datetime
from nuscenes.nuscenes import NuScenes
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Nuscenes.config import model_offload_dir, nuscenes_dir

# 设定 NuScenes mini 数据路径
NUSCENES_DIR = str(nuscenes_dir())
nusc = NuScenes(version='v1.0-trainval', dataroot=NUSCENES_DIR, verbose=False)

CAPTION_OUTPUT_DIR = "./Nuscenes/results_caption/"
os.makedirs(CAPTION_OUTPUT_DIR, exist_ok=True)


def load_model_and_processor(model_id, device_str):
    device_index = int(device_str.split(":")[1]) if ":" in device_str else 0
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map=device_str,
            offload_folder=str(model_offload_dir()),
            max_memory={device_index: "24GB"}
        )
    except ImportError:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device_str,
            offload_folder=str(model_offload_dir()),
            max_memory={device_index: "24GB"}
        )
    min_pixels = 64 * 28 * 28
    max_pixels = 512 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    return model, processor


def compress_image(image_path):
    """读取并压缩图片，返回 BytesIO 对象"""
    img = cv2.imread(image_path)
    #img = cv2.resize(img, (640, 360))  # 降低分辨率
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]  # 设置 JPEG 质量为 30（降低质量）
    _, compressed_img = cv2.imencode(".jpg", img, encode_param)  # 压缩
    return io.BytesIO(compressed_img.tobytes())


def generate_response(model, processor, image_data, text_query, generation_args, device):
    """使用 BytesIO 传递压缩图片，修正 TypeError"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_data)}
            ] + [
                {"type": "text", "text": text_query}
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
    )
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, **generation_args)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_texts[0].strip()


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
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.cuda.set_device(int(args.device.split(":")[1]))

    model_name = "Qwen2-VL-7B-Instruct"
    model, processor = load_model_and_processor(f"Qwen/{model_name}", args.device)

    text_query = (
        "This is an image captured from the front camera of a moving vehicle. "
        "Please describe the vehicles in the scene within 40 words."
        "Do not describe pedestrians, bicycles, weather conditions, buildings, or any unrelated details. "
    )

    generation_args = {
        "max_new_tokens": 100,
        "temperature": args.temperature,
        "do_sample": True,
        "top_k": 100,
        "top_p": 0.8,
        "num_return_sequences": 1,
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
        compressed_image = compress_image(image_path)

        # 生成描述
        with open(csv_filename, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            if existing_rows == 0:
                writer.writerow(["Model", "Timestamp", "Caption", "temperature"])

            for _ in range(num_needed):
                caption = generate_response(
                    model, processor, compressed_image, text_query, generation_args, torch.device(args.device)
                )
                timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
                writer.writerow([model_name, timestamp, caption, args.temperature])
                print(f"Processed \033[1;33m{scene_idx}\033[0m: {caption}")

        # 清理 CUDA 缓存
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
