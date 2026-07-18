#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image


Kind = Literal["llava", "qwen2vl", "deepseek-vl2"]


@dataclass
class RunResult:
    run_index: int
    seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int


@dataclass
class Summary:
    kind: str
    model_id: str
    device: str
    dtype: str
    image_path: str
    prompt: str
    system_prompt: str
    generation_args: dict[str, Any]
    warmup_runs: int
    measured_runs: int
    baseline_allocated_bytes: int
    baseline_reserved_bytes: int
    runs: list[RunResult]
    avg_peak_allocated_bytes: float
    avg_peak_reserved_bytes: float
    avg_peak_allocated_delta_bytes: float
    avg_peak_reserved_delta_bytes: float


def _bytes_to_gib(value: float) -> float:
    return float(value) / (1024.0**3)


def _torch_dtype(dtype_str: str) -> torch.dtype:
    normalized = dtype_str.lower().replace("-", "").replace("_", "")
    if normalized in ("fp16", "float16"):
        return torch.float16
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _require_cuda(device_index: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this environment (torch.cuda.is_available() == False). "
            "Run this script on a machine with an NVIDIA GPU + driver + CUDA runtime."
        )
    torch.cuda.set_device(device_index)
    return torch.device(f"cuda:{device_index}")


def _reset_peaks(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def _get_baseline(device: torch.device) -> tuple[int, int]:
    torch.cuda.synchronize(device)
    return (
        int(torch.cuda.memory_allocated(device)),
        int(torch.cuda.memory_reserved(device)),
    )


def _measure_generate(
    *,
    device: torch.device,
    generate_fn,
) -> tuple[float, int, int]:
    _reset_peaks(device)
    start = time.perf_counter()
    with torch.inference_mode():
        generate_fn()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    peak_alloc = int(torch.cuda.max_memory_allocated(device))
    peak_resv = int(torch.cuda.max_memory_reserved(device))
    return seconds, peak_alloc, peak_resv


def _load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _llava_runner(
    *,
    model_id: str,
    dtype: torch.dtype,
    device: torch.device,
    image: Image.Image,
    prompt: str,
    generation_args: dict[str, Any],
):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        }
    ]
    chat_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    def generate_once() -> None:
        inputs = processor(text=chat_prompt, images=image, return_tensors="pt")
        inputs = inputs.to(device, dtype=dtype)
        _ = model.generate(**inputs, **generation_args)

    return generate_once


def _qwen2vl_runner(
    *,
    model_id: str,
    dtype: torch.dtype,
    device: torch.device,
    image: Image.Image,
    prompt: str,
    generation_args: dict[str, Any],
    attn_implementation: str | None,
):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    try:
        from qwen_vl_utils import process_vision_info
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import qwen_vl_utils.process_vision_info. "
            "Make sure Qwen2-VL dependencies are installed in this environment."
        ) from e

    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    def generate_once() -> None:
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        _ = model.generate(**inputs, **generation_args)

    return generate_once


def _deepseek_vl2_runner(
    *,
    model_id: str,
    dtype: torch.dtype,
    device: torch.device,
    image_path: str,
    prompt: str,
    generation_args: dict[str, Any],
    system_prompt: str,
):
    from transformers import AutoModelForCausalLM

    try:
        from deepseek_vl2.models import DeepseekVLV2Processor
        from deepseek_vl2.utils.io import load_pil_images
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import deepseek_vl2. Make sure DeepSeek-VL2 dependencies are installed."
        ) from e

    processor = DeepseekVLV2Processor.from_pretrained(model_id)
    tokenizer = processor.tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\n<|ref|> " + prompt + "<|/ref|>.",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    pil_images = load_pil_images(conversation)
    inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=system_prompt,
    ).to(device)

    def generate_once() -> None:
        inputs_embeds = model.prepare_inputs_embeds(**inputs)
        _ = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **generation_args,
        )

    return generate_once


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure VLM inference peak VRAM (PyTorch CUDA).")
    parser.add_argument(
        "--kind",
        required=True,
        choices=["llava", "qwen2vl", "deepseek-vl2"],
        help="Model family / runner implementation.",
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path.")
    parser.add_argument(
        "--image",
        default="information_decomposition.jpg",
        help="Path to an image file used for inference.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--dtype", default="auto", help="auto|fp16|bf16|fp32")
    parser.add_argument("--runs", type=int, default=4, help="Measured runs (average over these).")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (not counted).")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling in generation.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--prompt", default="")
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Optional system prompt override (DeepSeek-VL2 uses this field).",
    )
    parser.add_argument("--out", default="", help="Write JSON summary to this path.")
    parser.add_argument(
        "--attn-implementation",
        default="",
        help='Qwen2-VL only, e.g. "flash_attention_2" (leave empty for default).',
    )
    args = parser.parse_args()

    device = _require_cuda(args.device)

    image_path = str(Path(args.image))
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    kind: Kind = args.kind  # type: ignore[assignment]

    if args.dtype == "auto":
        dtype = torch.bfloat16 if kind == "deepseek-vl2" else torch.float16
        dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    else:
        dtype = _torch_dtype(args.dtype)
        dtype_str = args.dtype.lower()

    if kind == "deepseek-vl2":
        # Match `Nuscenes/process/vlm_call_deepseek.py` defaults as closely as possible.
        default_generation_args: dict[str, Any] = {
            "max_new_tokens": 512,
            "temperature": 0.4,
            "do_sample": True,
            "top_p": 0.9,
            "use_cache": True,
        }
    else:
        default_generation_args = {
            "max_new_tokens": 64,
            "do_sample": False,
            "use_cache": True,
        }

    generation_args: dict[str, Any] = dict(default_generation_args)

    if args.max_new_tokens != 64:
        generation_args["max_new_tokens"] = int(args.max_new_tokens)
    if args.do_sample:
        generation_args["do_sample"] = True
    if args.temperature is not None:
        generation_args["temperature"] = float(args.temperature)
    if args.top_p is not None:
        generation_args["top_p"] = float(args.top_p)
    if args.top_k is not None:
        generation_args["top_k"] = int(args.top_k)

    image = _load_image(image_path)

    if kind == "llava":
        resolved_prompt = args.prompt or "Describe the image briefly and accurately."
        resolved_system_prompt = ""
        generate_once = _llava_runner(
            model_id=args.model,
            dtype=dtype,
            device=device,
            image=image,
            prompt=resolved_prompt,
            generation_args=generation_args,
        )
    elif kind == "qwen2vl":
        resolved_prompt = args.prompt or "Describe the image briefly and accurately."
        resolved_system_prompt = ""
        generate_once = _qwen2vl_runner(
            model_id=args.model,
            dtype=dtype,
            device=device,
            image=image,
            prompt=resolved_prompt,
            generation_args=generation_args,
            attn_implementation=(args.attn_implementation or None),
        )
    else:
        default_system_prompt = (
            "You are an autonomous‑driving captioning assistant, you need to generate the caption of the image to help understanding the traffic scenario "
        )
        default_prompt = (
            "Output several short sentences (<30 words) to describe the other road users, especially vehicles."
        )
        resolved_prompt = args.prompt or default_prompt
        resolved_system_prompt = args.system_prompt or default_system_prompt
        generate_once = _deepseek_vl2_runner(
            model_id=args.model,
            dtype=dtype,
            device=device,
            image_path=image_path,
            prompt=resolved_prompt,
            generation_args=generation_args,
            system_prompt=resolved_system_prompt,
        )

    baseline_alloc, baseline_resv = _get_baseline(device)

    for _ in range(int(args.warmup)):
        _ = _measure_generate(device=device, generate_fn=generate_once)

    results: list[RunResult] = []
    for i in range(int(args.runs)):
        seconds, peak_alloc, peak_resv = _measure_generate(device=device, generate_fn=generate_once)
        results.append(
            RunResult(
                run_index=i,
                seconds=seconds,
                peak_allocated_bytes=peak_alloc,
                peak_reserved_bytes=peak_resv,
            )
        )

    avg_peak_alloc = sum(r.peak_allocated_bytes for r in results) / len(results)
    avg_peak_resv = sum(r.peak_reserved_bytes for r in results) / len(results)
    avg_peak_alloc_delta = avg_peak_alloc - baseline_alloc
    avg_peak_resv_delta = avg_peak_resv - baseline_resv

    summary = Summary(
        kind=kind,
        model_id=args.model,
        device=str(device),
        dtype=dtype_str,
        image_path=image_path,
        prompt=resolved_prompt,
        system_prompt=resolved_system_prompt,
        generation_args=generation_args,
        warmup_runs=int(args.warmup),
        measured_runs=int(args.runs),
        baseline_allocated_bytes=baseline_alloc,
        baseline_reserved_bytes=baseline_resv,
        runs=results,
        avg_peak_allocated_bytes=avg_peak_alloc,
        avg_peak_reserved_bytes=avg_peak_resv,
        avg_peak_allocated_delta_bytes=avg_peak_alloc_delta,
        avg_peak_reserved_delta_bytes=avg_peak_resv_delta,
    )

    payload = asdict(summary)
    payload["baseline_allocated_gib"] = _bytes_to_gib(baseline_alloc)
    payload["baseline_reserved_gib"] = _bytes_to_gib(baseline_resv)
    payload["avg_peak_allocated_gib"] = _bytes_to_gib(avg_peak_alloc)
    payload["avg_peak_reserved_gib"] = _bytes_to_gib(avg_peak_resv)
    payload["avg_peak_allocated_delta_gib"] = _bytes_to_gib(avg_peak_alloc_delta)
    payload["avg_peak_reserved_delta_gib"] = _bytes_to_gib(avg_peak_resv_delta)

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
