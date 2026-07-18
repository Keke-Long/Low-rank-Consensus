"""
LLM-based Summarization baseline.

For each scene:
  1) Sample 10 candidate captions (same policy as prior work): deepseek=4, qwen=3, llava=3
  2) Provide all 10 captions to a single summarizer LLM in ONE call
  3) Ask it to produce ONE consensus caption that preserves shared grounded content
     and avoids speculative/inconsistent details.

This script runs ONE summarizer variant at a time (qwen/llava/deepseek) to keep VRAM usage low.

Outputs are written to: Nuscenes/baselines/summarize_baseline/<summarizer>/results/*_final.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Nuscenes.config import model_offload_dir, project_path


DEFAULT_INPUT_DIR = "./Nuscenes/results_caption"
DEFAULT_OUTPUT_BASE_DIR = str(project_path("Nuscenes", "baselines", "summarize_baseline"))


@dataclass(frozen=True)
class CandidateCaption:
    model: str
    caption: str


def parse_scene_file_name(scene_file: str) -> Tuple[str, str]:
    stem = os.path.splitext(os.path.basename(scene_file))[0]
    if "_" not in stem:
        return stem, ""
    scene_id, sample_token = stem.split("_", 1)
    return scene_id, sample_token


def list_scene_csvs(input_dir: str, include_synthetic: bool) -> List[str]:
    csvs = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
    if include_synthetic:
        return csvs
    return [f for f in csvs if "synthetic" not in f.lower()]


def load_scene_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _safe_str(x) -> str:
    if x is None:
        return ""
    s = str(x)
    if s.lower() == "nan":
        return ""
    return s


def build_candidates_for_scene(
    rows: Sequence[Dict[str, str]],
    model_sample_counts: Dict[str, int],
) -> List[CandidateCaption]:
    candidates: List[CandidateCaption] = []
    for model_name, k in model_sample_counts.items():
        picked = 0
        for row in rows:
            if row.get("Model") != model_name:
                continue
            candidates.append(CandidateCaption(model=model_name, caption=_safe_str(row.get("Caption"))))
            picked += 1
            if picked >= k:
                break
        if picked != k:
            raise ValueError(f"Expected {k} captions for model={model_name}, got {picked}")
    return candidates


def format_summarization_prompt(candidates: Sequence[CandidateCaption]) -> str:
    numbered = "\n".join([f"{i+1}. {c.caption}" for i, c in enumerate(candidates)])
    return (
        "You are given 10 candidate captions describing the same autonomous-driving scene.\n"
        "Write ONE consensus caption by preserving only the grounded content that is shared across the candidates.\n"
        "Avoid speculative, uncertain, inconsistent, or overly specific details (e.g., colors/positions/numbers) unless clearly supported by multiple candidates.\n"
        "Keep it concise (<= 30 words). Output ONLY the caption.\n\n"
        f"Candidate captions:\n{numbered}\n\n"
        "Consensus caption:"
    )


class Summarizer:
    def __init__(self, name: str):
        self.name = name

    def summarize(self, prompt: str) -> str:
        raise NotImplementedError


class QwenSummarizer(Summarizer):
    def __init__(self, device: str):
        super().__init__("Qwen2-VL-7B-Instruct")
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self._torch = torch
        self._device = device
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16,
            device_map=device,
            offload_folder=str(model_offload_dir()),
        )
        self._processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    def summarize(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=None, padding=True, return_tensors="pt").to(
            self._torch.device(self._device if self._device.startswith("cuda") else "cpu")
        )
        with self._torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=80, do_sample=False)
        gen = generated[0][inputs["input_ids"].shape[1] :]
        out = self._processor.tokenizer.decode(gen, skip_special_tokens=True).strip()
        return out


class LlavaSummarizer(Summarizer):
    def __init__(self, device: str):
        super().__init__("llava-1.5-7b-hf")
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self._torch = torch
        self._device = device
        self._model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        self._processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    def summarize(self, prompt: str) -> str:
        conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        chat = self._processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self._processor(text=chat, images=None, return_tensors="pt").to(
            self._torch.device(self._device if self._device.startswith("cuda") else "cpu"),
            self._torch.float16,
        )
        with self._torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=80, do_sample=False)
        gen = generated[0][inputs["input_ids"].shape[1] :]
        out = self._processor.decode(gen, skip_special_tokens=True).strip()
        return out


class DeepseekSummarizer(Summarizer):
    def __init__(self, device: str):
        # NOTE: deepseek-vl2-tiny requires a transformers version that provides
        # LlamaFlashAttention2; the current environment is incompatible.
        # For text-only summarization, we use a DeepSeek text chat model instead.
        super().__init__("deepseek-llm-7b-chat")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat", use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-llm-7b-chat",
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        self._model.eval()

    def summarize(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are an autonomous-driving assistant."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._tokenizer(text, return_tensors="pt")
        else:
            text = "System: You are an autonomous-driving assistant.\nUser: " + prompt + "\nAssistant:"
            inputs = self._tokenizer(text, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        with self._torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        gen = generated[0][inputs["input_ids"].shape[1] :]
        out = self._tokenizer.decode(gen, skip_special_tokens=True).strip()
        return out


def make_summarizer(kind: str, device: str) -> Summarizer:
    kind = kind.lower().strip()
    if kind == "qwen":
        return QwenSummarizer(device=device)
    if kind == "llava":
        return LlavaSummarizer(device=device)
    if kind == "deepseek":
        return DeepseekSummarizer(device=device)
    raise ValueError(f"Unknown summarizer kind: {kind}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_one_row_csv(path: str, model_name: str, caption: str, elapsed_s: float) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Caption", "Time", "hallucination_check"])
        w.writerow([model_name, caption, f"{elapsed_s:.2f}", ""])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarizer", required=True, choices=["qwen", "llava", "deepseek"])
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-base-dir", default=DEFAULT_OUTPUT_BASE_DIR)
    ap.add_argument("--limit-scenes", type=int, default=1)
    ap.add_argument("--include-synthetic", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--deepseek-k", type=int, default=4)
    ap.add_argument("--qwen-k", type=int, default=3)
    ap.add_argument("--llava-k", type=int, default=3)
    args = ap.parse_args()

    model_sample_counts = {
        "deepseek-vl2-tiny": args.deepseek_k,
        "Qwen2-VL-7B-Instruct": args.qwen_k,
        "llava-1.5-7b-hf": args.llava_k,
    }
    if sum(model_sample_counts.values()) != 10:
        raise ValueError(f"Expected 10 candidates total, got {sum(model_sample_counts.values())}")

    input_dir = args.input_dir
    output_dir = os.path.join(args.output_base_dir, args.summarizer, "results")
    ensure_dir(output_dir)

    scene_files = list_scene_csvs(input_dir, include_synthetic=args.include_synthetic)
    if args.limit_scenes is not None and args.limit_scenes > 0:
        scene_files = scene_files[: args.limit_scenes]

    summarizer = make_summarizer(args.summarizer, device=args.device)

    processed = 0
    for scene_file in scene_files:
        scene_id, sample_token = parse_scene_file_name(scene_file)
        out_path = os.path.join(output_dir, f"{scene_id}_{sample_token}_final.csv")
        if os.path.exists(out_path) and not args.overwrite:
            continue

        rows = load_scene_rows(os.path.join(input_dir, scene_file))
        candidates = build_candidates_for_scene(rows, model_sample_counts)
        prompt = format_summarization_prompt(candidates)

        t0 = time.time()
        caption = summarizer.summarize(prompt)
        elapsed = time.time() - t0

        write_one_row_csv(out_path, summarizer.name, caption, elapsed)
        processed += 1
        print(f"[{args.summarizer}] wrote {out_path}")

    print(f"Done. processed_files={processed}, output_dir={output_dir}")


if __name__ == "__main__":
    main()
