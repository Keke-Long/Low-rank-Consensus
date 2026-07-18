# A Low-Rank Consensus Method for Vision Language Model Hallucination Mitigation in Autonomous Driving

This repository contains research code for selecting reliable autonomous-driving scene captions from multiple vision-language models (VLMs).

## Core idea

For each scene, we encode candidate captions and stack the embeddings into a matrix $X$. A truncated SVD gives a low-rank reconstruction $X_r$, representing the semantic content shared across models. We use the reconstruction residual

$$
s_i = \lVert X_i - (X_r)_i \rVert_2
$$

as a consensus-based reliability score for caption $i$: lower residual means closer agreement with the shared semantic components. The code includes fixed-rank and dynamic-rank variants, and uses captions from DeepSeek-VL2, LLaVA, and Qwen2-VL on NuScenes scenes.

## Use

Install the project dependencies, obtain NuScenes independently, then copy `.env.example` to `.env` and set:

```bash
NUSCENES_DIR=/path/to/nuScenes
```

Run scripts from the repository root. The main implementation is in `Nuscenes/process/`; `Nuscenes/config.py` loads local paths without exposing them in source code.

> This repository provides code only. NuScenes data, generated captions, derived caches, and full experiment outputs are not distributed.

## Citation and license

Citation information and an open-source license will be added when the associated manuscript is finalized.
