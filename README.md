# Low-Rank Consensus for VLM Hallucination Mitigation in Autonomous Driving

**Official research code** for the paper:

**Keke Long, Jiacheng Guo, Tianyun Zhang, Hongkai Yu, and Xiaopeng Li.** "A low-rank consensus method for vision language model hallucination mitigation in autonomous driving." *Transportation Research Part C*, 194 (2027), 106092. [Paper / DOI](https://doi.org/10.1016/j.trc.2026.106092)

This work provides a lightweight, **training-free and reference-free caption selection method** for reducing hallucinations in vision-language model (VLM) descriptions of driving scenes. Given multiple candidate descriptions of the same image, it estimates their shared semantic structure and selects the caption with the smallest deviation from that consensus. The selection step needs only the candidate captions, not the original image, ground-truth labels, or internal VLM representations.

<!-- Figure 2 from the paper. Add the image to assets/figure2.png to display it here. -->
**Figure 2 (overview of the proposed method):** The diagram will be added when the figure image is available. Please place it at `assets/figure2.png`. It illustrates candidate VLM captions being embedded, decomposed into a low-rank consensus and residual, and ranked by residual magnitude.

## Highlights

- **Black-box compatibility:** Works with captions from one or multiple VLMs, without modifying or retraining their parameters.
- **Low-rank semantic consensus:** Uses sentence embeddings, truncated singular value decomposition (SVD), and caption-wise residual norms.
- **Adaptive rank:** Uses a gated spectral-gap rule to choose the retained rank for each scene.
- **Paper results:** 87% hallucination-free caption selection accuracy on the evaluated NuScenes scenes in the heterogeneous multi-VLM setting, versus 79%–82% for the tested multi-agent debate baselines. Reported inference time was 0.84 s per scene, compared with 1.72–2.56 s for debate under the paper's evaluation conditions. These are paper results, not measurements from a new checkout.

## Method

Given `n` candidate captions for one scene, encode each caption as a `d`-dimensional vector and stack them into the matrix `M`:

```text
Candidate captions -> Sentence embeddings (M)
                   -> Truncated SVD -> Low-rank reconstruction (R)
                   -> Residual E = M - R
                   -> Score each caption by ||E[i]||_2
                   -> Select the caption with the smallest score
```

The rank is selected using a **gated spectral gap**: if the first-to-second singular-value ratio is at least `4.75`, choose rank 1. Otherwise, choose the largest spectral gap among subsequent leading singular values, subject to maximum rank `3`. This is the policy implemented in the main experiment script.

A smaller residual indicates **stronger agreement with the candidate set**, not a guarantee of factual correctness. Shared hallucinations may enter the consensus, and correct but unusual details may receive larger residual scores.

## Where is the core algorithm?

| File | Purpose |
| --- | --- |
| **[`Nuscenes/process/gated_spectral_gap_select.py`](Nuscenes/process/gated_spectral_gap_select.py)** | **Main proposed method:** adaptive rank selection, SVD reconstruction, residual scoring, final caption selection, and evaluation summaries. |
| [`Nuscenes/process/cal_rank.py`](Nuscenes/process/cal_rank.py) | Earlier fixed-rank and multi-rank SVD analysis. |
| [`Nuscenes/process/large_vlm_low_rank_select_r1.py`](Nuscenes/process/large_vlm_low_rank_select_r1.py) | Rank-1 comparison. |
| [`Nuscenes/process/large_vlm_low_rank_select_r3.py`](Nuscenes/process/large_vlm_low_rank_select_r3.py) | Rank-3 comparison. |
| [`Nuscenes/process/pairwise_similarity_consensus.py`](Nuscenes/process/pairwise_similarity_consensus.py) | Pairwise cosine-similarity consensus baseline. |
| [`Nuscenes/process/vlm_call_deepseek.py`](Nuscenes/process/vlm_call_deepseek.py) | DeepSeek-VL2 candidate generation. |
| [`Nuscenes/process/vlm_call_llava.py`](Nuscenes/process/vlm_call_llava.py) | LLaVA candidate generation. |
| [`Nuscenes/process/vlm_call_Qwen.py`](Nuscenes/process/vlm_call_Qwen.py) | Qwen2-VL candidate generation. |
| [`Nuscenes/process/hallucination_check.py`](Nuscenes/process/hallucination_check.py) | Evaluation-label generation with scene images and detection facts; **not required for inference-time caption selection**. |
| [`Nuscenes/baselines/debate_baseline/`](Nuscenes/baselines/debate_baseline/) | Multi-agent debate baseline scripts. |

Other analysis and visualization scripts are under `Nuscenes/results/` and `tools/`.

## Getting started

The **low-rank selection stage** is much simpler to run than the full experiment: it operates on **already generated caption CSVs**. VLM inference and ground-truth evaluation are separate stages.

1. Clone the repository and create a Python environment. `environment.yaml` records the original research environment; some pinned CUDA/PyTorch dependencies are platform-specific, so a fresh installation may require adjustments.
2. Install the Python dependencies used by the selection stage, including `numpy`, `pandas`, `torch`, and `sentence-transformers`. The sentence encoder used for the experiments is [`intfloat/e5-large-v2`](https://huggingface.co/intfloat/e5-large-v2).
3. Supply one CSV file per scene in `Nuscenes/results_caption/` (or pass your own input directory). The main selector expects the columns `Model` and `Caption`, with model identifiers matching those in `MODEL_SAMPLE_COUNTS` in the script. For *evaluation scores* it additionally expects the `hallucination_check` field.
4. Run the main selector from the repository root:

```bash
python Nuscenes/process/gated_spectral_gap_select.py \
  --results-dir Nuscenes/results_caption \
  --output-dir Nuscenes/results/gated_spectral_gap_main \
  --gate 4.75 \
  --max-rank 3
```

Outputs include per-scene selected captions, chosen ranks, and evaluation summaries.

**Input assumptions:** The script currently selects the first four `deepseek-vl2-tiny` captions, three `Qwen2-VL-7B-Instruct` captions, and three `llava-1.5-7b-hf` captions per scene. The input CSVs and precomputed labels are **not included** in this repository. For inference-only use without labels, the selection and reporting logic would need to be separated from the evaluation-label parsing.

### Full NuScenes experiment

To regenerate VLM captions and evaluation annotations, separately obtain [NuScenes](https://www.nuscenes.org/) (the paper uses `v1.0-trainval`), configure local paths using `.env.example`, and run the relevant VLM generation and evaluation scripts. For example:

```bash
cp .env.example .env
# Edit .env to set NUSCENES_DIR to your own dataset directory.
```

Model downloads, substantial GPU resources, and environment-specific adjustments may be required. Check the paper's experimental configuration before reproducing the reported numbers: **some generation scripts currently default to temperature 0.4**, while the paper describes temperature 0.7 and up to 150 tokens. Generation prompts and per-script limits should also be checked against the published setup before making a strict reproduction claim.

**Data availability:** The repository does not include NuScenes images, caption CSVs, generated evaluation labels, or full experiment outputs. The code is provided as a research implementation; the complete published experiment is not yet a one-command reproduction pipeline.

## Citation

If you use this code or the method in your research, please cite:

```bibtex
@article{long2027lowrank,
  title   = {A low-rank consensus method for vision language model hallucination mitigation in autonomous driving},
  author  = {Long, Keke and Guo, Jiacheng and Zhang, Tianyun and Yu, Hongkai and Li, Xiaopeng},
  journal = {Transportation Research Part C: Emerging Technologies},
  volume  = {194},
  pages   = {106092},
  year    = {2027},
  doi     = {10.1016/j.trc.2026.106092}
}
```

## License

**No open-source license has been specified yet.** Please contact the authors for reuse permissions beyond those granted by applicable law until a license is added.
