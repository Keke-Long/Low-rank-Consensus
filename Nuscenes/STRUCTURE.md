# Nuscenes Directory Structure

This directory is organized into three groups:

## 1) Low-rank Mainline (proposed method)

- `process/`: core processing scripts for low-rank pipeline
- `results/`: low-rank analysis outputs
- `results_caption/`: caption result objects used by low-rank analysis
- `results_detection_fact/`: detection fact cache/supporting artifacts

## 2) Baselines

All baseline methods are grouped under:

- `baselines/debate_baseline/`
- `baselines/pairwise_baseline/`
- `baselines/summarize_baseline/`

## 3) Sensitivity Analysis

All sensitivity/ablation-style experiment folders are grouped under:

- `sensitivity_analysis/large_vlm/` (formerly `results_caption_large_VLM`)
- `sensitivity_analysis/low_temperature/` (formerly `experiments`)

