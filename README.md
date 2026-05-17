# PaliGemma-3B · QLoRA · RISC Captioning

Parameter-efficient fine-tuning of **PaliGemma-3B** on the **RISC** remote-sensing image captioning dataset using **QLoRA** (4-bit NF4 quantization + low-rank adapters).

## Headline result

| Run                            | BLEU-1     | BLEU-4     | METEOR     | ROUGE-L    | CIDEr      |
| ------------------------------ | ---------: | ---------: | ---------: | ---------: | ---------: |
| Baseline (zero-shot)           | 0.329      | 0.040      | 0.095      | 0.269      | 0.253      |
| **QLoRA fine-tuned** (1 epoch) | **0.853**  | **0.585**  | **0.409**  | **0.750**  | **2.207**  |

CIDEr improves 8.7× over the zero-shot baseline. Evaluation is on 500 held-out test images with multi-reference scoring against all five RISC captions.

## Overview

Vision-Language Models such as PaliGemma, built on SigLIP (vision encoder) and Gemma (language decoder), achieve strong general-purpose image captioning but adapting them to specialized domains is computationally expensive. QLoRA addresses this by combining 4-bit base-weight quantization with low-rank adapter training, enabling fine-tuning of multi-billion-parameter VLMs on a single consumer GPU with minimal accuracy loss.

This project adapts PaliGemma-3B to satellite imagery, where vocabulary, scale, and visual statistics differ substantially from the model's pre-training distribution.

## Dataset

**RISC** — Remote Sensing Image Captioning · [`caglarmert/full_riscm`](https://huggingface.co/datasets/caglarmert/full_riscm)

- 44,521 satellite images at 224 × 224 resolution
- 5 reference captions per image (222,605 total)
- Image-level train/val/test split (seed 42): 37,843 / 2,226 / 4,452

## Method

- **Base model:** `google/paligemma-3b-pt-224` (SigLIP-So400m vision encoder + Gemma-2B language decoder)
- **PEFT:** QLoRA with 4-bit NF4 double quantization, bf16 compute
- **LoRA:** rank 16, α=32, dropout 0.05 on `q,k,v,o,gate,up,down` projections
- **Main run:** rsLoRA scaling + LoRA+ (16× LR on B matrices), AdamW at 1e-4, 1 epoch (2,366 steps)
- **Trainable:** 22.6M / 2.95B (0.77%)
- **Metrics:** BLEU-1/4, METEOR, ROUGE-L, CIDEr via `pycocoevalcap`

## Structure

```text
.
├── paligemma_qlora_risc.ipynb              Main notebook (Parts 1 & 2)
├── paligemma2_qlora_risc_exploration.ipynb Part 3: PaliGemma 2 exploration
├── report.tex                              1-page IEEE report (LaTeX source)
├── report.pdf                              Compiled report
├── qualitative.pdf                         5-example qualitative comparison figure
├── results/                                Raw metric outputs (JSON + CSV)
└── README.md
```

The main notebook covers:

- **Part 1:** baseline evaluation, main 1-epoch QLoRA fine-tuning, qualitative comparison
- **Part 2 (exploratory):** caption-selection strategy ablation (4 configs), cumulative PEFT method ablation (vanilla LoRA, rsLoRA, LoRA+, DoRA stacks)

The Part 3 notebook covers the PaliGemma 2 swap under the same recipe.

## Experiment tracking

Weights & Biases dashboards:

- Main project: [wandb.ai/sceran/paligemma3b-QLoRA-RISC](https://wandb.ai/sceran/paligemma3b-QLoRA-RISC)
- Initial overnight runs (default project): [wandb.ai/sceran/huggingface](https://wandb.ai/sceran/huggingface)

All training loss curves, qualitative tables, and per-experiment metrics are logged.

## Reproducibility

Splits are deterministic given the fixed seed. The `results/` folder snapshots the JSON/CSV outputs of every experiment. Each notebook is self-contained: a fresh Colab A100 runtime should reproduce the metrics with one upload-and-run pass (about 3 hours for the full Part 1 + Part 2 + Part 3 sequence).

## Acknowledgments

Course project for **DI 725 — Transformers and Attention-Based Deep Networks**, METU.
