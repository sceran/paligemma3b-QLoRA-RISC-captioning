# PaliGemma-3B · QLoRA · RISC Captioning

Parameter-efficient fine-tuning of **PaliGemma-3B** on the **RISC** remote-sensing image captioning dataset using **QLoRA** (4-bit NF4 quantization + low-rank adapters).

## Overview

Vision-Language Models such as PaliGemma — built on top of SigLIP (vision encoder) and Gemma (language decoder) — achieve strong general-purpose image captioning, but adapting them to specialized domains is computationally expensive. QLoRA addresses this by combining 4-bit base-weight quantization with low-rank adapter training, enabling fine-tuning of multi-billion-parameter VLMs on a single consumer GPU with minimal accuracy loss.

This project adapts PaliGemma-3B to satellite imagery, where vocabulary, scale, and visual statistics differ substantially from the model's pre-training distribution.

## Dataset

**RISC** — Remote Sensing Image Captioning · [`caglarmert/full_riscm`](https://huggingface.co/datasets/caglarmert/full_riscm)

- 44,521 satellite images at 224 × 224 resolution
- 5 reference captions per image (222,605 total)

## Method

- **Base model:** PaliGemma-3B (SigLIP-400M vision encoder + Gemma-2B language decoder)
- **Fine-tuning:** QLoRA — 4-bit NF4 base weights, LoRA adapters on attention and MLP projections
- **Evaluation:** standard caption-quality metrics on a held-out split, comparing baseline vs. fine-tuned model on identical inputs

## Status

Work in progress.

## Acknowledgments

Course project for **DI 725 — Transformers and Attention-Based Deep Networks**, METU.
