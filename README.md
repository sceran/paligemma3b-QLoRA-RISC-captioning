# PaliGemma-3B · QLoRA · RISC Captioning

Parameter-efficient fine-tuning of **PaliGemma-3B** on the **RISC** remote-sensing image captioning dataset using **QLoRA** (4-bit NF4 quantization + low-rank adapters).

Course project for **DI 725 — Transformers and Attention-Based Deep Networks**, METU.

## Headline result

| Run                            |     BLEU-1 |     BLEU-4 |     METEOR |    ROUGE-L |      CIDEr |
| ------------------------------ | ---------: | ---------: | ---------: | ---------: | ---------: |
| Baseline (zero-shot)           |      0.329 |      0.040 |      0.095 |      0.269 |      0.253 |
| **QLoRA fine-tuned** (1 epoch) |  **0.853** |  **0.585** |  **0.409** |  **0.750** |  **2.207** |

CIDEr improves **8.7× over the zero-shot baseline**. Evaluation is on 500 held-out test images with multi-reference scoring against all five RISC captions per image.

Full numerical results for every experiment are kept under [`results/`](results/) as JSON and CSV. The compiled 1-page IEEE report is [`Report.pdf`](Report.pdf); the LaTeX source is [`report.tex`](report.tex).

## Project scope

The project goes beyond the assignment's required QLoRA + comparison and adds two ablations plus a base-model swap:

- **Part 1 (main).** Baseline evaluation, 1-epoch QLoRA fine-tune, qualitative comparison.
- **Part 2 (exploratory).**
  - *Caption-selection strategy ablation:* random vs. first vs. longest vs. concatenated. Random wins; concatenation collapses CIDEr to zero because the model learns to produce overly long outputs.
  - *Cumulative PEFT method ablation:* vanilla LoRA, rsLoRA, LoRA+, the stacked recipe, and DoRA. Replicate runs measure an inter-run noise floor of ~0.16 CIDEr. Under this lens only rsLoRA marginally exceeds the noise band, suggesting modern PEFT improvements do **not** stack monotonically on this domain.
- **Part 3 (further exploration).** Swap the base to PaliGemma 2 3B under an identical rsLoRA-only recipe. PG2 ties PG1 at the fixed 500-step budget; PG2's zero-shot baseline is actually worse, but fine-tuning closes the gap.

The Part 3 notebook is not committed to this public repo (kept local for the ODTUCLASS submission). The numerical comparison is included in [`results/results_part3_pg2_vs_pg1.csv`](results/results_part3_pg2_vs_pg1.csv).

## Dataset

**RISC** — Remote Sensing Image Captioning · [`caglarmert/full_riscm`](https://huggingface.co/datasets/caglarmert/full_riscm)

- 44,521 satellite images at 224 × 224 resolution
- 5 reference captions per image (222,605 total)
- Image-level train/val/test split (seed 42): 37,843 / 2,226 / 4,452

## Method (main run)

- **Base model:** `google/paligemma-3b-pt-224` (SigLIP-So400m vision encoder + Gemma-2B language decoder)
- **PEFT:** QLoRA with 4-bit NF4 double quantization, bf16 compute
- **LoRA:** rank 16, α=32, dropout 0.05 on `q,k,v,o,gate,up,down` projections
- **Main run extras:** rsLoRA scaling (α/√r) + LoRA+ (16× LR on the B matrices)
- **Optimizer:** AdamW, learning rate 1e-4, batch 4 × grad accumulation 4, 3% warmup, 1 epoch (2,366 steps)
- **Trainable parameters:** 22.6M / 2.95B (0.77%)
- **Metrics:** BLEU-1/4, METEOR, ROUGE-L, CIDEr via `pycocoevalcap` (multi-reference)
- **Hardware:** Single Colab A100, ≈ 2.5 h for the main run

## Repository structure

```text
.
├── paligemma_qlora_risc.ipynb   Main notebook (Parts 1 and 2, all outputs preserved)
├── report.tex                   1-page IEEE report source
├── Report.pdf                   Compiled report
├── qualitative.pdf              5-example baseline vs. fine-tuned comparison figure
├── docs/
│   └── literature-survey.md     Upfront literature scan (informed design decisions)
├── results/                     Raw metric outputs (JSON + CSV per experiment)
└── README.md                    This file
```

## Experiment tracking

Weights & Biases dashboards (training curves, qualitative tables, per-experiment metrics):

- Main project: [wandb.ai/sceran/paligemma3b-QLoRA-RISC](https://wandb.ai/sceran/paligemma3b-QLoRA-RISC)
- Early overnight runs (default project): [wandb.ai/sceran/huggingface](https://wandb.ai/sceran/huggingface)

## Literature survey

The design decisions in this project (PEFT method choices, metric protocol, base-model version, etc.) were informed by an upfront literature scan. See [`docs/literature-survey.md`](docs/literature-survey.md) for the full review, including the post-hoc addendum scouting recent PEFT initialization methods (EVA, CorDA, HiRA, RandLoRA) for the Part 2 cumulative ablation.

## Reproducibility

Splits are deterministic given the fixed seed (42). The `results/` folder snapshots the JSON/CSV outputs of every experiment. The main notebook is self-contained: a fresh Colab A100 runtime should reproduce the numbers with one upload-and-run pass.
