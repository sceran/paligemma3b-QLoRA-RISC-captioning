# Literature Review — QLoRA on PaliGemma for RISC

> Upfront literature scan that informed the design decisions of this project. Compiled 2026-05-16 (one day before the assignment deadline). Coverage: late 2023 through mid-2026. The Addendum at the bottom was added 2026-05-17 specifically to scout recent PEFT initialization methods (EVA, CorDA, HiRA, RandLoRA) for the cumulative ablation in Part 2 of the main notebook.

Sections roughly map to the project's decision points:

- **Section 1** — informed the choice of `rsLoRA + LoRA+` stack as the main recipe and seeded the Part 2 cumulative PEFT ablation.
- **Section 2** — informed the choice of PaliGemma 1 (matching the lab) and the Part 3 PaliGemma 2 swap.
- **Section 3** — grounded the project in the RS-VLM landscape (RSGPT, GeoChat, RS-MoE).
- **Section 4** — selected `pycocoevalcap` BLEU/METEOR/ROUGE-L/CIDEr as the metric suite.
- **Section 5** — motivated a planned vision-tower 2×2 experiment that was eventually dropped in favor of the more PEFT-focused cumulative method comparison.
- **Addendum** — surfaced EVA and CorDA-KPM as candidates; EVA was attempted but ran into a multimodal calibration incompatibility on PaliGemma, documented in the project notes.

---

## Section 1 — PEFT / QLoRA improvements and variants

Recent (late 2025) benchmarks: with proper learning-rate tuning, vanilla LoRA/QLoRA is hard to beat; only a few variants reliably win.

**Key papers:**
- **DoRA** (Liu et al., NVIDIA, ICML 2024 Oral) — [arxiv 2402.09353](https://arxiv.org/abs/2402.09353). Decomposes weights into magnitude + direction; LoRA only updates direction. +0.7% over LoRA, +1.1% over full FT on LLaVA-1.5-7B. Supported in HF PEFT.
- **rsLoRA** (Kalajdzievski, 2023) — [arxiv 2312.03732](https://arxiv.org/abs/2312.03732). Replaces α/r scaling with α/√r to avoid gradient collapse at higher ranks. Directly relevant at rank > 32-64.
- **LoRA+** (Hayou, Ghosh, Yu, 2024) — [arxiv 2402.12354](https://arxiv.org/abs/2402.12354). Higher LR for matrix B than A (ratio ~16x). 1-2% accuracy gain and ~2x speedup, zero extra params.
- **PiSSA** (Meng et al., NeurIPS 2024 Spotlight) — [arxiv 2404.02948](https://arxiv.org/abs/2404.02948). SVD-init adapters with principal components. WARNING: Dec-2025 study showed catastrophic collapse on some tasks. Higher variance.
- **AdaLoRA** (Zhang et al., ICLR 2023) — [arxiv 2303.10512](https://arxiv.org/abs/2303.10512). Adaptive singular-value pruning, strong in low-budget.
- **VeRA** (Kopiczko et al., 2023) — [arxiv 2310.11454](https://arxiv.org/abs/2310.11454). Shared frozen random matrices, 10x fewer params. Underperforms at small ranks.
- **MoSLoRA** / Mixture-of-LoRAs — used in Phi-4-multimodal (2025) and multiple 2025 multimodal papers.

**Consensus (Dec 2025 study, [arxiv 2512.23165](https://arxiv.org/abs/2512.23165)):** DoRA (46.6%) > AdaLoRA (44.2%) > standard LoRA (42.5%) > full FT (44.9%) on average. Vanilla LoRA is competitive when LR is grid-searched per variant.

**For this project:** Start with QLoRA + rsLoRA scaling + LoRA+ LR ratio — essentially free wins, no architectural complexity. Run a QLoRA vs QLoRA+DoRA ablation on the same RISC subset. DoRA's strongest empirical wins are on LLaVA/VL-BART, so a controlled PaliGemma test contributes genuinely novel data.

---

## Section 2 — PaliGemma

**PaliGemma 2 was released December 5, 2024** ([HF blog](https://huggingface.co/blog/paligemma2)). Family is much larger than v1:

| Variant | LLM | Encoder | Resolutions |
|---|---|---|---|
| PaliGemma 2 3B | Gemma 2 2B | SigLIP-So400m | 224 / 448 / 896 |
| PaliGemma 2 10B | Gemma 2 9B | SigLIP-So400m | 224 / 448 / 896 |
| PaliGemma 2 28B | Gemma 2 27B | SigLIP-So400m | 224 / 448 / 896 |

Plus **PaliGemma 2 mix** (Feb 2025) — instruction-tuned for general use, but Google still emphasizes that base PT models should be fine-tuned to a single downstream task, not used zero-shot.

**Fine-tuning resources (2025):**
- [HF blog `paligemma2`](https://huggingface.co/blog/paligemma2) (Niels Rogge, Merve Noyan, Pedro Cuenca, Dec 2024) — official launch + recipe.
- [`merveenoyan/smol-vision`](https://github.com/merveenoyan/smol-vision/blob/main/Fine_tune_PaliGemma.ipynb) `Fine_tune_PaliGemma.ipynb` — canonical reference notebook.
- [Roboflow blog "How to Fine-tune PaliGemma 2"](https://blog.roboflow.com/fine-tune-paligemma-2/) — production-quality recipes.
- [PyImageSearch "Fine-Tuning PaliGemma 2 for Brain Tumor Detection"](https://pyimagesearch.com/) — domain-adaptation case study.

**Community-flagged pitfalls:**
1. **Training instability** — fine-tuning occasionally collapses even on clean data. Use multiple seeds and short warm-up runs to detect.
2. **Token-position loss penalties** — for object detection, output ordering matters; less relevant for captioning.
3. **Domain-OOD images** — community recommends *not freezing* the image encoder when target images differ from pre-training distribution (Roboflow, Datature blogs). **Direct relevance for satellite imagery.**
4. **Resolution / model-size trade-off** — for OCR/document tasks, 448px+ helps; for general captioning at 3B, 224px usually adequate and 5-10x cheaper.

**For this project:** Use PaliGemma 2 3B at 224 resolution. If stretch goal, test 448px on same QLoRA setup to isolate resolution effect.

---

## Section 3 — Remote sensing image captioning with VLMs

PaliGemma + RISC is a defendable choice and not a saturated topic.

**Key papers:**
- **RSGPT** (Hu et al., 2023-2025) — [arxiv 2307.15266](https://arxiv.org/abs/2307.15266). InstructBLIP fine-tuned on RSICap (2,585 hand-annotated). Introduced RSIEval benchmark. SOTA at release on RSICD, RSITMD.
- **GeoChat** ([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Kuckreja_GeoChat_Grounded_Large_Vision-Language_Model_for_Remote_Sensing_CVPR_2024_paper.pdf)). First grounded RS-VLM (LLaVA-1.5 based). Frozen vision + tuned projector + LoRA LLM. Strong on VQA, weaker on dense captioning.
- **RS-MoE** (Lin et al., 2024) — [arxiv 2411.01595](https://arxiv.org/abs/2411.01595). MoE adapter on top of a VLM, strong on spatial-relationship descriptions.
- **SkySenseGPT** (Luo et al., 2024) — [arxiv 2406.10100](https://arxiv.org/html/2406.10100v2). Fine-grained instruction tuning with scene-graph data. Same freeze-pattern as GeoChat.
- **LHRS-Bot** (Muhtar et al., ECCV 2024) — [arxiv 2402.02544](https://arxiv.org/abs/2402.02544). OpenStreetMap-style VGI for grounding, two-stage curriculum.
- **RS-CapRet** (Silva et al., 2024) — [arxiv 2402.06475](https://arxiv.org/html/2402.06475v1). Reports surpassing RSGPT on RSICD CIDEr by +1.576.
- Surveys: [arxiv 2505.14361](https://arxiv.org/html/2505.14361v1) (May 2025) and [arxiv 2410.17283](https://arxiv.org/abs/2410.17283).

**Typical metrics** (varies by split):
- RSICD CIDEr: classic CNN+RNN ~2.0-2.3; recent SOTA RS-specific VLMs ~2.5-2.8.
- Sydney-Captions and UCM-Captions saturate higher (CIDEr 3+).
- RISC: fewer published numbers — defensible "first-mover" benchmark.

**Dominant architectural pattern:** **frozen vision encoder + projector tuning + LoRA LLM** (GeoChat, SkySenseGPT, RSGPT, LHRS-Bot). PaliGemma 2 has the same shape. Cite RSGPT, GeoChat, RS-MoE as RS-VLM baselines (don't try to outperform — trained on millions of pairs, not 44K).

---

## Section 4 — Caption evaluation metrics

**The field is bifurcating in 2025:** classic n-gram metrics remain mandatory for back-compat; polished papers now add reference-free or learned metrics, increasingly an LLM-as-judge variant.

**Key papers:**
- **PAC-S / PAC-S++** (Sarto et al., CVPR 2023 / IJCV 2025) — [github aimagelab/pacscore](https://github.com/aimagelab/pacscore). CLIP-based reference-free metric. **Most popular learned metric in 2025 captioning papers.**
- **CapArena** (2025) — [arxiv 2503.12329](https://arxiv.org/abs/2503.12329) (Findings of ACL 2025). VLM-as-a-judge (GPT-4o) achieves 93.4% correlation with human ranking at ~$4 per benchmark run.
- **Survey:** "Image Captioning Evaluation in the Age of Multimodal LLMs" (IJCAI 2025) — [github aimagelab/awesome-captioning-evaluation](https://github.com/aimagelab/awesome-captioning-evaluation).
- **VCRScore** (2025) — [arxiv 2501.09155](https://arxiv.org/abs/2501.09155). Newer V&L precision-recall reference-free metric.
- **ReconScore** — RS-specific: reconstructs the scene from caption, measures perceptual similarity. Niche but RS-tailored.

**Caveats**: CLIPScore, PAC-S, UMIC are insensitive to caption structure and struggle with negation and fine-grained hallucination. CIDEr remains the most-reported single number in RS captioning.

**Recommended reporting for this project:**
1. **Required (comparability):** BLEU-1, BLEU-4, METEOR, ROUGE-L, CIDEr, SPICE.
2. **Modern (one extra column):** CLIPScore *and* RefPAC-S++ — cheap, signal current literacy.
3. **Stretch:** 100-200 sampled test images, GPT-4o (or Claude) as judge with short rubric, report inter-judge agreement.

---

## Section 5 — Vision encoder adaptation in VLM fine-tuning

Most project-relevant section; literature is genuinely mixed.

**Key findings:**
- **Default in RS-VLM literature** (GeoChat, SkySenseGPT, SkyEyeGPT, RSGPT): vision encoder **frozen**, only projector + LLM (LoRA) trained. Inherited from LLaVA conventions, not RS-specific evidence.
- **Counter-evidence:** SigLIP-HD ([arxiv 2502.14786](https://arxiv.org/pdf/2502.14786), Feb 2025) — unfreezing SigLIP during LLaVA-NeXT-style training improves DocVQA and ChartQA each by +3.6 points. Suggests encoder adaptation helps for OOD visual domains.
- **Medical VLM evidence** ([PMC12730038](https://pmc.ncbi.nlm.nih.gov/articles/PMC12730038/), 2025): Targeted LoRA at <1% of parameters (including vision tower) outperforms wider adaptations *and* full fine-tuning. Multi-stage common: freeze encoder for caption stabilization, then unfreeze for radiology-specific adaptation.
- **Robotics / VLA** ([arxiv 2512.11921](https://arxiv.org/html/2512.11921v1), late 2025): Vision encoder LoRA adds ~25M params, ~4x training time; needed when camera setups differ from pre-training.
- **HF community thread on GeoChat domain adaptation** confirms practitioners regularly experiment with unfreezing the vision-to-language projection adapter for OOD target images.

**Consensus:** For domains close to natural images, freezing is fine. For substantially shifted domains (satellite top-down views) — suggestive but **not conclusive** evidence that adapting the vision tower (via LoRA, not full FT) helps. **No definitive controlled study on remote sensing exists in 2025.**

**For this project:** Best originality experiment. Run a clean 2×2:
- {freeze SigLIP, LoRA on SigLIP} × {freeze projector, train projector}
- All with LoRA on Gemma 2
- Same compute budget, same data, same metrics.

Either result is publishable in assignment context: confirming convention shows rigor; finding LoRA-on-SigLIP helps on RISC = real contribution. Cite SigLIP-HD + medical VLM literature as motivation.

---

## Gaps / uncertainty

- Could not confirm published benchmark numbers on `caglarmert/full_riscm`. Baseline on RSICD first (well-documented), treat RISC as headline novel dataset.
- No direct 2025-2026 paper compares LoRA variants specifically on PaliGemma 2. DoRA wins on LLaVA-1.5; whether that transfers is open.
- LLM-as-judge for *remote-sensing* captioning specifically is not yet established practice — applying it would be novel.

## Highest-leverage suggestions

1. PaliGemma 2 3B @ 224 + QLoRA + rsLoRA scaling + LoRA+ LR ratio = strong, defensible baseline.
2. ONE clean originality experiment: vision-encoder LoRA vs. frozen. Addresses real gap in RS literature.
3. Report BLEU-1/4, METEOR, ROUGE-L, CIDEr, SPICE + CLIPScore + RefPAC-S++; GPT-4o-judge appendix on 200 sampled images.
4. In related work: cite RSGPT, GeoChat, RS-MoE, SkySenseGPT as RS-VLM baselines. Frame project as "QLoRA-light with explicit vision-encoder adaptation study."

---

## Addendum — Recent PEFT methods scan (2026-05-17)

Targeted scan for **late 2025 / 2026 PEFT methods**, filtered by: peft library integration + QLoRA compatibility + practical implementation under 30-min/A100 budget.

### Tier 1 — Worth trying (in peft, QLoRA-compatible)

#### EVA — Explained Variance Adaptation

- [arxiv 2410.07170](https://arxiv.org/abs/2410.07170) (Oct 2024, NeurIPS 2025)
- [peft PR #2142](https://github.com/huggingface/peft/pull/2142) — merged
- **Innovation:** LoRA init via incremental SVD of layer **input activations** (not weights like PiSSA). Adaptively redistributes ranks across layers based on explained variance.
- **Config:** `init_lora_weights="eva"` + `eva_config=EvaConfig(rho=2.0)` + call `initialize_lora_eva_weights(model, dataloader)` after building peft model
- **Project relevance:** Activation-driven SVD likely synergizes with satellite imagery (domain shift). Rank redistribution differentiates from our stacked LoRA+/rsLoRA.

#### CorDA — Context-Oriented Decomposition Adaptation

- [arxiv 2406.05223](https://arxiv.org/abs/2406.05223) (NeurIPS 2024, peft late-2024 integration)
- [peft PR #2231](https://github.com/huggingface/peft/pull/2231)
- **Innovation:** SVD on covariance of activations from ~256 task samples. Two modes:
  - **KPM** (knowledge-preserved) — protects pretraining knowledge from catastrophic forgetting
  - **IPM** (instruction-previewed) — faster convergence; beats PiSSA
- **Config:** `init_lora_weights="corda"` + `corda_config=CordaConfig(corda_method="kpm")`
- **Project relevance:** KPM mode tells story: *"preserve PaliGemma's general visual knowledge while specializing on satellite captions"* — pairs with RS domain shift narrative.

### Tier 2 — Interesting but friction

#### HiRA — Hadamard High-Rank Adaptation

- [arxiv (ICLR 2025 Oral)](https://iclr.cc/virtual/2025/oral/31839)
- peft PR #2668 **OPEN, not merged**
- **Innovation:** ΔW = W ⊙ (BA) — Hadamard with frozen W enables high-rank update with LoRA-equivalent param count
- **Risk:** Need to vendor official repo or wait for PR merge; untested on VLMs

#### RandLoRA

- [arxiv 2502.00987](https://arxiv.org/abs/2502.00987) (ICLR 2025)
- **Project relevance:** Paper explicitly notes VLM tasks benefit most
- **Risk:** Not in peft core; custom drop-in module needed; not validated with 4-bit

### Tier 3 — Skip

ABBA (2505.14238) too new + no peft port. GoRA (2502.12171) requires custom gradient calibration. DoRAN (2510.04331) inherits DoRA's quantization friction. Dual LoRA (2512.03402) too new. NEAT, VB-LoRA, OLoRA, MoE-LoRA family: each has either expressivity-equivalent-to-higher-rank OR known QLoRA bugs OR major training-time overhead.

### Concrete recommendation for cumulative ablation

Run alongside stacked LoRA + rsLoRA + LoRA+ baseline:

1. **EVA init** (replaces random init, keeps rsLoRA scaling) — one-line config + 1-2 min calibration pass
2. **CorDA-KPM init** — same drop-in pattern, RS-relevant framing

Both are NeurIPS-class methods, officially in peft, compose with QLoRA, fit 30-min/A100 budget. Together they give the exploratory section two genuinely 2024-2025-era methods without rolling custom modules.

### Sources added in this addendum

- [EVA paper](https://arxiv.org/abs/2410.07170)
- [EVA peft PR #2142](https://github.com/huggingface/peft/pull/2142)
- [CorDA paper](https://arxiv.org/abs/2406.05223)
- [CorDA peft PR #2231](https://github.com/huggingface/peft/pull/2231)
- [HiRA ICLR 2025 Oral](https://iclr.cc/virtual/2025/oral/31839)
- [HiRA peft PR #2668](https://github.com/huggingface/peft/pull/2668)
- [RandLoRA](https://arxiv.org/abs/2502.00987)
- [OLoRA quantization issue #1999](https://github.com/huggingface/peft/issues/1999)
- [Advanced LoRA fine-tuning comparison (Kaitchup)](https://kaitchup.substack.com/p/advanced-lora-fine-tuning-how-to)
