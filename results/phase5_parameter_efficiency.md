# Phase 5 — Step 4 Parameter Efficiency Report

## Scope

This report documents the parameter-efficiency comparison required by Phase 5 Step 4:
- Full fine-tuning trainable parameters
- LoRA fine-tuning trainable parameters
- Relative percentage and practical impact
- Quality comparison before vs after LoRA adaptation

Model family used:
- Base checkpoint: `results/colab-checkpoints/exp1_step5000.pt`
- Base architecture: `d_model=128`, `n_layers=4`, `n_heads=4`, `vocab_size=8000`

LoRA setup used:
- Adapter file: `results/lora/lora_poetry_rank4.pt`
- Rank: `r=4`
- Target modules: `W_q`, `W_k`, `W_v`, `W_o`
- LoRA fine-tune steps: `1000`

---

## Parameter Count Comparison

From LoRA training logs:
- Total model parameters: `1,827,968`
- Trainable LoRA parameters: `16,384`
- Trainable ratio: `0.8963%`

Therefore:
- **Full fine-tuning** trainable params = `1,827,968` (100%)
- **LoRA fine-tuning** trainable params = `16,384` (~0.90%)

Reduction factor:

`1,827,968 / 16,384 = 111.57x`

So LoRA reduces trainable parameter count by approximately **111.6x** for this project configuration.

---

## Adapter Storage Efficiency

Saved LoRA adapter artifact:
- `results/lora/lora_poetry_rank4.pt`
- Size from training log: ~`64.0 KB` (adapter tensors only)

Practical implication:
- Store one shared base checkpoint + multiple tiny domain adapters
- No need to duplicate full model weights per domain

---

## Quality Comparison (Before vs After)

Comparison artifacts:
- `results/lora/before_after.txt`
- `results/lora/before_after.json`

Observed behavior:
- Base output remains Shakespeare-style and generic.
- LoRA output shifts toward poetry-domain phrasing/style patterns.
- Adaptation effect is visible while preserving coherent generation.

Fine-tune validation trend (poetry corpus):
- Step 800: `val_loss=5.4298`, `ppl=228.09`
- Step 900: `val_loss=5.4226`, `ppl=226.47`
- Step 1000: `val_loss=5.3922`, `ppl=219.68`

This indicates the adapter learned the new domain distribution during the 1000-step run.

---

## Step 4 Deliverable Statement

Required comparison recorded for this project setup:

- Full fine-tuning: `1,827,968` trainable parameters
- LoRA fine-tuning: `16,384` trainable parameters (`0.8963%` of total)
- Quality difference: domain-adapted output is achieved with a very small trainable subset

Phase 5 Step 4 is completed with reproducible artifacts and measured evidence.
