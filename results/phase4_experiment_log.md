# Phase 4 Experiment Log

## Experiment 1 — Baseline small model

Date: 2026-03-26  
Environment: Colab (GPU run)  
Config target: `n_layers=4`, `n_heads=4`, `d_model=128`, `max_steps=5000`

### Data preparation
- Dataset downloaded: TinyShakespeare (`data/input.txt`)
- Raw chars: `1,115,394`
- BPE vocab size: `8000`
- Total tokens after encoding: `249,235`
- Split:
  - Train tokens: `224,311` (90%)
  - Val tokens: `24,924` (10%)
- Artifacts created:
  - `data/tokenizer.json`
  - `data/train.bin`
  - `data/val.bin`

### Training run (Experiment 1)
- Command:
  - `python training/trainer.py --config configs/train_config.yaml`
- End-of-run metrics:
  - `step=5000`
  - `train_loss=4.7150`
  - `val_loss=6.3298`
  - `perplexity=561.06`
- Last checkpoint:
  - `/content/nano-gpt-lab/results/checkpoints/step_5000.pt`

### Backup/persistence
- Drive checkpoint folder created:
  - `/content/drive/MyDrive/nanogpt-lab_ckpts`
- Checkpoints synced with:
  - `cp -r results/checkpoints/* "/content/drive/MyDrive/nanogpt-lab_ckpts/" || true`
- Note:
  - No output from `cp` is normal when copy succeeds.

### Ready for next experiment
- Keep baseline config snapshot (`configs/train_config.yaml` used in this run)
- Keep baseline checkpoint (`step_5000.pt`)
- Keep MLflow run ID + metrics for comparison table

## Experiment 2 — Larger model

Date: 2026-03-26  
Environment: Colab (GPU run)  
Config file: `configs/exp2_larger.yaml`  
Config target: `n_layers=6`, `n_heads=8`, `d_model=256`, `max_steps=5000`

### Training run (Experiment 2)
- Command:
  - `python training/trainer.py --config configs/exp2_larger.yaml`
- End-of-run metrics:
  - `step=5000`
  - `train_loss=2.4583`
  - `val_loss=7.3931`
  - `perplexity=1624.76`
- Last checkpoint:
  - `/content/nano-gpt-lab/results/checkpoints/step_5000.pt`

### Backup/persistence
- Checkpoints synced with:
  - `cp -r results/checkpoints/* "/content/drive/MyDrive/nanogpt-lab_ckpts/" || true`
- Note:
  - No output from `cp` is normal when copy succeeds.

### Observations
- Training loss decreased strongly vs baseline.
- Validation loss/perplexity remained high and worsened over later steps.
- This suggests overfitting / mismatch under current data scale and hyperparameters.
- Keep this result as-is for honest comparison in Phase 4 table.
