# LoRA Rank Sweep Report

- Baseline mean perplexity: **990.2874**
- Best rank: **r=4** (mean perplexity **962.5481**)

| Rank | Mean PPL | Std PPL | Trainable Params | Trainable % | Delta vs Best PPL | Decision |
|---|---:|---:|---:|---:|---:|---|
| 4 | 962.5481 | 1.3378 | 16,384 | 0.8963% | 0.0000 | Rank 4 is near-best (within 1% of best perplexity) with 0.896% trainable parameters, so it is a strong efficiency candidate. |
