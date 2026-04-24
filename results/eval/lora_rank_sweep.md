# LoRA Rank Sweep Report

- Baseline mean perplexity: **990.2874**
- Best rank: **r=16** (mean perplexity **923.0597**)

| Rank | Mean PPL | Std PPL | Trainable Params | Trainable % | Delta vs Best PPL | Decision |
|---|---:|---:|---:|---:|---:|---|
| 1 | 101026.3604 | 786.3998 | 4,096 | 0.2256% | 100103.3007 | Rank 1 underperforms the best rank by 10844.73% perplexity; only keep it if parameter budget is stricter than quality target. |
| 2 | 4621.5574 | 6.4286 | 8,192 | 0.4502% | 3698.4977 | Rank 2 underperforms the best rank by 400.68% perplexity; only keep it if parameter budget is stricter than quality target. |
| 4 | 962.5481 | 1.3378 | 16,384 | 0.8963% | 39.4885 | Rank 4 underperforms the best rank by 4.28% perplexity; only keep it if parameter budget is stricter than quality target. |
| 8 | 925.6619 | 1.8594 | 32,768 | 1.7767% | 2.6022 | Rank 8 is near-best (within 1% of best perplexity) with 1.777% trainable parameters, so it is a strong efficiency candidate. |
| 16 | 923.0597 | 2.5692 | 65,536 | 3.4913% | 0.0000 | Rank 16 is near-best (within 1% of best perplexity) with 3.491% trainable parameters, so it is a strong efficiency candidate. |
