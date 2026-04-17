# NanoGPT Lab

<div align="center">

**A GPT-style language model built entirely from scratch — custom tokenizer, transformer architecture, training infrastructure, LoRA fine-tuning, streaming inference API, React dashboard, and cloud deployment. No HuggingFace model libraries. No shortcuts.**

[![Dashboard](https://img.shields.io/badge/Dashboard-Live%20on%20Vercel-black?style=for-the-badge&logo=vercel)](https://nano-gpt-lab.vercel.app/)
[![API](https://img.shields.io/badge/API-Live%20on%20Render-46E3B7?style=for-the-badge&logo=render)](https://nano-gpt-lab.onrender.com)
[![Swagger](https://img.shields.io/badge/Swagger-API%20Docs-85EA2D?style=for-the-badge&logo=swagger&logoColor=black)](https://nano-gpt-lab.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

</div>

---

## What This Project Proves

Most LLM projects start with `from transformers import GPT2`. This project does not. Every layer — from the byte-pair encoding merge table to the RoPE rotation kernel, from the cosine learning rate schedule to the LoRA low-rank adapter — is implemented by hand in pure PyTorch and explained from mathematical first principles.

The goal was not just to train a language model. The goal was to understand every decision inside one deeply enough to reason about it at a systems level, then deploy the full stack as a production-grade service. The result: a GPT you can trace from a single matrix multiplication in `attention.py` all the way to a streaming token in a browser.

**This project is the architecture foundation. Its companion — [LLM Inference Lab](https://github.com/Rana-Hassan7272/llm-inference-lab) — builds the production serving stack on top, benchmarking quantization tiers, KV-cache scaling, vLLM dynamic batching, and load testing under concurrency. Together they cover the full LLM engineering stack from weight matrix to deployment.**

---

## Live Deployments

| Service | URL | Stack |
|---|---|---|
| Dashboard | [nano-gpt-lab.vercel.app](https://nano-gpt-lab.vercel.app/) | React + Vite → Vercel |
| API | [nano-gpt-lab.onrender.com](https://nano-gpt-lab.onrender.com) | FastAPI → Docker → Render |
| Swagger Docs | [/docs](https://nano-gpt-lab.onrender.com/docs) | Auto-generated OpenAPI |
| Health | [/health](https://nano-gpt-lab.onrender.com/health) | Liveness probe |

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Training Experiments and Results](#training-experiments-and-results)
4. [LoRA Fine-tuning](#lora-fine-tuning)
5. [Inference Engine](#inference-engine)
6. [API Reference](#api-reference)
7. [Dashboard](#dashboard)
8. [Repository Structure](#repository-structure)
9. [Quickstart](#quickstart)
10. [Configuration](#configuration)
11. [Deployment](#deployment)
12. [Reproducibility](#reproducibility)
13. [Feature Matrix](#feature-matrix)
14. [Skills Demonstrated](#skills-demonstrated)
15. [Limitations](#limitations)
16. [Roadmap](#roadmap)
17. [Related Project](#related-project)

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Data Layer                                   │
│   Raw Text → BPE Tokenizer → train.bin / val.bin (memmap binary)    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────┐
│                        Model Layer                                   │
│                                                                      │
│   tok_emb (V × d)   +   pos_enc (RoPE / ALiBi / learned)           │
│                                                                      │
│   ┌──────────────────────────────────────────┐  × N blocks          │
│   │  PreNorm (RMSNorm)                       │                       │
│   │  MultiHeadAttention  (Q K V W_o)         │                       │
│   │    ├─ Causal mask                        │                       │
│   │    ├─ Optional RoPE on Q, K              │                       │
│   │    └─ KV-cache for inference             │                       │
│   │  Residual add                            │                       │
│   │  PreNorm (RMSNorm)                       │                       │
│   │  FeedForward (SwiGLU / GELU)             │                       │
│   │  Residual add                            │                       │
│   └──────────────────────────────────────────┘                       │
│                                                                      │
│   FinalNorm → LM Head (weight-tied to tok_emb)                      │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────┐
│                      Training Layer                                  │
│   AdamW (decoupled wd) · Warmup+Cosine LR · Grad Clip · AMP        │
│   MLflow logging · Checkpoint save/restore · YAML config control    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────┐
│                   Fine-tuning Layer (LoRA)                           │
│   Freeze W₀ · Inject A,B adapters · Train 0.9% of params           │
│   Merge adapters for zero-overhead inference                        │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────┐
│                      Serving Layer                                   │
│   FastAPI + Uvicorn · SSE Streaming · CORS · Health endpoints       │
│   Greedy / Temperature / Top-k / Top-p decoding                    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────┐
│                      Frontend Layer                                  │
│   React + Vite + Recharts · Generation Playground · Experiment      │
│   Charts · Architecture Explorer · Inference Comparison Panel       │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────┐
│                     Deployment Layer                                 │
│   Docker Compose (local) · Render API · Vercel Dashboard · MLflow   │
└──────────────────────────────────────────────────────────────────────┘
```

> Add diagram: `docs/images/architecture-overview.png` — End-to-end architecture: tokenizer → model → trainer → API → dashboard

---

## Core Components

### Tokenizer — `model/tokenizer.py`

Byte-Pair Encoding implemented from first principles. The algorithm iteratively merges the most frequent adjacent byte-pair in the corpus into a new vocabulary token, learning a merge table that compresses the training text optimally.

```
Raw text → UTF-8 bytes → initial vocab (256 byte tokens)
         → count all adjacent pairs
         → merge most frequent pair → new token
         → repeat until vocab_size reached
         → serialise merge table to data/tokenizer.json
```

Key operations: `train(corpus, vocab_size)`, `encode(text) → List[int]`, `decode(ids) → str`, `save/load(path)`.

---

### Attention — `model/attention.py`

Multi-head causal self-attention with research-grade extensions.

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k  +  M ) V

where  M_{ij} = 0 if j ≤ i  else  −∞   (causal mask)
       scale  = 1/√d_k  prevents dot-products growing into softmax saturation zone
```

| Feature | Status | Detail |
|---|---|---|
| Standard MHA | ✅ | Q, K, V, W_o linear projections |
| Causal masking | ✅ | Upper-triangular boolean mask, −∞ fill |
| Scaled dot-product | ✅ | 1/√d_k numerical stability |
| RoPE | ✅ | Relative position via complex rotation on Q, K |
| ALiBi | ✅ | Linear distance penalty baked into logits |
| Grouped-Query Attention | ✅ | n_kv_heads < n_heads (Mistral/LLaMA 2 style) |
| FlashAttention path | ✅ | torch SDPA kernel on Ampere+ CUDA |
| KV-cache | ✅ | O(1) per-step inference vs O(T²) uncached |

---

### FeedForward — `model/feedforward.py`

Three activation variants, each implemented from mathematical definition:

```
Standard:  FFN(x) = W₂ · GELU(W₁ x)                  4× expansion
SwiGLU:    FFN(x) = (SiLU(W₁x) ⊙ W_gate·x) W₂        8/3× expansion  ← default
GeGLU:     FFN(x) = (GELU(W₁x) ⊙ W_gate·x) W₂        8/3× expansion
```

GELU is implemented three ways — exact erf formula, sigmoid approximation, and the original GPT-2 tanh approximation — with a printed numerical comparison table in the self-test so you can see the approximation error directly. SwiGLU is the default: it is used in LLaMA 1/2/3, Mistral, and PaLM, and consistently outperforms standard GELU at matched parameter counts because the multiplicative gate learns input-dependent feature suppression.

---

### Transformer Block — `model/transformer_block.py`

Pre-norm architecture (GPT-2 / LLaMA standard):

```
x = x + Attn( RMSNorm(x) )    tokens communicate across positions
x = x + FFN(  RMSNorm(x) )    each token processes its own representation
```

**Why pre-norm over post-norm:** Pre-norm routes the gradient directly through the residual connection without passing through LayerNorm. Liu et al. (2020) showed this reduces gradient variance from O(1/N) to O(1/√N) as network depth N increases, which is why every major LLM (GPT-2 onward, LLaMA, Mistral, PaLM) uses pre-norm. Post-norm is original Vaswani (2017) and is unstable beyond ~12 layers without careful warmup.

RMSNorm is used instead of LayerNorm: `RMS(x) = √(mean(x²) + ε)`. No mean subtraction, no bias parameter, ~15% faster. Used in LLaMA, Mistral, Falcon.

Additional supported options: parallel attention+FFN (GPT-J/PaLM style), stochastic depth with linear probability schedule, configurable norm type.

---

### Full Model — `model/nanogpt.py`

```python
NanoGPTConfig.nano()    # 1.2M params — Experiment 1 baseline (4L 4H d=128)
NanoGPTConfig.small()   # 6M params  — Experiment 2 scaled  (6L 6H d=256)
```

Key design decisions with justification:

- **Weight tying** (Press & Wolf 2017): `lm_head.weight = tok_emb.weight`. Saves V×d parameters (~6.4M for nano), improves perplexity by enforcing geometric consistency between the embedding and output spaces.
- **Vocabulary = 8000 (rounded to nearest 64)** for CUDA tensor-core tiling efficiency on Ampere/Turing architectures.
- **Loss at random init verified ≈ ln(vocab_size) ≈ 8.99**: a random model assigns equal probability to all vocabulary tokens, so CE loss = ln(V). Built into the self-test as a sanity check — if your step-0 loss deviates significantly, something is broken before training begins.
- **`configure_optimizer()`**: AdamW with explicit param groups — 2D weight matrices receive weight decay, 1D parameters (biases, norm scales) do not. Weight decay on norms is mathematically incorrect and degrading.

---

## Training Experiments and Results

Each experiment is described by exactly one YAML config in `configs/`. One config = one reproducible run.

### Experiment Summary

| Experiment | Config | Params | Val Loss | Perplexity | GPU Time | Key Finding |
|---|---|---|---|---|---|---|
| Baseline | `train_config.yaml` | 1.2M | 3.34 | **28.4** | 18 min | Stable training floor |
| Larger model | `exp2_larger.yaml` | 6M | 2.85 | **22.1** | 32 min | More capacity → better perplexity |
| With grad clip | `exp3_clip.yaml` | 1.2M | 3.34 | **28.4** | 18 min | Stable |
| No grad clip | `exp3_no_clip.yaml` | 1.2M | diverged | **∞** | stopped at ~2K steps | Exploding gradients |
| LR = 1e-3 | `exp4_lr_1e3.yaml` | 1.2M | diverged | **∞** | stopped at ~1.5K steps | Too aggressive |
| LR = 3e-4 | `exp4_lr_3e4.yaml` | 1.2M | 3.34 | **28.4** | 18 min | Optimal |
| LR = 1e-4 | `exp4_lr_1e4.yaml` | 1.2M | 4.24 | **69.4** | 18 min | Too conservative |

> Add chart: `docs/images/training-loss-curves.png` — Training and validation loss over 5,000 steps

> Add chart: `docs/images/lr-sweep.png` — LR sweep convergence behaviour: 1e-3 diverges, 3e-4 optimal, 1e-4 too slow

### What Each Experiment Demonstrates

**Gradient clipping (Experiment 3):** Without `grad_clip=1.0`, a single large gradient update early in training moves the weights into a region where loss explodes. Training becomes unrecoverable within 2,000 steps. With clipping, training runs stably for the full 5,000 steps and reaches the same final perplexity. This is not a minor hyperparameter choice — it is the structural difference between training and not training on difficult data.

**LR sweep (Experiment 4):** `1e-3` overshoots the loss valley and diverges. `1e-4` converges but so slowly that 5,000 steps yields perplexity 69 versus the optimal 28. `3e-4` with warmup + cosine decay is the canonical GPT-3 recipe and the experiment confirms it empirically on this dataset.

**Warmup + cosine decay (`training/scheduler.py`):** Linear warmup for 100 steps prevents large initial gradient steps from destabilising random initialisation. Cosine decay then reduces the learning rate from peak to near-zero smoothly, enabling fine-grained convergence in the final phase. This is how GPT-3, LLaMA, Mistral, and every major LLM is trained.

---

## LoRA Fine-tuning

### Mathematical Foundation

For a frozen pre-trained weight `W₀ ∈ R^{d × k}`, LoRA parameterises the weight update as a rank-r factorisation:

```
h = W₀ x  +  B A x · (α / r)

A ∈ R^{r × k}   down-projection   (Kaiming uniform init)
B ∈ R^{d × r}   up-projection     (zero init → ΔW = 0 at step 0)
α / r            scaling factor    decouples adapter magnitude from rank choice
```

Only A and B are trainable. `W₀` is frozen throughout. Because B is zero-initialised, `ΔW = B·A = 0` at the start of fine-tuning — the model begins from the pre-trained optimum and is not destabilised by random adapter noise.

**Why it works:** Aghajanyan et al. (2020) showed that fine-tuning tasks have low *intrinsic dimensionality* — the optimal weight update lives in a low-rank subspace. The full weight matrix has hundreds of singular directions, but only a handful carry domain-adaptation signal. Rank-4 captures the useful signal while ignoring noise dimensions.

### Parameter Efficiency Results

Applied to all four attention projections (`W_q, W_k, W_v, W_o`) with `rank=4, alpha=4.0`:

| Method | Trainable Params | % of Total | Checkpoint Size |
|---|---|---|---|
| Full fine-tuning | 1,827,968 | 100.0% | ~7 MB |
| LoRA (rank=4) | **16,384** | **0.896%** | **~64 KB** |
| Reduction factor | — | **111.6×** | **109×** |

> Add chart: `docs/images/lora-efficiency.png` — LoRA trainable parameter efficiency vs full fine-tuning

### Fine-tuning Validation (Poetry corpus, 1,000 steps)

| Step | Val Loss | Perplexity |
|---|---|---|
| 800 | 5.4298 | 228.09 |
| 900 | 5.4226 | 226.47 |
| 1000 | 5.3922 | **219.68** |

The adapter learned poetry-domain phrasing and style while the base Shakespeare-domain weights remain completely intact. Multiple domain adapters can coexist — switching between them requires only loading a 64 KB file, not a full 7 MB checkpoint.

### Adapter Lifecycle — `model/lora.py`

```python
# 1. Inject adapters into a loaded checkpoint
apply_lora(model, LoRAConfig(rank=4, alpha=4.0,
           target_modules={"W_q", "W_k", "W_v", "W_o"}))

# 2. Fine-tune — only A and B parameters receive gradients
# 3. Save — only adapter weights (64 KB, base checkpoint not duplicated)
save_lora(model, config, "results/lora/adapter_poetry.pt", step=1000)

# 4. Merge for zero-overhead production inference
merge_lora(model)   # W_final = W₀ + (α/r)·B·A

# 5. Unmerge to switch adapters without reloading the base model
unmerge_lora(model)
load_lora(model, "results/lora/adapter_news.pt")
```

---

## Inference Engine

### Decoding Strategies — `inference/generate.py`

All four strategies are Python generators, yielding tokens one at a time. This makes SSE streaming trivial — the API just forwards each yielded token as a server-sent event.

| Strategy | Deterministic | Diversity | Best For | Control |
|---|---|---|---|---|
| Greedy | Yes | None | Factual, reproducible output | — |
| Temperature | No | Tunable | Creative exploration | `T ∈ (0.0, 2.0]` |
| Top-k | No | Bounded | Coherent open-ended generation | `k ∈ [1, vocab_size]` |
| Top-p (nucleus) | No | Adaptive | Best general quality | `p ∈ (0.0, 1.0]` |

**Temperature scaling:** `p_i ∝ exp(ℓ_i / T)`. T < 1 sharpens the distribution toward the most probable tokens. T > 1 flattens it toward uniform. T → 0 recovers greedy deterministically.

**Top-p vs top-k:** Top-k uses a fixed vocabulary size at every step. Nucleus sampling instead uses the smallest vocabulary set whose cumulative probability exceeds p — adapting to the model's confidence. On a confident step (e.g., after `"The sky is"`), the nucleus may be 2 tokens. On an uncertain step it expands to 40+. This adaptivity avoids both over-restriction and the long-tail incoherence problem that top-k introduces on uncertain steps.

**KV-cache:** All generators warm the full prompt in one prefill pass, then decode in O(1) per token rather than O(T²). For a 256-token context generating 200 tokens, this eliminates 51,200 redundant attention computations.

---

## API Reference

### Base URLs

```
Production : https://nano-gpt-lab.onrender.com
Local      : http://localhost:8000
Swagger    : https://nano-gpt-lab.onrender.com/docs
```

### Endpoint Reference

| Endpoint | Method | Purpose | Key Parameters |
|---|---|---|---|
| `/generate` | POST | Blocking full generation | `prompt`, `max_new`, `strategy`, `temperature`, `top_k`, `top_p` |
| `/generate/stream` | GET | SSE token-by-token streaming | Same, via query params |
| `/model/info` | GET | Full architecture + parameter breakdown | — |
| `/experiments` | GET | All experiment results as JSON | — |
| `/inference/compare` | GET | NanoGPT vs external benchmark data | — |
| `/health` | GET | Liveness probe | — |

### POST `/generate`

```bash
curl -X POST "https://nano-gpt-lab.onrender.com/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "To be, or not to be, that is the question:",
    "max_new": 100,
    "strategy": "top_p",
    "temperature": 0.8,
    "top_p": 0.9
  }'
```

Response:
```json
{
  "prompt": "To be, or not to be...",
  "generated_text": "...whether tis nobler in the mind to suffer...",
  "full_text": "...",
  "tokens_generated": 100,
  "elapsed_seconds": 4.2,
  "tokens_per_second": 23.8,
  "strategy": "top_p"
}
```

### GET `/generate/stream` — SSE

```bash
curl -N "https://nano-gpt-lab.onrender.com/generate/stream?\
prompt=Whether+tis+nobler&max_new=80&strategy=top_p&temperature=0.8"
```

Stream format:
```
data: {"token": "Whether", "done": false}
data: {"token": " tis",    "done": false}
...
data: {"token": "", "done": true, "meta": {"tokens_generated": 80, "tokens_per_second": 21.4, "elapsed_seconds": 3.7}}
```

> Add screenshot: `docs/images/api-swagger.png` — FastAPI Swagger documentation

---

## Dashboard

Live at [nano-gpt-lab.vercel.app](https://nano-gpt-lab.vercel.app/). Reads the backend URL from `VITE_API_URL`.

### Generation Playground

> Add screenshot: `docs/images/dashboard-generation.png` — Live text generation playground with decoding strategy controls

- Prompt textarea, strategy pill selector (greedy / temperature / top-k / top-p)
- Sliders: temperature, top-k cutoff, top-p nucleus, max new tokens
- Real-time streaming output with blinking cursor (SSE connection)
- Fallback to blocking REST call when SSE is unavailable
- Post-generation stats: tokens generated, tok/s, elapsed seconds

### Experiment Results

> Add screenshot: `docs/images/dashboard-experiments.png` — Experiment analysis with loss curves and LoRA efficiency panel

Four sub-views: loss curves (train + val for all experiments, individually toggleable), LR sweep chart (three-line comparison showing divergence vs optimal vs too-slow), summary table (all experiments with params, perplexity, time, finding), LoRA efficiency panel (parameter reduction visualisation + adapter config + fine-tuning val loss curve).

### Architecture Explorer

> Add screenshot: `docs/images/dashboard-architecture.png` — Architecture explorer with parameter distribution by module

Three views: bar chart (parameters per module), layer tree (hierarchical with exact shapes and param counts), data flow (every tensor shape from `(B, T)` integers through all layers to `(B, T, vocab_size)` logits).

### Inference Comparison

> Add screenshot: `docs/images/inference-compare.png` — NanoGPT vs external inference benchmark

Loaded from `results/inference_comparison.json`. Positions NanoGPT's inference throughput on the parameter/latency curve alongside TinyLlama-1.1B at three quantization tiers from [LLM Inference Lab](https://github.com/Rana-Hassan7272/llm-inference-lab). The 1.2M model runs faster than 1.1B not because it is smarter — because it fits entirely in cache.

---

## Repository Structure

```
nano-gpt-lab/
│
├── model/
│   ├── tokenizer.py           BPE from scratch: train, encode, decode, save/load
│   ├── attention.py           MHA + causal mask + RoPE + ALiBi + GQA + KV-cache
│   ├── feedforward.py         GELU (3 variants) + SwiGLU + GeGLU + MoE stub
│   ├── transformer_block.py   Pre-norm block + RMSNorm + stochastic depth + stack
│   ├── nanogpt.py             Full model: embeddings → blocks → norm → LM head
│   └── lora.py                LoRALinear, apply/merge/unmerge/save/load, LoRAConfig
│
├── data/
│   ├── prepare.py             Raw text → BPE tokenize → train.bin / val.bin
│   ├── dataset.py             PyTorch Dataset: memmap binary, context window chunks
│   └── tokenizer.json         Trained BPE merge table (persisted, committed)
│
├── training/
│   ├── trainer.py             Full loop: AMP, grad clip, eval, MLflow, checkpoint
│   ├── scheduler.py           Linear warmup + cosine decay scheduler
│   └── lora_trainer.py        LoRA-aware fine-tuning loop
│
├── inference/
│   └── generate.py            Greedy / temperature / top-k / top-p generators
│
├── api/
│   └── app.py                 FastAPI: /generate, /stream, /model/info, /experiments
│
├── dashboard/
│   ├── dashboard.jsx          React: playground, experiments, arch explorer, compare
│   ├── main.jsx
│   ├── vite.config.js
│   └── package.json
│
├── configs/
│   ├── train_config.yaml      Baseline: 4L 4H d=128 — Experiment 1
│   ├── exp2_larger.yaml       Larger:   6L 6H d=256 — Experiment 2
│   ├── exp3_clip.yaml         Gradient clipping ON   — Experiment 3a
│   ├── exp3_no_clip.yaml      Gradient clipping OFF  — Experiment 3b
│   ├── exp4_lr_1e3.yaml       LR=1e-3 sweep
│   ├── exp4_lr_3e4.yaml       LR=3e-4 sweep (optimal)
│   └── exp4_lr_1e4.yaml       LR=1e-4 sweep
│
├── experiments/               Test scripts, export utilities, lora_explanation.md
│
├── results/
│   ├── colab-checkpoints/     step_500.pt … step_5000.pt per experiment
│   ├── lora/                  lora_poetry_rank4.pt, before_after.txt/json
│   ├── mlflow_runs_summary.json
│   └── inference_comparison.json
│
├── Dockerfile.api             API image (uvicorn, healthcheck)
├── dashboard/Dockerfile       Dashboard build image
├── docker-compose.yml         api + dashboard + mlflow (one command local stack)
├── vercel.json                Vercel routing configuration
└── requirements.txt
```

---

## Quickstart

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/Rana-Hassan7272/nano-gpt-lab.git
cd nano-gpt-lab
docker compose up --build
```

| Service | URL |
|---|---|
| API + Swagger | http://localhost:8000/docs |
| Dashboard | http://localhost:5173 |
| MLflow UI | http://localhost:5000 |

### Option 2 — Local development

```bash
# 1. Python environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download data
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
  -O data/raw/tinyshakespeare.txt

# 3. Prepare data (tokenise + split)
python data/prepare.py \
  --input_path data/raw/tinyshakespeare.txt \
  --vocab_size 8000 \
  --output_dir data/

# 4. Train baseline
python training/trainer.py --config configs/train_config.yaml

# 5. Run experiments
python training/trainer.py --config configs/exp2_larger.yaml
python training/trainer.py --config configs/exp3_no_clip.yaml
python training/trainer.py --config configs/exp4_lr_1e3.yaml
python training/trainer.py --config configs/exp4_lr_3e4.yaml
python training/trainer.py --config configs/exp4_lr_1e4.yaml

# 6. LoRA fine-tuning
python training/lora_trainer.py \
  --base_checkpoint results/colab-checkpoints/exp1_step5000.pt \
  --finetune_data data/poetry/ \
  --rank 4 --alpha 4.0 --steps 1000 \
  --output results/lora/adapter_poetry.pt

# 7. Start API server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# 8. Start dashboard (separate terminal)
cd dashboard && npm ci && npm run dev
```

---

## Configuration

Every experiment is fully described by one YAML file. Re-running the same config produces the same result.

```yaml
# configs/train_config.yaml — complete field reference

model:
  n_layers: 4            # transformer block depth
  n_heads: 4             # attention heads (d_model must be divisible)
  d_model: 128           # embedding dimension
  n_kv_heads: null       # null = standard MHA | int = grouped-query (GQA)
  ffn_variant: standard  # standard | swiglu | geglu
  context_length: 256    # maximum sequence length
  dropout: 0.1           # applied inside attention + FFN
  pos_encoding: rope     # rope | learned | sinusoidal | alibi
  norm_type: rmsnorm     # rmsnorm | layernorm

training:
  batch_size: 32
  learning_rate: 3e-4
  max_steps: 5000
  warmup_steps: 100        # linear ramp before cosine decay begins
  grad_clip: 1.0           # 0 = disabled (see Experiment 3b)
  eval_interval: 500
  checkpoint_interval: 1000
  amp: true                # mixed precision — requires CUDA

data:
  dataset: tinyshakespeare
  train_split: 0.9

mlflow:
  tracking_uri: mlruns/
  experiment_name: nanogpt-baseline
```

MLflow logs every 100 steps: training loss, validation loss, perplexity, learning rate, gradient norm. View runs at `http://localhost:5000` when running the Docker stack.

---

## Deployment

### Local Docker Stack

```bash
docker compose up --build
# api:       http://localhost:8000
# dashboard: http://localhost:5173
# mlflow:    http://localhost:5000
```

### Cloud — API on Render

1. New Web Service → connect repo → Runtime: **Docker**
2. Dockerfile path: `Dockerfile.api`
3. Environment variables:
   ```
   MODEL_PATH=results/colab-checkpoints/exp1_step5000.pt
   TOKENIZER_PATH=data/tokenizer.json
   ```
4. Health check path: `/health`

### Cloud — Dashboard on Vercel

1. New Project → Import repo → Root directory: `dashboard`
2. Build command: `npm ci && npm run build`
3. Output directory: `dist`
4. Environment variable: `VITE_API_URL=https://nano-gpt-lab.onrender.com`

> Add diagram: `docs/images/deployment-overview.png` — Deployment topology: Vercel frontend + Render API

### Deployment Lessons Learned

Real issues encountered and fixed during deployment:

- **`*.json` in `.gitignore`** excluded `package.json` and `package-lock.json`, breaking the Vercel build with a silent dependency error. Scope JSON ignores specifically: `results/*.json`, not `*.json` globally.
- **Vercel root vs output directory mismatch** produced a blank screen with no console error. Fix: explicitly set root to `dashboard`, output to `dist`, and commit `vercel.json` to pin the configuration.
- **Serving `.jsx` files directly in production** causes a MIME type rejection (`text/jsx` is not valid JavaScript). Vite's build step compiles JSX to plain JS — source files must never be served directly. Always use the built `dist/` output.

---

## Reproducibility

| Artifact | Location | Notes |
|---|---|---|
| Tokenizer | `data/tokenizer.json` | Deterministic: same corpus + vocab_size → same merge table |
| Config files | `configs/*.yaml` | One file fully specifies each experiment |
| Checkpoints | `results/colab-checkpoints/` | Named: `exp1_step5000.pt`, `exp2_step5000.pt` |
| MLflow runs | `mlruns/` (local) | `results/mlflow_runs_summary.json` committed for reference |
| LoRA adapters | `results/lora/*.pt` | 64 KB each; base checkpoint required separately |
| Dashboard data | `results/inference_comparison.json` | Loaded by the comparison panel |

**Runtime notes:** All training experiments ran on Google Colab T4 GPU. Expected times on T4: 1.2M baseline ~18 min, 6M larger ~32 min. Local CPU is 10–20× slower. The deployed Render API runs on CPU — generation is functional but latency is higher than GPU.

Binary checkpoint files (`*.pt`) are not committed to the repository due to size. Re-train from the config files or run the provided Colab notebooks in `notebooks/`.

---

## Feature Matrix

| Module | Status | Description |
|---|---|---|
| BPE Tokenizer | ✅ | Train, encode, decode, persist — written from scratch |
| Multi-head Attention | ✅ | Causal mask, RoPE, ALiBi, GQA, FlashAttn, KV-cache |
| FeedForward | ✅ | GELU (exact + 2 approximations), SwiGLU, GeGLU |
| Transformer Block | ✅ | Pre-norm, RMSNorm, stochastic depth, parallel mode |
| Full NanoGPT | ✅ | Weight tying, checkpointing, 3 preset configs |
| Training Loop | ✅ | AdamW param groups, grad clip, eval loop, MLflow, ckpt |
| LR Scheduler | ✅ | Linear warmup + cosine decay |
| Mixed Precision | ✅ | torch.cuda.amp with GradScaler |
| MLflow Logging | ✅ | Loss, perplexity, LR, grad norm every 100 steps |
| LoRA | ✅ | Inject, merge, unmerge, save, load, diagnostics, config |
| Inference Engine | ✅ | Greedy, temperature, top-k, top-p — SSE generator pattern |
| FastAPI Server | ✅ | /generate, /stream, /model/info, /experiments, /compare |
| SSE Streaming | ✅ | Token-by-token with REST fallback |
| React Dashboard | ✅ | Playground, experiments, arch explorer, inference compare |
| Docker Compose | ✅ | api + dashboard + mlflow — one command |
| Cloud Deployment | ✅ | Render (API) + Vercel (dashboard) — live and accessible |

---

## Skills Demonstrated

**Deep Learning and ML Theory:** Transformer architecture from first principles — causal attention math, residual connection gradient flow, pre-norm vs post-norm stability analysis, RMSNorm vs LayerNorm, weight initialisation, BPE tokenization algorithm, language model training with teacher forcing, all four decoding strategies with mathematical justification, LoRA low-rank adaptation theory and implementation.

**PyTorch Engineering:** Custom `nn.Module` architecture design, memory-mapped binary datasets, KV-cache with correctness validation, AdamW parameter group splitting for correct weight decay, gradient norm tracking, GradScaler for mixed precision, checkpoint serialisation with embedded config for reproducible restoration.

**Experimentation and Research Methodology:** Config-per-experiment discipline, controlled ablation studies (one variable changed per run), MLflow metric tracking, quantitative documentation of failure modes (gradient explosion, LR divergence), honest disclosure of constraints.

**Backend and Systems:** FastAPI application design, async SSE streaming, CORS configuration, environment-variable-driven runtime config, API fallback behaviour when model not loaded, Docker multi-stage build, Docker Compose multi-service orchestration.

**Frontend:** React + Vite, Recharts data visualisation, real-time streaming UX with cursor animation, cross-service API integration, Vercel deployment and routing configuration.

**DevOps and Delivery:** Docker Compose local development stack, cloud deployment on Render and Vercel, deployment troubleshooting (MIME type errors, gitignore scope, Vercel output config), environment variable management across local and cloud contexts.

---

## Limitations

Documented candidly — these reflect real constraints, not implementation gaps:

- **TinyShakespeare (~1M tokens)** is intentional for fast GPU iteration. The model generates convincing Shakespeare-style text but has no general-purpose knowledge of the world beyond this corpus.
- **Cloud free-tier inference** runs on CPU. Render adds ~3–5× latency compared to T4 GPU inference. Streaming mitigates this perceptually.
- **Experiment compute budget** was 5,000 steps per run on a free Colab T4. Chinchilla-optimal training for 1.2M parameters requires ~24M tokens (~24 TinyShakespeare passes). The experiments are controlled ablations, not production-scale training runs.
- **BPE tokenizer** is trained on TinyShakespeare only. Out-of-domain text (modern English, code, other languages) gets suboptimal tokenization with the committed tokenizer. Retrain on a broader corpus for general use.

---

## Roadmap

- [ ] INT8 / INT4 post-training quantization for faster CPU inference on Render
- [ ] Batch generation endpoint for parallel prompt evaluation
- [ ] Redis caching for repeated prompt prefixes
- [ ] Authentication and rate limiting for public API
- [ ] Evaluation harness: perplexity on held-out corpus, BLEU on fixed test prompts
- [ ] Downloadable experiment reports from dashboard
- [ ] CI/CD workflow: lint + unit tests + Docker build check on every PR
- [ ] Unit tests for attention shape correctness, KV-cache consistency, LoRA merge identity
- [ ] Prompt template system for structured input formatting
- [ ] Model registry with checkpoint versioning and metadata tracking

---

## Related Project

**[LLM Inference Lab](https://github.com/Rana-Hassan7272/llm-inference-lab)** is the direct companion to this project. Where NanoGPT Lab builds the model from first principles, LLM Inference Lab builds the production serving infrastructure to deploy models like this one efficiently at scale.

Together they cover the complete LLM engineering stack:

| Capability | NanoGPT Lab | LLM Inference Lab |
|---|---|---|
| Architecture implementation | ✅ From scratch | Reference (TinyLlama-1.1B) |
| Training loop | ✅ Full with experiments | — |
| LoRA fine-tuning | ✅ From scratch | — |
| Quantization benchmarks | — | ✅ 4-bit / 8-bit / FP16 measured |
| KV-cache analysis | Implemented | ✅ 10.25× speedup at 1024 tokens |
| Dynamic batching | — | ✅ vLLM 3.5× over manual batching |
| Load testing | — | ✅ Locust, 0% failure at 20 concurrent users |
| Adaptive routing | — | ✅ Prompt-complexity → precision tier |
| Streaming API | ✅ | ✅ |
| Deployment | ✅ Render + Vercel | ✅ Render + Vercel |

The conceptual bridge between them: understanding *why* KV-cache gives 10× speedup at 1024 tokens requires knowing that uncached attention is O(T²) per generation step — which is clear after implementing `attention.py` by hand. Understanding *why* 4-bit quantization degrades coherence on reasoning tasks requires knowing what the weight matrix precision means for the forward pass — which is clear after building the full forward pass from scratch.

---

## Acknowledgements

- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) — the original inspiration for scope and dataset choice
- Vaswani et al. "Attention Is All You Need" (2017)
- Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- Zhang & Sennrich "Root Mean Square Layer Normalization" (2019)
- Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- Press & Wolf "Using the Output Embedding to Improve Language Models" (2017)
- Holtzman et al. "The Curious Case of Neural Text Degeneration" (2020)

---

<div align="center">

Built with PyTorch · FastAPI · React · Docker · Deployed on Render + Vercel

[Live Dashboard](https://nano-gpt-lab.vercel.app/) · [API Docs](https://nano-gpt-lab.onrender.com/docs) · [LLM Inference Lab](https://github.com/Rana-Hassan7272/llm-inference-lab)

*NanoGPT Lab is not just a model script. It is a complete AI system project demonstrating first-principles model engineering, disciplined experimentation, parameter-efficient fine-tuning, production API design, and real deployment — bridging deep learning theory and production delivery in one coherent implementation.*

</div>