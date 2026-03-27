# LoRA — Low-Rank Adaptation of Large Language Models
### A Mathematical First-Principles Explanation
*Phase 5, Step 1 — NanoGPT Lab*

---

## The Problem LoRA Solves

You have trained a language model and want to adapt it to a new domain — poetry, medical text, legal documents. Full fine-tuning means all base parameters become trainable again. You then need to store a full copy of the model for every fine-tuned variant. For very large models this is expensive or impossible on a single free-GPU setup.

LoRA (Hu et al. 2021, *"LoRA: Low-Rank Adaptation of Large Language Models"*) answers the question: **can we represent the update to a weight matrix during fine-tuning using far fewer parameters, without losing quality?**

The answer is yes, and the justification is mathematical.

---

## What is Frozen — The Base Weights

Let `W₀ ∈ R^{d × k}` be a pre-trained weight matrix — for example, the query projection `W_q` in one attention layer. After pre-training, `W₀` encodes compressed knowledge about language: syntax, semantics, world facts. We treat this as a fixed point of reference.

During LoRA fine-tuning, `W₀` is **completely frozen**. Its gradients are zeroed. It does not change. It acts as a permanent foundation — the "prior" in Bayesian terms — on top of which we learn only a small correction.

Formally: `∂L / ∂W₀ = 0` throughout fine-tuning.

This has two consequences. First, we never need to store updated copies of `W₀` — multiple fine-tuned variants can share the same base model weights in memory. Second, the base model's generalisation capability is preserved; catastrophic forgetting is structurally suppressed.

---

## What is Trained — The LoRA Matrices A and B

LoRA hypothesises that the **weight update** `ΔW` needed for adaptation has **low intrinsic rank**. This is the central mathematical claim.

A matrix `M ∈ R^{d × k}` has rank r if it can be written exactly as a product of two skinny matrices:

```
M = A · B    where  A ∈ R^{d × r},  B ∈ R^{r × k},  r ≪ min(d, k)
```

This is the **rank factorisation**. Instead of storing `d × k` numbers, we store `d × r + r × k = r(d + k)` numbers. For `d = k = 512` and `r = 4`:

```
Full matrix  : 512 × 512 = 262,144 parameters
LoRA factors :   4 × 512 +  4 × 512 = 4,096 parameters   (1.56% of full)
```

LoRA introduces exactly this factorisation for `ΔW`:

```
ΔW = B · A
```

where:
- `A ∈ R^{r × k}` — the **down-projection matrix**, initialised from `N(0, σ²)`
- `B ∈ R^{d × r}` — the **up-projection matrix**, initialised to **zero**

`A` and `B` are the only trainable parameters. Their gradients flow normally. Backpropagation optimises them exactly as any other `nn.Parameter`.

**Why initialise B to zero?** At the start of fine-tuning, `ΔW = B · A = 0`. The model behaves identically to the pre-trained base model. Training starts from a stable, known-good point. If we initialised both A and B randomly, the model would generate noise on step 0 and training would have to first "unlearn" the random perturbation before learning anything useful.

---

## The Forward Update Rule and Scaling

The modified forward pass for a linear layer with LoRA is:

```
h = W₀ x  +  ΔW x  ·  (α / r)
  = W₀ x  +  B A x  ·  (α / r)
```

where:
- `x ∈ R^{k}` is the input (one token's representation)
- `W₀ x` is the frozen base computation — unchanged from pre-training
- `B A x` is the low-rank adaptation — the learned correction
- `α / r` is the **LoRA scaling factor**

### The Scaling Factor α/r

`α` (alpha) is a scalar hyperparameter, typically set to `2r` or simply `r` so that `α/r = 1` or `α/r = 2`. Its purpose is subtle.

When you change `r`, the magnitude of `ΔW = B · A` changes because you are summing over more or fewer rank-1 outer products. Without scaling, changing `r` from 4 to 16 would quadruple the effective learning rate of the adapter, requiring you to retune the optimiser. The `α/r` normalisation **decouples the adapter's contribution magnitude from the choice of r**, so you can sweep rank as an ablation without touching the learning rate.

In code this looks like:

```python
# During forward pass:
lora_output = (x @ self.A.T @ self.B.T) * (self.alpha / self.rank)
output = base_output + lora_output
```

The scaling is applied at **inference time, not at merge time** — though when you merge LoRA weights back into the base model for deployment, you bake in the scale:

```
W_merged = W₀  +  (α / r) · B · A
```

After merging, the model is architecturally identical to the original — zero inference overhead. This is LoRA's killer feature for production deployment.

---

## Why Rank = 4 Keeps Trainable Parameters Small

The rank `r` controls the **expressiveness** of the adapter. Let's compute what rank 4 means for our NanoGPT attention projections.

For a projection with `d = 256` (`W ∈ R^{256×256}`):

| r  | Params per matrix (A + B)       | 4 matrices (Wq,Wk,Wv,Wo) | % of 6M model |
|----|----------------------------------|---------------------------|---------------|
| 1  | 256×1 + 1×256 = 512              | 2,048                     | 0.034%        |
| 4  | 256×4 + 4×256 = 2,048            | 8,192                     | 0.137%        |
| 8  | 256×8 + 8×256 = 4,096            | 16,384                    | 0.273%        |
| 16 | 256×16 + 16×256 = 8,192          | 32,768                    | 0.546%        |
| 64 | 256×64 + 64×256 = 32,768         | 131,072                   | 2.185%        |

Rank 4 gives **8,192 trainable parameters** across the four attention projections in one layer group. In our project flow, this keeps trainable parameters as a tiny fraction of the full model while preserving the base checkpoint knowledge.

**Why does low rank work at all?**

Aghajanyan et al. (2020) showed empirically that the **intrinsic dimensionality** of fine-tuning tasks is surprisingly low. When you fine-tune a model on sentiment classification, the change in the weight space can be well-approximated by a matrix of rank 1-4. The pre-trained model already lives in a region of weight space that is close to the optimal fine-tuned model — you only need a small perturbation in a low-dimensional subspace to get there.

Formally: if the fine-tuning loss landscape near `W₀` is well-conditioned along only `r` directions, then a rank-`r` update captures all the useful signal. The remaining `min(d,k) - r` singular directions are either noise or directions the base model already handles correctly.

This is why you almost always see diminishing returns past `r = 8` or `r = 16` on standard NLP tasks: you are fitting noise beyond that rank.

---

## Which Layers to Apply LoRA To

In the original paper, LoRA is applied to `W_q` and `W_v` only. Later work (including QLoRA, LoftQ) applies it to all four attention projections `W_q, W_k, W_v, W_o` and sometimes to the FFN projections `W1, W2` as well.

For this project we apply LoRA to **all four attention projections** per layer, which gives:

```
Layers:              n_layers = 6
Projections per layer:          4  (W_q, W_k, W_v, W_o)
Matrices per projection:        2  (A, B)
Params per matrix pair:     2,048  (rank=4, d=256)
─────────────────────────────────────────────────────
Total LoRA params:  6 × 4 × 2,048 = 49,152   ≈ 48K
```

48K trainable parameters. 6M frozen. **0.8% of the model trains. 99.2% is frozen.**

---

## Summary

| Concept | What it is | Why it matters |
|---|---|---|
| `W₀` frozen | Original pre-trained weight | Preserves language knowledge, never changes |
| `A` matrix | Down-projection `R^{r×k}`, random init | Projects input into low-rank subspace |
| `B` matrix | Up-projection `R^{d×r}`, zero init | Stable start: ΔW=0 at step 0 |
| `α/r` scaling | Normalises adapter magnitude | Decouples rank choice from learning rate |
| `r = 4` | Rank of the update | 0.8% of params trainable, captures task signal |
| Merge at deploy | `W_final = W₀ + (α/r)BA` | Zero inference overhead after fine-tuning |

The insight that makes LoRA profound is not the engineering — it is the empirical observation that **fine-tuning is intrinsically low-dimensional**. The mathematical machinery (rank factorisation, zero init, scale normalisation) is elegant precisely because it is the minimum structure needed to exploit that observation.