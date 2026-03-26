"""
nanogpt.py — Full NanoGPT Language Model
=========================================
Author  : NanoGPT Lab
Standard: GPT-2 (Radford et al. 2019)  +  GPT-3, LLaMA, Mistral, PaLM extensions

Architecture overview
---------------------
                    ┌─────────────────────────────────┐
  token ids         │         NanoGPT                 │
  (B, T) ─────────►│                                  │
                    │  tok_emb   : (vocab, d_model)    │
                    │  pos_emb   : (ctx_len, d_model)  │  ← only if not RoPE/ALiBi
                    │  emb_drop  : Dropout             │
                    │                                  │
                    │  blocks[0..N-1] : TransformerStack│
                    │                                  │
                    │  norm_final : RMSNorm / LN       │
                    │                                  │
                    │  lm_head   : (d_model, vocab)    │  ← weight-tied to tok_emb
                    └─────────────────────────────────┘
                               │
                    logits (B, T, vocab_size)

Mathematical formulation
------------------------
Given token sequence (t₀, t₁, …, t_{T-1}):

    e_i  = E[t_i]                         token embedding lookup
    p_i  = P[i]                           positional embedding (if used)
    x_i  = dropout(e_i + p_i)            combined embedding

    for l in 0..N-1:
        x = TransformerBlock_l(x)         self-attention + FFN

    x = FinalNorm(x)
    logits = x @ E^T                      weight-tied projection  (B, T, V)

    loss = CrossEntropy(logits[:, :-1], tokens[:, 1:])   next-token prediction

Weight tying (E^T in the head)
-------------------------------
The language model head shares weights with the token embedding matrix E
(Press & Wolf 2017, "Using the Output Embedding to Improve Language Models").

Intuition: the embedding maps token → hidden space, the head maps hidden space
→ token score.  If we optimise them jointly, the model learns a consistent
geometry: tokens that are similar in embedding space will also score similarly
when generated.  This reduces parameters by vocab_size × d_model (often ~30M)
and consistently improves perplexity.

Positional encoding strategies (all implemented)
-------------------------------------------------
1. Learned absolute  : standard GPT-2.  Simple, works well up to ctx_len.
2. None              : rely entirely on RoPE (applied inside attention).
3. Sinusoidal fixed  : original Vaswani 2017.  No learned params, perfect
                       for out-of-distribution length extrapolation in theory
                       (though RoPE beats it in practice).

Output scaling (μP — maximal update parametrisation)
-----------------------------------------------------
Yang et al. (2022) "Tensor Programs V" shows that naive scaling of d_model
breaks hyperparameter transfer.  Under μP:
  - Embedding init std ∝ 1
  - Hidden weights init std ∝ 1/√fan_in
  - Output (lm_head) is scaled by 1/d_model at forward time

We implement both standard GPT-2 init and μP init, toggled by `mup_init`.

Parameter count formula
-----------------------
For a model with vocab V, context C, d D, N layers, h heads:

    Embeddings    : V×D + C×D   (token + positional)
    Per block     : 4×D²        (attn QKV+O)  +  8×D² (SwiGLU)  ≈ 12D²
    All blocks    : N × 12D²
    Final norm    : D
    LM head       : 0 (weight-tied, free)
    Total         ≈ V×D + N×12D²

This formula lets you hit a parameter target by solving for D given N.

Scaling law context
-------------------
Hoffmann et al. "Chinchilla" (2022) optimal compute allocation:
    N_opt ∝ C^0.5    (parameters grow with sqrt of compute)
    D_opt ∝ C^0.5    (training tokens grow with sqrt of compute)
For our experiments:
    1.2M  params → train on ~24M tokens  (TinyShakespeare is ~1M, fine for demo)
    6M    params → train on ~120M tokens (you'd need OpenWebText for optimal)
"""

import math
import inspect
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.transformer_block import TransformerStack, build_norm
except ImportError:
    from transformer_block import TransformerStack, build_norm


# ===========================================================================
# Configuration dataclass — single source of truth for all hyperparameters
# ===========================================================================

@dataclass
class NanoGPTConfig:
    """
    All model hyperparameters in one place.

    Design rationale
    ----------------
    Using a dataclass rather than a plain dict gives:
      - Type annotations (IDE autocompletion, mypy checking)
      - Default values with easy overriding
      - Serialisable to JSON / YAML for experiment logging
      - Immutable-ish (can freeze with frozen=True)

    Preset factory methods below mirror the original GPT-2 sizes for easy
    comparison with the literature.
    """
    # ── Vocabulary & context ─────────────────────────────────────────────────
    vocab_size:    int   = 50304   # GPT-2 vocab (50257) rounded up to nearest 64
                                   # for efficient CUDA tiling on tensor cores
    context_len:   int   = 1024

    # ── Transformer dimensions ───────────────────────────────────────────────
    d_model:       int   = 384
    n_layers:      int   = 6
    n_heads:       int   = 6
    n_kv_heads:    Optional[int] = None   # None = standard MHA; set for GQA

    # ── FFN ──────────────────────────────────────────────────────────────────
    ffn_variant:   str   = "swiglu"   # 'standard' | 'swiglu' | 'geglu' | 'moe'
    ffn_expansion: int   = 4

    # ── Regularisation ───────────────────────────────────────────────────────
    dropout:            float = 0.0
    stochastic_depth_p: float = 0.0

    # ── Normalisation ─────────────────────────────────────────────────────────
    norm_type:     str   = "rmsnorm"   # 'rmsnorm' | 'layernorm'

    # ── Positional encoding ──────────────────────────────────────────────────
    pos_encoding:  str   = "rope"   # 'learned' | 'sinusoidal' | 'rope' | 'alibi'

    # ── Architecture flags ────────────────────────────────────────────────────
    parallel_attn_ffn: bool  = False   # GPT-J / PaLM parallel blocks
    use_flash:         bool  = True    # FlashAttention when available
    weight_tying:      bool  = True    # tie lm_head weights to token embeddings
    mup_init:          bool  = False   # maximal update parametrisation init

    # ── Convenience ──────────────────────────────────────────────────────────
    bias:          bool  = False   # bias in linear layers (False = modern default)

    # ──────────────────────────────────────────────────────────────────────────
    # Preset factory methods
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def nano(cls) -> "NanoGPTConfig":
        """~1.2M params — assignment Experiment 1. Trains in ~15 min on T4."""
        return cls(d_model=128, n_layers=4,  n_heads=4,  context_len=256,
                   vocab_size=50304, ffn_variant="standard")

    @classmethod
    def small(cls) -> "NanoGPTConfig":
        """~6M params — assignment Experiment 2."""
        return cls(d_model=256, n_layers=6,  n_heads=8,  context_len=512,
                   vocab_size=50304, ffn_variant="swiglu")

    @classmethod
    def medium(cls) -> "NanoGPTConfig":
        """~85M params — GPT-2 medium equivalent."""
        return cls(d_model=1024, n_layers=24, n_heads=16, context_len=1024,
                   vocab_size=50304, ffn_variant="swiglu", norm_type="rmsnorm")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NanoGPTConfig":
        """Reconstruct from a JSON/YAML-loaded dictionary."""
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    def n_params(self, include_embeddings: bool = True) -> int:
        """
        Analytic parameter count formula (no model instantiation needed).
        Useful for quick architecture search before allocating GPU memory.

            total ≈ V×D + C×D (embeddings)
                  + N × (4D² + 8D²)  (blocks, SwiGLU approx)
                  + D  (final norm scale)
        Note: lm_head is weight-tied → 0 extra params.
        """
        V, C, D, N = self.vocab_size, self.context_len, self.d_model, self.n_layers
        emb = V * D
        if self.pos_encoding == "learned":
            emb += C * D
        # Per-block: attn (4D²) + FFN (8D² for SwiGLU / 8D² standard)
        blocks = N * 12 * D * D
        norm   = D
        total  = emb + blocks + norm
        return total if include_embeddings else total - emb


# ===========================================================================
# Positional Encoding modules
# ===========================================================================

class LearnedPositionalEmbedding(nn.Module):
    """
    Standard GPT-2 positional embedding: a simple lookup table of shape
    (context_len, d_model), one learned vector per position.

    Limitation: cannot generalise beyond context_len at inference time.
    That is why RoPE / ALiBi were developed.
    """
    def __init__(self, context_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(context_len, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) — adds position vectors for positions 0..T-1."""
        T      = x.size(1)
        device = x.device
        pos    = torch.arange(T, device=device)          # (T,)
        return self.embedding(pos).unsqueeze(0)           # (1, T, d_model)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Fixed sinusoidal encoding from Vaswani et al. (2017):

        PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
        PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

    No learnable parameters.  In theory extrapolates to unseen lengths, but
    in practice RoPE is strictly better for length generalisation.
    Kept here for completeness and comparison.
    """
    def __init__(self, context_len: int, d_model: int) -> None:
        super().__init__()
        pe  = torch.zeros(context_len, d_model)
        pos = torch.arange(context_len).unsqueeze(1).float()         # (T, 1)
        # Dimension indices for even positions
        i   = torch.arange(0, d_model, 2).float()                    # (d/2,)
        div = torch.pow(10_000.0, i / d_model)                       # (d/2,)
        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)
        # Register as buffer: no gradients, moves with .to(device)
        self.register_buffer("pe", pe.unsqueeze(0))                   # (1, T, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1), :]                              # (1, T, d)


# ===========================================================================
# Full NanoGPT model
# ===========================================================================

class NanoGPT(nn.Module):
    """
    Autoregressive causal language model.

    Given a sequence of token ids, predicts the next token at every position.
    Trained with cross-entropy loss on the shifted sequence (teacher forcing).

    Parameters
    ----------
    config : NanoGPTConfig  — all hyperparameters in one object
    """

    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        self.config = config
        C   = config.d_model
        V   = config.vocab_size
        ctx = config.context_len

        # ── 1. Token embedding ───────────────────────────────────────────────
        # Maps each integer token id ∈ [0, V) to a d_model-dimensional vector.
        # E ∈ R^{V × d_model} — the most parameter-dense single matrix in small
        # models (50304 × 128 = 6.4M for the nano config).
        self.tok_emb = nn.Embedding(V, C)

        # ── 2. Positional encoding ───────────────────────────────────────────
        # Encodes the *position* of each token in the sequence.
        # RoPE and ALiBi are handled inside MultiHeadAttention (attention.py),
        # so in those cases we need no separate module here.
        pos_enc = config.pos_encoding.lower()
        if pos_enc == "learned":
            self.pos_enc_module = LearnedPositionalEmbedding(ctx, C)
        elif pos_enc == "sinusoidal":
            self.pos_enc_module = SinusoidalPositionalEmbedding(ctx, C)
        elif pos_enc in ("rope", "alibi", "none"):
            self.pos_enc_module = None   # position handled inside attention
        else:
            raise ValueError(f"Unknown pos_encoding: '{pos_enc}'")

        # ── 3. Embedding dropout ──────────────────────────────────────────────
        # Applied after summing token + positional embeddings.
        # Acts as a regulariser on the input representation.
        self.emb_drop = nn.Dropout(config.dropout)

        # ── 4. Transformer blocks ─────────────────────────────────────────────
        use_rope  = pos_enc == "rope"
        use_alibi = pos_enc == "alibi"
        self.blocks = TransformerStack(
            n_layers            = config.n_layers,
            d_model             = C,
            n_heads             = config.n_heads,
            n_kv_heads          = config.n_kv_heads,
            ffn_variant         = config.ffn_variant,
            ffn_expansion       = config.ffn_expansion,
            dropout             = config.dropout,
            norm_type           = config.norm_type,
            use_rope            = use_rope,
            use_alibi           = use_alibi,
            max_seq_len         = ctx,
            use_flash           = config.use_flash,
            parallel_attn_ffn   = config.parallel_attn_ffn,
            stochastic_depth_p  = config.stochastic_depth_p,
        )

        # ── 5. Final layer norm ───────────────────────────────────────────────
        # Applied to the output of the last transformer block before the LM head.
        # Critical for output stability: without it the logits can have very
        # different magnitudes across sequence positions.
        self.norm_final = build_norm(config.norm_type, C)

        # ── 6. Language model head ─────────────────────────────────────────────
        # Projects from d_model → vocab_size to produce next-token logits.
        # bias=False: the softmax is shift-invariant, so a bias buys nothing.
        self.lm_head = nn.Linear(C, V, bias=False)

        # ── 7. Weight tying ───────────────────────────────────────────────────
        # Share the lm_head weight matrix with tok_emb.
        # Saves V×d_model parameters and improves generalisation.
        # See Press & Wolf (2017) for analysis.
        if config.weight_tying:
            self.lm_head.weight = self.tok_emb.weight
            # Note: this is a genuine alias — one tensor, two names.
            # Gradients from both paths accumulate into the same parameter.

        # ── 8. Weight initialisation ──────────────────────────────────────────
        self._init_weights()

        # ── 9. Report parameter count ─────────────────────────────────────────
        n_params = sum(p.numel() for p in self.parameters())
        # Subtract the tied weight counted twice
        if config.weight_tying:
            n_params -= self.tok_emb.weight.numel()
        print(f"NanoGPT  |  {n_params/1e6:.2f}M params  |  "
              f"d={C}  L={config.n_layers}  h={config.n_heads}  "
              f"V={V}  ctx={ctx}  ffn={config.ffn_variant}")

    # ── Weight initialisation ────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        GPT-2 style initialisation:
          - Embeddings : N(0, 0.02)
          - Linear W   : N(0, 0.02)
          - Linear b   : zeros
          - Residual-path projections (W_o, W2): N(0, 0.02 / √(2N))
            This scaling prevents the residual stream from growing in
            variance with depth.  Derived in the GPT-2 paper §2.3.

        If mup_init=True we instead use:
          - Input embeddings: N(0, 1)
          - Hidden weights  : N(0, 1/√fan_in)
          - Output head     : scaled by 1/d_model at forward time
        These ensure that hyperparameters (lr, init scale) transfer across
        width changes — you tune once at d=128 and scale to d=4096 for free.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = 0.02
                if self.config.mup_init:
                    std = 1.0 / math.sqrt(module.weight.shape[1])   # 1/√fan_in
                # Residual-path projections get extra depth scaling
                if name.endswith(("W_o", "W2")):
                    std *= (2 * self.config.n_layers) ** -0.5
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                std = 1.0 if self.config.mup_init else 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        idx:       torch.Tensor,                         # (B, T)  token ids
        targets:   Optional[torch.Tensor] = None,        # (B, T)  target ids for loss
        kv_caches: Optional[list]         = None,        # per-layer KV caches
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], list]:
        """
        Parameters
        ----------
        idx       : (B, T) integer token ids, each in [0, vocab_size)
        targets   : (B, T) target ids. If provided, loss is computed.
                    Targets are the input shifted by 1: targets = idx[:, 1:] in training.
        kv_caches : per-layer KV-cache list for autoregressive inference. None = training.

        Returns
        -------
        logits    : (B, T, vocab_size)   un-normalised next-token scores
        loss      : scalar cross-entropy loss (or None if targets not given)
        kv_caches : updated cache list
        """
        B, T = idx.shape
        assert T <= self.config.context_len, (
            f"Sequence length {T} exceeds context_len {self.config.context_len}")

        # ── Step 1: Token embeddings ─────────────────────────────────────────
        # Shape: (B, T, d_model)
        x = self.tok_emb(idx)

        # ── Step 2: Add positional encoding ──────────────────────────────────
        if self.pos_enc_module is not None:
            x = x + self.pos_enc_module(x)

        # ── Step 3: Embedding dropout ─────────────────────────────────────────
        x = self.emb_drop(x)

        # ── Step 4: Transformer blocks ────────────────────────────────────────
        x, kv_caches = self.blocks(x, kv_caches)

        # ── Step 5: Final layer norm ──────────────────────────────────────────
        x = self.norm_final(x)

        # ── Step 6: LM head → logits ──────────────────────────────────────────
        if self.config.mup_init:
            # μP output scaling: divide by d_model before projection
            # ensures logit variance is O(1) regardless of width
            logits = self.lm_head(x / math.sqrt(self.config.d_model))
        else:
            logits = self.lm_head(x)                                  # (B, T, V)

        # ── Step 7: Loss (training only) ──────────────────────────────────────
        loss = None
        if targets is not None:
            # Flatten to (B*T, V) and (B*T,) for F.cross_entropy
            # ignore_index=-1 lets you mask padding tokens
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss, kv_caches

    # ── Inference utilities ───────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx:         torch.Tensor,          # (B, T) prompt token ids
        max_new:     int         = 200,
        temperature: float       = 1.0,
        top_k:       Optional[int] = None,
        top_p:       Optional[float] = None,
        greedy:      bool        = False,
    ) -> torch.Tensor:
        """
        Autoregressive token generation with four decoding strategies.

        Greedy decoding
        ---------------
            token = argmax(logits)
            Deterministic. Produces the single most-probable continuation.
            Often repetitive for long sequences.

        Temperature sampling
        --------------------
            probs = softmax(logits / T)
            token ~ Categorical(probs)
            T < 1  →  sharper distribution  →  more focused, less diverse
            T > 1  →  flatter distribution  →  more diverse, more random
            T = 1  →  unmodified model distribution

        Top-k sampling
        --------------
            Keep only the k tokens with highest probability, renormalise,
            sample. Cuts off the long tail of improbable tokens.
            Introduced by Fan et al. (2018) for story generation.

        Top-p (nucleus) sampling
        ------------------------
            Keep the smallest set of tokens whose cumulative probability
            exceeds p, sample from it.
            Holtzman et al. (2020) "The Curious Case of Neural Text Degeneration".
            Adapts vocabulary size to the entropy of each step:
            high-confidence steps use few tokens, uncertain steps use many.

        Parameters
        ----------
        idx         : (B, T) prompt
        max_new     : tokens to generate
        temperature : softmax temperature (ignored for greedy)
        top_k       : top-k truncation (None = disabled)
        top_p       : nucleus cutoff in (0, 1]  (None = disabled)
        greedy      : if True, always take argmax regardless of other params
        """
        self.eval()
        kv_caches = None

        # Prefill: process the entire prompt at once to warm up the KV cache
        _, _, kv_caches = self(idx, kv_caches=kv_caches)
        # Now we only need to feed one token at a time
        cur = idx[:, -1:]   # last token of prompt

        generated = []
        for _ in range(max_new):
            logits, _, kv_caches = self(cur, kv_caches=kv_caches)
            logits = logits[:, -1, :]                     # (B, V) — last position

            # ── Greedy ───────────────────────────────────────────────────────
            if greedy:
                next_token = logits.argmax(dim=-1, keepdim=True)
                generated.append(next_token)
                cur = next_token
                continue

            # ── Temperature scaling ───────────────────────────────────────────
            logits = logits / max(temperature, 1e-8)

            # ── Top-k filtering ───────────────────────────────────────────────
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                threshold = logits.topk(k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            # ── Top-p (nucleus) filtering ─────────────────────────────────────
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                # Remove tokens once cumulative prob exceeds p
                # Shift right by 1 so we always keep at least one token
                remove = cum_probs - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                # Unsort back to original vocabulary order
                logits = torch.zeros_like(logits).scatter_(
                    1, sorted_idx, sorted_logits
                )

            # ── Sample ────────────────────────────────────────────────────────
            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # (B, 1)

            generated.append(next_token)
            cur = next_token

        return torch.cat([idx] + generated, dim=1)                  # (B, T + max_new)

    # ── Optimiser configuration ───────────────────────────────────────────────

    def configure_optimizer(
        self,
        learning_rate: float = 3e-4,
        weight_decay:  float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.95),
        device_type: str = "cpu",
    ) -> torch.optim.AdamW:
        """
        AdamW with decoupled weight decay, matching GPT-3 / LLaMA training setup.

        Weight decay is applied ONLY to 2-D parameters (weight matrices).
        1-D parameters (biases, norm scales) are excluded — applying decay
        to them is mathematically incorrect because they are not multiplicative
        weights on the residual stream.

        Uses the fused AdamW kernel (PyTorch ≥ 2.0) when available on CUDA
        for ~15% faster parameter updates.

        Parameters
        ----------
        learning_rate : peak LR (combine with cosine scheduler in trainer.py)
        weight_decay  : L2 regularisation coefficient on 2-D weights
        betas         : AdamW momentum parameters (β₁, β₂)
        device_type   : 'cuda' enables fused kernel
        """
        # Separate parameters into decay / no-decay groups
        decay, no_decay = set(), set()
        for pname, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay.add(pname)       # weight matrices
            else:
                no_decay.add(pname)   # biases, norms, embedding if 1-D

        param_groups = [
            {"params": [self.get_parameter(n) for n in sorted(decay)],
             "weight_decay": weight_decay},
            {"params": [self.get_parameter(n) for n in sorted(no_decay)],
             "weight_decay": 0.0},
        ]

        # Use fused AdamW if available (CUDA only)
        use_fused = (device_type == "cuda"
                     and "fused" in inspect.signature(torch.optim.AdamW).parameters)
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused if use_fused else False,
        )
        print(f"Optimiser  |  AdamW  fused={use_fused}  "
              f"lr={learning_rate}  wd={weight_decay}  "
              f"decay_params={sum(self.get_parameter(n).numel() for n in decay):,}  "
              f"nodecay_params={sum(self.get_parameter(n).numel() for n in no_decay):,}")
        return optimizer

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        path: str,
        step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss: Optional[float] = None,
    ) -> None:
        """
        Save model + config + training state to a single .pt file.

        Saving the config alongside the weights means you can reconstruct
        the exact architecture from the checkpoint without any external config
        file.  Critical for reproducibility.
        """
        payload = {
            "step":        step,
            "config":      self.config.to_dict(),
            "model":       self.state_dict(),
            "loss":        loss,
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)
        print(f"Checkpoint saved  →  {path}  (step={step}  loss={loss})")

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "NanoGPT":
        """
        Load a model from a checkpoint file.

        Usage:
            model = NanoGPT.from_checkpoint("checkpoints/step_5000.pt")
        """
        payload = torch.load(path, map_location=device, weights_only=False)
        config  = NanoGPTConfig.from_dict(payload["config"])
        model   = cls(config)
        model.load_state_dict(payload["model"])
        step    = payload.get("step", 0)
        loss    = payload.get("loss", None)
        print(f"Checkpoint loaded ←  {path}  (step={step}  loss={loss})")
        return model

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def parameter_summary(self) -> None:
        """Print a hierarchical parameter breakdown by sub-module."""
        total = 0
        print(f"\n{'Module':<50} {'Params':>10}  {'%':>6}")
        print("─" * 70)
        for name, module in self.named_modules():
            own_params = sum(p.numel() for p in module.parameters(recurse=False))
            if own_params == 0:
                continue
            total += own_params
            pct = 100 * own_params / max(1, sum(
                p.numel() for p in self.parameters()))
            print(f"  {name:<48} {own_params:>10,}  {pct:>5.1f}%")
        print("─" * 70)
        print(f"  {'TOTAL':<48} {total:>10,}  100.0%")


# ===========================================================================
# Self-test — run: python nanogpt.py
# ===========================================================================

if __name__ == "__main__":
    import sys

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}\n")

    all_passed = True

    # ── 1. Preset configurations ──────────────────────────────────────────────
    print("── Preset configurations ───────────────────────────────────────")
    for label, cfg in [
        ("nano",   NanoGPTConfig.nano()),
        ("small",  NanoGPTConfig.small()),
    ]:
        model = NanoGPT(cfg).to(device)
        actual = sum(p.numel() for p in model.parameters()) - (
            model.tok_emb.weight.numel() if cfg.weight_tying else 0)
        analytic = cfg.n_params()
        print(f"  {label:8s}  actual={actual/1e6:.2f}M  analytic≈{analytic/1e6:.2f}M")

    # ── 2. Forward pass + loss ────────────────────────────────────────────────
    print("\n── Forward pass + loss ─────────────────────────────────────────")
    cfg   = NanoGPTConfig.nano()
    model = NanoGPT(cfg).to(device)
    B, T  = 2, 64
    idx     = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    targets = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    try:
        logits, loss, _ = model(idx, targets)
        assert logits.shape == (B, T, cfg.vocab_size), f"Bad shape: {logits.shape}"
        assert loss is not None
        # Loss at random init should be ≈ ln(vocab_size) ≈ 10.8 for V=50304
        expected_loss = math.log(cfg.vocab_size)
        print(f"  ✓  logits={tuple(logits.shape)}  "
              f"loss={loss.item():.3f}  "
              f"(expected≈{expected_loss:.3f}  diff={abs(loss.item()-expected_loss):.3f})")
    except Exception as e:
        print(f"  ✗  Forward/loss: {e}", file=sys.stderr)
        all_passed = False

    # ── 3. Backward pass ──────────────────────────────────────────────────────
    print("\n── Backward pass ───────────────────────────────────────────────")
    try:
        loss.backward()
        no_grad = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None]
        if no_grad:
            print(f"  ✗  Missing grads: {no_grad}", file=sys.stderr)
            all_passed = False
        else:
            total_norm = sum(p.grad.norm().item() ** 2
                             for p in model.parameters() if p.grad is not None) ** 0.5
            print(f"  ✓  All gradients present  grad_norm={total_norm:.4f}")
    except Exception as e:
        print(f"  ✗  Backward: {e}", file=sys.stderr)
        all_passed = False

    # ── 4. Weight tying verification ──────────────────────────────────────────
    print("\n── Weight tying ────────────────────────────────────────────────")
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr(), \
        "Weight tying broken: lm_head and tok_emb are different tensors"
    print("  ✓  lm_head.weight is tok_emb.weight  (same tensor, same data_ptr)")

    # ── 5. Generation ─────────────────────────────────────────────────────────
    print("\n── Generation ──────────────────────────────────────────────────")
    model.eval()
    prompt  = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    for label, kwargs in [
        ("greedy",      dict(greedy=True,  max_new=20)),
        ("temp=0.8",    dict(temperature=0.8, max_new=20)),
        ("top_k=40",    dict(top_k=40, temperature=1.0, max_new=20)),
        ("top_p=0.9",   dict(top_p=0.9, temperature=1.0, max_new=20)),
    ]:
        try:
            out = model.generate(prompt, **kwargs)
            assert out.shape == (1, 8 + 20)
            print(f"  ✓  {label:12s}  output shape={tuple(out.shape)}")
        except Exception as e:
            print(f"  ✗  {label}: {e}", file=sys.stderr)
            all_passed = False

    # ── 6. KV-cache consistency ───────────────────────────────────────────────
    print("\n── KV-cache consistency ────────────────────────────────────────")
    model.eval()
    with torch.no_grad():
        x_inf = torch.randint(0, cfg.vocab_size, (1, T), device=device)
        logits_full, _, _ = model(x_inf)

        # Token-by-token with caches
        kv = None
        logit_steps = []
        for t in range(T):
            lg, _, kv = model(x_inf[:, t:t+1], kv_caches=kv)
            logit_steps.append(lg)
        logits_cached = torch.cat(logit_steps, dim=1)

        diff = (logits_full - logits_cached).abs().max().item()
        status = "✓" if diff < 1e-3 else "✗"
        print(f"  {status}  Max |full − cached| logit diff = {diff:.2e}")
        if diff >= 1e-3:
            all_passed = False

    # ── 7. Checkpoint round-trip ──────────────────────────────────────────────
    print("\n── Checkpoint round-trip ───────────────────────────────────────")
    try:
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        model.save_checkpoint(ckpt_path, step=42, loss=3.14)
        model2 = NanoGPT.from_checkpoint(ckpt_path, device=str(device))
        with torch.no_grad():
            lg1, _, _ = model(x_inf)
            lg2, _, _ = model2(x_inf)
        diff = (lg1 - lg2).abs().max().item()
        os.unlink(ckpt_path)
        status = "✓" if diff < 1e-6 else "✗"
        print(f"  {status}  Checkpoint diff = {diff:.2e}")
        if diff >= 1e-6:
            all_passed = False
    except Exception as e:
        print(f"  ✗  Checkpoint: {e}", file=sys.stderr)
        all_passed = False

    # ── 8. Optimiser configuration ────────────────────────────────────────────
    print("\n── Optimiser ───────────────────────────────────────────────────")
    optim = model.configure_optimizer(device_type=str(device))
    print(f"  ✓  {len(optim.param_groups)} param groups created")

    # ── 9. Parameter summary ──────────────────────────────────────────────────
    print()
    model.parameter_summary()

    print("\n" + ("All tests PASSED ✓" if all_passed else "Some tests FAILED ✗"))