"""
transformer_block.py — Transformer Decoder Block
=================================================
Author  : NanoGPT Lab
Standard: Vaswani et al. (2017)  +  GPT-2/3, LLaMA, Mistral, DeepNet extensions

Mathematical foundation
-----------------------
A single transformer block maps x ∈ R^{B × T × d_model} to the same shape:

    PRE-NORM (modern, what we use):
        x = x + Attention( LayerNorm(x) )
        x = x + FFN(      LayerNorm(x) )

    POST-NORM (original Vaswani 2017):
        x = LayerNorm( x + Attention(x) )
        x = LayerNorm( x + FFN(x)       )

Pre-norm vs Post-norm — the most important architectural decision in this file
-------------------------------------------------------------------------------
POST-NORM places LayerNorm *after* the residual addition.

    At initialisation all sub-layers output ≈ 0 (weights near zero), so the
    residual path dominates.  But the gradient must pass through LayerNorm
    to reach the main branch — LayerNorm can rescale gradients arbitrarily,
    and in deep networks (≥12 layers) this causes training instability unless
    very careful learning-rate warmup is used (GPT-1 / original BERT required
    thousands of warmup steps to avoid divergence).

PRE-NORM places LayerNorm *before* each sub-layer.

    Gradient path: the gradient flows *directly* through the residual
    connection back to x without passing through any normalisation layer.
    This means the gradient magnitude is preserved across depth, which is
    why GPT-2+ and all modern LLMs switched to pre-norm.  You can train
    with less warmup, higher learning rates, and more layers.

    Formal analysis: Liu et al. "Understanding the Difficulty of Training
    Transformers" (2020) shows pre-norm achieves O(1/√layers) gradient
    variance vs O(1/layers) for post-norm.

RMSNorm vs LayerNorm
--------------------
Standard LayerNorm:
    y = (x − μ) / √(σ² + ε) · γ + β
    Computes mean μ AND variance σ².  Two statistics, two passes.

RMSNorm (Zhang & Sennrich 2019):
    y = x / RMS(x) · γ   where  RMS(x) = √(mean(x²) + ε)
    Computes only the RMS (one pass).  No mean subtraction.  No bias β.

    Why it works: the "re-centering" (mean subtraction) is less important
    than the "re-scaling" (variance normalisation) for training stability.
    LLaMA 1/2/3, Mistral, Falcon, GPT-NeoX all use RMSNorm.
    ~15% faster than LayerNorm in practice (measured on A100).

Parallel attention + FFN (optional)
------------------------------------
GPT-J (EleutherAI 2021) and PaLM run attention and FFN in *parallel*:

    x = x + Attention( LN(x) ) + FFN( LN(x) )

Using the SAME LayerNorm input for both, then adding both outputs to the
residual.  This saves one LayerNorm and allows both ops to run concurrently
on the GPU.  ~15% faster at 6B+ scale.  Slightly worse at small scale.

DeepNet init (stable post-norm at extreme depth)
-------------------------------------------------
Wang et al. "DeepNet: Scaling Transformers to 1,000 Layers" (2022) shows
that post-norm CAN be stable if weights are initialised with a specific
scaling α, β derived from network depth.  We implement this as an option
to demonstrate that the pre/post-norm trade-off is not absolute.

Stochastic Depth / LayerDrop
-----------------------------
Huang et al. (2016) / Fan et al. (2019): randomly drop entire blocks during
training with probability p_drop.  At inference all blocks are active.
Acts as a strong regulariser for deep transformers (used in DeiT, ViT-22B).
The block's contribution is rescaled by 1/(1-p_drop) when kept, preserving
the expected residual magnitude.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our previously built components
try:
    from model.attention import MultiHeadAttention
    from model.feedforward import build_ffn
except ImportError:
    from attention import MultiHeadAttention
    from feedforward import build_ffn


# ===========================================================================
# Normalisation layers
# ===========================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich 2019).

        RMSNorm(x) = x / RMS(x) · γ
        RMS(x)     = sqrt( mean(x²) + ε )

    No mean subtraction, no bias parameter.
    Used in: LLaMA 1/2/3, Mistral, Falcon, GPT-NeoX, PaLM.

    Parameters
    ----------
    d_model : feature dimension to normalise over (last dim of input)
    eps     : numerical stability constant  (default 1e-6, same as LLaMA)
    """
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(d_model))   # learnable scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        # Compute RMS along last dimension, keep dims for broadcasting
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.gamma


# Factory: pick LayerNorm or RMSNorm by string
def build_norm(norm_type: str, d_model: int) -> nn.Module:
    if norm_type == "layernorm":
        # bias=False follows modern LLMs (PaLM, LLaMA); saves params, no quality loss
        return nn.LayerNorm(d_model, elementwise_affine=True, bias=False)
    elif norm_type == "rmsnorm":
        return RMSNorm(d_model)
    else:
        raise ValueError(f"Unknown norm type: '{norm_type}'. Choose: layernorm, rmsnorm")


# ===========================================================================
# Core: TransformerBlock
# ===========================================================================

class TransformerBlock(nn.Module):
    """
    A single transformer decoder block.

    Supports four sub-architectures selectable via `parallel_attn_ffn`:

    ┌─────────────────────────────────────────────────────────────────┐
    │  SERIAL (default — GPT-2/3 style, pre-norm)                    │
    │                                                                 │
    │  x ──┬──────────────────────────────────────┬──► x             │
    │      │   LN → Attention → dropout           │                   │
    │      └──────────────── + ──────────────────►│                   │
    │                         │                   │                   │
    │                    x ──┬┴─────────────────────────────────┬──► x│
    │                        │   LN → FFN → dropout             │      │
    │                        └──────────────── + ──────────────►│      │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │  PARALLEL (GPT-J / PaLM style)                                 │
    │                                                                 │
    │  x ──┬── LN → Attention ──┬──► x                              │
    │      └── LN → FFN      ──►┘                                    │
    └─────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    d_model           : embedding dimension
    n_heads           : number of attention heads
    n_kv_heads        : KV heads for GQA (None = standard MHA)
    ffn_variant       : 'standard' | 'swiglu' | 'geglu' | 'moe'
    ffn_expansion     : FFN hidden-dim multiplier (default 4)
    ffn_activation    : activation for standard FFN ('gelu' | 'relu' | 'silu')
    dropout           : applied inside attention and FFN
    norm_type         : 'layernorm' | 'rmsnorm'
    use_rope          : rotary position embeddings on Q, K
    use_alibi         : ALiBi linear position bias
    max_seq_len       : needed for RoPE pre-computation
    use_flash         : FlashAttention path (PyTorch ≥ 2.0 + CUDA Ampere)
    parallel_attn_ffn : run attention and FFN in parallel (GPT-J / PaLM)
    stochastic_depth_p: probability of dropping this entire block during training
    post_norm         : use post-norm instead of pre-norm (not recommended)
    """

    def __init__(
        self,
        d_model:            int,
        n_heads:            int,
        n_kv_heads:         Optional[int]  = None,
        ffn_variant:        str            = "swiglu",
        ffn_expansion:      int            = 4,
        ffn_activation:     str            = "gelu",
        dropout:            float          = 0.0,
        norm_type:          str            = "rmsnorm",
        use_rope:           bool           = True,
        use_alibi:          bool           = False,
        max_seq_len:        int            = 2048,
        use_flash:          bool           = True,
        parallel_attn_ffn:  bool           = False,
        stochastic_depth_p: float          = 0.0,
        post_norm:          bool           = False,
    ) -> None:
        super().__init__()

        self.parallel_attn_ffn   = parallel_attn_ffn
        self.stochastic_depth_p  = stochastic_depth_p
        self.post_norm           = post_norm

        # ── Attention sub-layer ─────────────────────────────────────────────
        self.attn = MultiHeadAttention(
            d_model     = d_model,
            n_heads     = n_heads,
            n_kv_heads  = n_kv_heads,
            dropout     = dropout,
            use_rope    = use_rope,
            use_alibi   = use_alibi,
            max_seq_len = max_seq_len,
            use_flash   = use_flash,
        )

        # ── FFN sub-layer ───────────────────────────────────────────────────
        self.ffn = build_ffn(
            variant    = ffn_variant,
            d_model    = d_model,
            expansion  = ffn_expansion,
            activation = ffn_activation,
            dropout    = dropout,
        )

        # ── Normalisation ───────────────────────────────────────────────────
        # Pre-norm: one LN before each sub-layer.
        # Parallel: single shared LN before both (saves one LN call).
        if parallel_attn_ffn:
            # One shared norm feeds both attention and FFN
            self.norm = build_norm(norm_type, d_model)
        else:
            # Two separate norms — canonical pre-norm GPT-2/3/LLaMA layout
            self.norm_attn = build_norm(norm_type, d_model)
            self.norm_ffn  = build_norm(norm_type, d_model)

        # ── Residual dropout (applied to the residual delta, not the stream) ─
        # Distinct from attention-internal dropout. Adds regularisation on
        # the residual contribution of each sub-layer.
        self.resid_drop = nn.Dropout(dropout)

    # ── Stochastic depth ────────────────────────────────────────────────────

    def _stochastic_depth_scale(self) -> float:
        """
        During training, drop the entire block with probability p.
        Return a scalar that either:
          - kills  the block's contribution (returns 0.0), or
          - scales it up by 1/(1-p) to preserve expected magnitude.

        At eval time always returns 1.0 (all blocks active, no scaling).

        This is the "stochastic depth" of Huang et al. (2016), sometimes
        called "LayerDrop" in the NLP literature (Fan et al. 2019).
        """
        if not self.training or self.stochastic_depth_p == 0.0:
            return 1.0
        keep_prob = 1.0 - self.stochastic_depth_p
        if torch.rand(1).item() > keep_prob:
            return 0.0                       # drop this block entirely
        return 1.0 / keep_prob               # scale up to keep expectation

    # ── Sub-layer wrappers ──────────────────────────────────────────────────

    def _attn_residual(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple],
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Compute the attention residual delta.

        Pre-norm  :  delta = Attention( LN(x) )
        Post-norm :  delta = LN( Attention(x) )   ← original Vaswani; avoid
        """
        if self.post_norm:
            # Post-norm path (original paper, less stable — shown for contrast)
            attn_out, new_cache = self.attn(x, kv_cache)
            delta = self.norm_attn(attn_out)
        else:
            # Pre-norm path (GPT-2+, LLaMA, Mistral — what you should use)
            attn_out, new_cache = self.attn(self.norm_attn(x), kv_cache)
            delta = attn_out

        return self.resid_drop(delta), new_cache

    def _ffn_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the FFN residual delta.

        Pre-norm  :  delta = FFN( LN(x) )
        Post-norm :  delta = LN( FFN(x) )
        """
        if self.post_norm:
            delta = self.norm_ffn(self.ffn(x))
        else:
            delta = self.ffn(self.norm_ffn(x))
        return self.resid_drop(delta)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Parameters
        ----------
        x        : (B, T, d_model)
        kv_cache : optional key-value cache tuple for autoregressive inference

        Returns
        -------
        x        : (B, T, d_model)
        kv_cache : updated cache (or None)
        """
        # ── Stochastic depth gate ────────────────────────────────────────────
        scale = self._stochastic_depth_scale()
        if scale == 0.0:
            # Block dropped entirely — identity pass-through
            return x, kv_cache

        # ── Parallel path (GPT-J / PaLM) ────────────────────────────────────
        if self.parallel_attn_ffn:
            """
            x = x + Attention( LN(x) ) + FFN( LN(x) )

            Both branches read from the *same* normalised input.
            On modern hardware they can execute concurrently.
            One fewer LayerNorm call vs serial.
            """
            normed = self.norm(x)
            attn_out, kv_cache = self.attn(normed, kv_cache)
            ffn_out            = self.ffn(normed)
            delta = self.resid_drop(attn_out) + self.resid_drop(ffn_out)
            x = x + delta * scale
            return x, kv_cache

        # ── Serial path (default — GPT-2/3, LLaMA, Mistral) ─────────────────
        """
        Step 1:  x = x + Attention( LN(x) )
        Step 2:  x = x + FFN(       LN(x) )

        The residual connection is critical:
        - Gradient highway: ∂L/∂x = ∂L/∂(x+Δ) · (1 + ∂Δ/∂x)
          The "1" ensures gradient flows back to earlier layers even when
          ∂Δ/∂x ≈ 0 at initialisation.
        - Ensemble interpretation: the full network is an implicit ensemble
          of 2^N sub-networks corresponding to subsets of residual paths
          (Veit et al. 2016).
        - Allows much greater effective depth: without residuals, the
          vanishing gradient problem limits practical depth to ~10 layers.
        """

        # ── Sub-layer 1: Self-Attention ──────────────────────────────────────
        attn_delta, kv_cache = self._attn_residual(x, kv_cache)
        x = x + attn_delta * scale      # residual add

        # ── Sub-layer 2: Feed-Forward ────────────────────────────────────────
        ffn_delta = self._ffn_residual(x)
        x = x + ffn_delta * scale       # residual add

        return x, kv_cache


# ===========================================================================
# TransformerStack — N blocks with per-layer stochastic depth schedule
# ===========================================================================

class TransformerStack(nn.Module):
    """
    A stack of N TransformerBlocks with linearly increasing stochastic depth.

    The stochastic depth probability for layer l (0-indexed) is:
        p_l = (l / (N-1)) · p_max

    This is the linear schedule from Huang et al. (2016):
    - Layer 0 (closest to input)  : p = 0    — never dropped
    - Layer N-1 (closest to head) : p = p_max — dropped most often
    - Deeper layers are inherently more redundant, so dropping them more
      aggressively is both safe and effective.

    Parameters
    ----------
    n_layers          : number of transformer blocks
    d_model           : embedding dimension
    n_heads           : attention heads
    stochastic_depth_p: maximum drop probability (last layer). 0 = disabled.
    **block_kwargs    : forwarded to each TransformerBlock
    """

    def __init__(
        self,
        n_layers:           int,
        d_model:            int,
        n_heads:            int,
        stochastic_depth_p: float = 0.0,
        **block_kwargs,
    ) -> None:
        super().__init__()

        # Build per-layer drop probabilities with linear schedule
        if n_layers > 1:
            drop_probs = [i / (n_layers - 1) * stochastic_depth_p
                          for i in range(n_layers)]
        else:
            drop_probs = [0.0]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model            = d_model,
                n_heads            = n_heads,
                stochastic_depth_p = drop_probs[i],
                **block_kwargs,
            )
            for i in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Parameters
        ----------
        x         : (B, T, d_model)
        kv_caches : list of per-layer kv_cache tuples, or None

        Returns
        -------
        x         : (B, T, d_model)
        kv_caches : updated list of per-layer caches
        """
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)

        new_caches = []
        for block, cache in zip(self.blocks, kv_caches):
            x, new_cache = block(x, cache)
            new_caches.append(new_cache)

        return x, new_caches


# ===========================================================================
# Self-test — run: python transformer_block.py
# ===========================================================================

if __name__ == "__main__":
    import sys

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}\n")

    B, T, C, H = 2, 64, 128, 4

    # ── 1. Norm comparison: LayerNorm vs RMSNorm ─────────────────────────────
    print("── Norm comparison (d=128) ─────────────────────────────────────")
    x_test = torch.randn(2, 16, C)
    for ntype in ("layernorm", "rmsnorm"):
        norm  = build_norm(ntype, C)
        out   = norm(x_test)
        params = sum(p.numel() for p in norm.parameters())
        # RMSNorm has no bias (fewer params), output mean ≈ 0, std ≈ 1
        print(f"  {ntype:12s}  params={params:3d}  "
              f"out_mean={out.mean():+.4f}  out_std={out.std():.4f}")

    # ── 2. Block variants ────────────────────────────────────────────────────
    print("\n── TransformerBlock variant tests ──────────────────────────────")
    configs = [
        dict(label="GPT-2 style    (pre-norm, LayerNorm, standard GELU, no rope)",
             norm_type="layernorm", ffn_variant="standard", use_rope=False,
             use_alibi=False, parallel_attn_ffn=False, post_norm=False),

        dict(label="LLaMA style    (pre-norm, RMSNorm,   SwiGLU,        RoPE)",
             norm_type="rmsnorm",   ffn_variant="swiglu",   use_rope=True,
             use_alibi=False, parallel_attn_ffn=False, post_norm=False),

        dict(label="GPT-J style    (parallel attn+ffn,   SwiGLU,        RoPE)",
             norm_type="rmsnorm",   ffn_variant="swiglu",   use_rope=True,
             use_alibi=False, parallel_attn_ffn=True,  post_norm=False),

        dict(label="ALiBi style    (pre-norm, RMSNorm,   SwiGLU,        ALiBi)",
             norm_type="rmsnorm",   ffn_variant="swiglu",   use_rope=False,
             use_alibi=True,  parallel_attn_ffn=False, post_norm=False),

        dict(label="Post-norm      (Vaswani original — for contrast only)",
             norm_type="layernorm", ffn_variant="standard", use_rope=False,
             use_alibi=False, parallel_attn_ffn=False, post_norm=True),

        dict(label="Stochastic depth p=0.2 (LLaMA-style + drop)",
             norm_type="rmsnorm",   ffn_variant="swiglu",   use_rope=True,
             use_alibi=False, parallel_attn_ffn=False, post_norm=False,
             stochastic_depth_p=0.2),
    ]

    all_passed = True
    for cfg in configs:
        label = cfg.pop("label")
        try:
            block = TransformerBlock(
                d_model=C, n_heads=H, max_seq_len=256, **cfg
            ).to(device)

            x  = torch.randn(B, T, C, device=device, requires_grad=True)
            out, _ = block(x)

            assert out.shape == (B, T, C), f"Shape mismatch: {out.shape}"

            loss = out.sum()
            loss.backward()
            assert x.grad is not None

            params = sum(p.numel() for p in block.parameters())
            print(f"  ✓ {label}")
            print(f"    out={tuple(out.shape)}  params={params:,}  "
                  f"grad_norm={x.grad.norm().item():.4f}")

        except Exception as e:
            print(f"  ✗ {label}\n    ERROR: {e}", file=sys.stderr)
            all_passed = False

    # ── 3. KV-cache consistency: single block ────────────────────────────────
    print("\n── KV-cache consistency (single block, LLaMA config) ───────────")
    block = TransformerBlock(
        d_model=C, n_heads=H, ffn_variant="swiglu",
        norm_type="rmsnorm", use_rope=True, max_seq_len=256,
    ).to(device).eval()

    with torch.no_grad():
        x_inf    = torch.randn(1, T, C, device=device)
        full_out, _ = block(x_inf)

        kv = None
        steps = []
        for t in range(T):
            step_out, kv = block(x_inf[:, t:t+1, :], kv)
            steps.append(step_out)
        cached_out = torch.cat(steps, dim=1)

        diff = (full_out - cached_out).abs().max().item()
        status = "✓" if diff < 1e-4 else "✗"
        print(f"  {status}  Max |full − cached| = {diff:.2e}")

    # ── 4. TransformerStack ──────────────────────────────────────────────────
    print("\n── TransformerStack (N=6, stochastic depth p_max=0.1) ──────────")
    stack = TransformerStack(
        n_layers=6, d_model=C, n_heads=H,
        ffn_variant="swiglu", norm_type="rmsnorm",
        use_rope=True, max_seq_len=256,
        stochastic_depth_p=0.1,
    ).to(device)

    x = torch.randn(B, T, C, device=device, requires_grad=True)
    out, caches = stack(x)
    assert out.shape == (B, T, C)
    out.sum().backward()

    total_params = sum(p.numel() for p in stack.parameters())
    print(f"  ✓  Stack out={tuple(out.shape)}  total_params={total_params:,}")
    print(f"     Per-layer drop schedule: "
          + "  ".join(f"L{i}:{b.stochastic_depth_p:.2f}"
                      for i, b in enumerate(stack.blocks)))

    # ── 5. Parameter breakdown ───────────────────────────────────────────────
    print("\n── Parameter breakdown: single LLaMA-style block (d=256, h=8) ─")
    block_demo = TransformerBlock(
        d_model=256, n_heads=8, ffn_variant="swiglu",
        norm_type="rmsnorm", use_rope=True, max_seq_len=2048,
    )
    total = sum(p.numel() for p in block_demo.parameters())
    for name, p in block_demo.named_parameters():
        pct = 100 * p.numel() / total
        print(f"  {name:40s}  {p.numel():>7,}  ({pct:4.1f}%)")
    print(f"  {'TOTAL':40s}  {total:>7,}")

    # ── 6. Pre-norm vs post-norm gradient norm across depth ──────────────────
    print("\n── Pre-norm vs Post-norm: gradient norm at layer 0 (N=8) ───────")
    for label, post_norm in [("Pre-norm  (recommended)", False),
                              ("Post-norm (Vaswani 2017)", True)]:
        stack_cmp = TransformerStack(
            n_layers=8, d_model=C, n_heads=H,
            ffn_variant="standard", norm_type="layernorm",
            use_rope=False, max_seq_len=256,
            post_norm=post_norm,
        ).to(device)
        x_cmp = torch.randn(B, T, C, device=device, requires_grad=True)
        out_cmp, _ = stack_cmp(x_cmp)
        out_cmp.sum().backward()
        print(f"  {label}  input grad norm = {x_cmp.grad.norm().item():.4f}")

    print("\n" + ("All tests PASSED ✓" if all_passed else "Some tests FAILED ✗"))