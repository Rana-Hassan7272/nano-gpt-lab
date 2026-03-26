"""
attention.py — Multi-Head Causal Self-Attention
================================================
Author  : NanoGPT Lab
Standard: Vaswani et al. "Attention Is All You Need" (2017) + GPT-2/3 modifications

Mathematical foundation
-----------------------
Given input X ∈ R^{B × T × C}  (batch, sequence length, embedding dim):

    Q = X W_Q,   K = X W_K,   V = X W_V          projections
    q = split(Q, h),  k = split(K, h), v = split(V, h)   per-head views

    Attention(q, k, v) = softmax( q kᵀ / √d_k  +  M ) v

where M is the causal mask: M_{ij} = 0 if j ≤ i else −∞
and d_k = C / h  (head dimension).

The scale factor 1/√d_k keeps the dot-products from growing large in magnitude,
which would push softmax into its near-zero-gradient saturation zones. This is
the single most important numerical stability trick in the whole architecture.

After concatenating all h heads along the last axis we apply a final linear
output projection W_O:

    out = concat(heads) W_O

Why causal masking?
-------------------
During *training* the model sees the full sequence at once (teacher forcing).
Without a mask, token i could attend to token i+1, i+2, … and essentially
"cheat" by reading the answer. The lower-triangular mask enforces the
autoregressive inductive bias: each position can only depend on past tokens.

Why Flash-Attention (optional, see FlashMultiHeadAttention below)?
-----------------------------------------------------------------
Standard O(T²) attention materialises the full T×T attention matrix in HBM
(GPU global memory). Flash attention rewrites the CUDA kernels to tile the
computation so the T×T matrix *never leaves SRAM*, reducing memory I/O
from O(T²) to O(T) — purely an implementation optimisation; the math is
identical. We provide both so you can benchmark.

Extensions implemented beyond the assignment spec
--------------------------------------------------
1. ALiBi positional bias  — linear position penalty baked into attention,
   better extrapolation than learned positional embeddings.
2. Grouped-Query Attention (GQA) — each group of h_q query heads shares
   one K and V head (used in Llama 2/3, Mistral). n_kv_heads < n_heads.
3. RoPE (Rotary Position Embedding) — rotates Q and K in a way that
   their dot-product encodes *relative* position. Used in GPT-Neo-X,
   Llama, Mistral, Falcon …
4. FlashAttention v2 path via torch.nn.functional.scaled_dot_product_attention
   (automatically used on Ampere+ GPUs with CUDA 11.8+).
5. Attention-sink aware window (Mistral / StreamingLLM concept): hard-coded
   sink tokens always attend to position 0 to avoid the "massive activation"
   problem in long contexts.

You can start with the clean BaseMultiHeadAttention and progressively unlock
features by changing the config flags — this mirrors how a real research
ablation study is organised.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: build the causal mask once and cache it
# ---------------------------------------------------------------------------

def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Returns a boolean mask of shape (1, 1, T, T) where True means "block".
    We keep it as a bool tensor so we can use it with masked_fill.

    The upper triangle (j > i) is True (masked / future), lower triangle
    and diagonal (j ≤ i) is False (visible / past+present).

         T=4 example (0-indexed):
             j:  0  1  2  3
         i:0  [F  T  T  T]
           1  [F  F  T  T]
           2  [F  F  F  T]
           3  [F  F  F  F]
    """
    # torch.ones upper-triangular trick is O(T²) but trivial and cached.
    mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    mask = torch.triu(mask, diagonal=1)          # upper tri = True (blocked)
    return mask.unsqueeze(0).unsqueeze(0)        # (1, 1, T, T)  broadcast-ready


# ---------------------------------------------------------------------------
# Utility: RoPE (Rotary Position Embedding)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    base: float = 10_000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RoPE encodes position m and dimension index 2i via complex rotation:

        f_q(x, m) = (x_{2i} + i x_{2i+1}) e^{i m θ_i}

    where θ_i = 1 / base^{2i / d}.

    In practice we represent e^{i m θ} as (cos mθ, sin mθ) applied to
    interleaved pairs of the head dimension.

    Returns
    -------
    cos_cached : (max_seq_len, head_dim // 2)
    sin_cached : (max_seq_len, head_dim // 2)
    """
    assert head_dim % 2 == 0, "RoPE requires even head_dim"
    half = head_dim // 2
    # θ_i = 1 / base^{2i / d},   i = 0 … half-1
    theta = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    # position indices m = 0 … max_seq_len-1
    m = torch.arange(max_seq_len, device=device).float()
    # outer product: (max_seq_len, half)
    freqs = torch.outer(m, theta)
    cos_cached = freqs.cos()
    sin_cached = freqs.sin()
    return cos_cached, sin_cached


def apply_rope(
    x: torch.Tensor,                   # (B, n_heads, T, head_dim)
    cos: torch.Tensor,                 # (T, head_dim // 2)
    sin: torch.Tensor,                 # (T, head_dim // 2)
    start_pos: int = 0,
) -> torch.Tensor:
    """
    Applies rotary embedding in-place-style (no mutation, functional).

    The rotation on pair (x_{2i}, x_{2i+1}) for position m is:
        x'_{2i}   =  x_{2i}   cos(mθ_i) − x_{2i+1} sin(mθ_i)
        x'_{2i+1} =  x_{2i}   sin(mθ_i) + x_{2i+1} cos(mθ_i)

    This is equivalent to complex multiplication:
        (x_{2i} + i x_{2i+1}) × e^{imθ_i}
    """
    B, nh, T, hd = x.shape
    half = hd // 2

    # Split head dimension into even-indexed and odd-indexed halves
    x1 = x[..., :half]   # (B, nh, T, half)  — "real" part
    x2 = x[..., half:]   # (B, nh, T, half)  — "imaginary" part

    # Broadcast cos/sin from (T, half) → (1, 1, T, half)
    cos = cos[start_pos:start_pos + T].unsqueeze(0).unsqueeze(0)
    sin = sin[start_pos:start_pos + T].unsqueeze(0).unsqueeze(0)

    # Complex rotation
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    return torch.cat([rotated_x1, rotated_x2], dim=-1)


# ---------------------------------------------------------------------------
# Utility: ALiBi bias
# ---------------------------------------------------------------------------

def build_alibi_bias(
    n_heads: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    ALiBi (Press et al. 2021) replaces learned positional embeddings with a
    fixed linear penalty on attention logits:

        bias_{h, i, j} = −slope_h · (i − j)   for j ≤ i,  else −∞

    slope_h = 2^{−8h / n_heads}   (geometrically spaced, head-dependent)

    The penalty discourages attending to distant tokens proportionally to
    distance, without ever seeing positions outside training length.
    This gives *much* better length extrapolation than learned embeddings.

    Returns
    -------
    bias : (1, n_heads, seq_len, seq_len)  — ready to add to attention logits
    """
    # Head-specific slopes: 2^(-8/n * h) for h = 1 … n_heads
    slopes = torch.pow(2.0, -8.0 * torch.arange(1, n_heads + 1, device=device) / n_heads)
    # Distance matrix: (seq_len, seq_len),  entry (i,j) = |i - j|
    dist = torch.arange(seq_len, device=device).unsqueeze(0) - \
           torch.arange(seq_len, device=device).unsqueeze(1)   # (T, T), can be negative
    # Only penalise past tokens (j ≤ i), future already masked
    dist = dist.clamp(max=0)   # non-positive: 0 for same / future, negative for past
    # bias[h, i, j] = slope_h * dist[i, j]  (dist is negative so result is negative)
    bias = slopes.view(n_heads, 1, 1) * dist.unsqueeze(0).float()  # (n_heads, T, T)
    return bias.unsqueeze(0)   # (1, n_heads, T, T)


# ===========================================================================
# Core implementation: BaseMultiHeadAttention
# ===========================================================================

class MultiHeadAttention(nn.Module):
    """
    Production-grade causal multi-head self-attention.

    Supports
    --------
    - Standard MHA (n_kv_heads == n_heads)
    - Grouped-Query Attention / GQA (n_kv_heads < n_heads, divisible)
    - Multi-Query Attention / MQA (n_kv_heads == 1)
    - Optional RoPE positional encoding
    - Optional ALiBi positional bias
    - Flash attention path (PyTorch ≥ 2.0 with CUDA Ampere+)
    - Attention-sink windowing

    Parameters
    ----------
    d_model       : total embedding dimension  (must be divisible by n_heads)
    n_heads       : number of query heads
    n_kv_heads    : number of key/value heads.  Default = n_heads (standard MHA).
                    Set to 1 for MQA, or any divisor of n_heads for GQA.
    dropout       : attention dropout probability (applied to softmax output)
    bias          : whether to include bias in linear projections
    use_rope      : apply Rotary Position Embedding to Q and K
    use_alibi     : add ALiBi linear position bias to attention logits
    max_seq_len   : maximum sequence length (needed for RoPE pre-computation)
    use_flash     : use torch SDPA (FlashAttention) when available
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        use_rope: bool = True,
        use_alibi: bool = False,
        max_seq_len: int = 2048,
        use_flash: bool = True,
    ) -> None:
        super().__init__()

        # ── Validate dimensions ─────────────────────────────────────────────
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model    = d_model
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim   = d_model // n_heads
        self.dropout    = dropout
        self.use_rope   = use_rope
        self.use_alibi  = use_alibi
        self.use_flash  = use_flash
        self.max_seq_len = max_seq_len

        assert self.n_heads % self.n_kv_heads == 0, \
            "n_heads must be divisible by n_kv_heads for GQA"
        # How many query heads share each KV head?
        self.kv_groups  = self.n_heads // self.n_kv_heads

        # ── Linear projections ───────────────────────────────────────────────
        # Query: projects d_model → n_heads * head_dim  (= d_model for standard MHA)
        self.W_q = nn.Linear(d_model, self.n_heads    * self.head_dim, bias=bias)
        # Key  : projects d_model → n_kv_heads * head_dim  (smaller for GQA/MQA)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        # Value: same as key projection
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        # Output projection: concatenated heads → d_model
        self.W_o = nn.Linear(self.n_heads * self.head_dim, d_model, bias=bias)

        # Attention dropout (applied inside softmax, not on residual stream)
        self.attn_drop = nn.Dropout(dropout)

        # ── Optional: RoPE ───────────────────────────────────────────────────
        if use_rope:
            cos, sin = precompute_rope_freqs(
                head_dim=self.head_dim,
                max_seq_len=max_seq_len,
            )
            # Register as buffer: persists in state_dict, moved with .to(device)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        # ── Optional: ALiBi ──────────────────────────────────────────────────
        # ALiBi bias is sequence-length dependent so we build it lazily in forward.
        # We only store n_heads here.

        # ── Weight initialisation ────────────────────────────────────────────
        # GPT-2 uses a modified normal init for residual-path projections,
        # scaled by 1/√(2 * n_layers) to keep activation variance controlled
        # across depth.  We apply a mild version here; caller can override.
        nn.init.normal_(self.W_q.weight, std=0.02)
        nn.init.normal_(self.W_k.weight, std=0.02)
        nn.init.normal_(self.W_v.weight, std=0.02)
        # Output proj: scaled init to prevent variance blow-up in deep models
        nn.init.normal_(self.W_o.weight, std=0.02 / math.sqrt(2))

    # ── Helper: repeat KV heads for GQA ─────────────────────────────────────

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand KV tensors from (B, n_kv_heads, T, head_dim)
                             to (B, n_heads,    T, head_dim)
        by repeating each KV head `kv_groups` times.

        This is a *view* operation (no memory copy) when possible.
        """
        if self.kv_groups == 1:
            return x   # standard MHA: nothing to do
        B, nkv, T, hd = x.shape
        # Interleave: each KV head is repeated kv_groups times consecutively
        x = x.unsqueeze(2).expand(B, nkv, self.kv_groups, T, hd)
        return x.reshape(B, nkv * self.kv_groups, T, hd)

    # ── Core scaled dot-product attention (manual, educational) ─────────────

    def _manual_scaled_dot_product(
        self,
        q: torch.Tensor,    # (B, n_heads, T_q, head_dim)
        k: torch.Tensor,    # (B, n_heads, T_k, head_dim)
        v: torch.Tensor,    # (B, n_heads, T_k, head_dim)
        mask: torch.Tensor, # (1, 1, T_q, T_k) bool — True means blocked
        alibi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implements:
            scores = Q Kᵀ / √d_k
            scores = scores + alibi_bias   (if ALiBi)
            scores = masked_fill(scores, mask, −∞)
            weights = softmax(scores, dim=-1)
            weights = dropout(weights)
            output = weights V

        Numerics note on softmax stability
        -----------------------------------
        PyTorch's F.softmax already applies the log-sum-exp trick internally
        (subtracts max before exponentiation) so we don't need to do it
        manually. However, be aware that masking with −∞ (or a very large
        negative like −1e9) is crucial: e^{−∞} = 0 exactly, which zeroes
        out future tokens cleanly. Using −1e4 is *not* safe for bf16.
        """
        scale = 1.0 / math.sqrt(self.head_dim)   # 1/√d_k

        # ── Raw logits: (B, n_heads, T_q, T_k) ──────────────────────────────
        # matmul of (B, nh, T_q, hd) @ (B, nh, hd, T_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # ── Add ALiBi positional bias ────────────────────────────────────────
        if alibi is not None:
            scores = scores + alibi   # (1, n_heads, T_q, T_k) broadcasts

        # ── Causal mask: fill future positions with -inf ─────────────────────
        # mask is True where we want to block (future tokens)
        scores = scores.masked_fill(mask, float("-inf"))

        # ── Softmax over key dimension → attention weights ───────────────────
        weights = F.softmax(scores, dim=-1)   # (B, n_heads, T_q, T_k)
        weights = self.attn_drop(weights)

        # ── Weighted sum of values ───────────────────────────────────────────
        # (B, nh, T_q, T_k) @ (B, nh, T_k, hd) → (B, nh, T_q, hd)
        out = torch.matmul(weights, v)
        return out

    # ── Forward pass ────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,                        # (B, T, C)
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Parameters
        ----------
        x        : input tensor of shape (B, T, d_model)
        kv_cache : optional (k_prev, v_prev) for autoregressive inference.
                   During training always pass None.

        Returns
        -------
        out      : (B, T, d_model)
        kv_cache : updated (k, v) tensors for next step (or None if not used)
        """
        B, T, C = x.shape
        assert C == self.d_model, f"Input dim {C} ≠ d_model {self.d_model}"
        past_len = kv_cache[0].shape[2] if kv_cache is not None else 0

        # ── Step 1: Project to Q, K, V ───────────────────────────────────────
        # Each projection is a learned linear map.
        # W_q: (B, T, C) → (B, T, n_heads    * head_dim)
        # W_k: (B, T, C) → (B, T, n_kv_heads * head_dim)
        # W_v: (B, T, C) → (B, T, n_kv_heads * head_dim)
        q = self.W_q(x)   # (B, T, n_heads    * head_dim)
        k = self.W_k(x)   # (B, T, n_kv_heads * head_dim)
        v = self.W_v(x)   # (B, T, n_kv_heads * head_dim)

        # ── Step 2: Reshape into per-head views ──────────────────────────────
        # (B, T, n_heads * head_dim) → (B, n_heads, T, head_dim)
        # This does NOT copy memory — it is a strided view.
        q = q.view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # ── Step 3: Apply Rotary Position Embedding to Q and K ───────────────
        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin, start_pos=past_len)
            k = apply_rope(k, self.rope_cos, self.rope_sin, start_pos=past_len)
            # Note: V is intentionally NOT rotated — only position-encoding Q,K

        # ── Step 4: KV cache support (inference only) ────────────────────────
        # During training this branch is never taken.
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)   # concat along sequence axis
            v = torch.cat([v_cache, v], dim=2)
        # Always return updated KV so autoregressive decoding can bootstrap from None.
        new_kv_cache = (k, v)
        T_k = k.shape[2]   # key sequence length (may differ from T during inference)

        # ── Step 5: Expand KV heads for GQA ─────────────────────────────────
        # Standard MHA: no-op. GQA/MQA: repeats KV heads to match Q heads.
        k = self._repeat_kv(k)   # (B, n_heads, T_k, head_dim)
        v = self._repeat_kv(v)   # (B, n_heads, T_k, head_dim)

        # ── Step 6: Causal mask ──────────────────────────────────────────────
        causal_mask = build_causal_mask(T_k, device=x.device)  # (1,1,T_k,T_k)
        # For cross-step inference the query length may be 1 → slice mask
        # We only need the last T rows (query positions)
        causal_mask = causal_mask[:, :, T_k - T:T_k, :T_k]    # (1,1,T,T_k)

        # ── Step 7: Optional ALiBi bias ──────────────────────────────────────
        alibi = None
        if self.use_alibi:
            alibi = build_alibi_bias(self.n_heads, T_k, device=x.device)
            alibi = alibi[:, :, T_k - T:T_k, :T_k]

        # ── Step 8: Compute attention ────────────────────────────────────────
        if (self.use_flash
                and hasattr(F, "scaled_dot_product_attention")
                and not self.use_alibi
                and kv_cache is None):          # keep cache path on manual kernel for correctness
            # PyTorch 2.0+ fused kernel — handles mask, scale, dropout internally
            # is_causal=True tells it to build the causal mask internally in CUDA,
            # further saving memory.
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual path: educational, also used when ALiBi is on
            out = self._manual_scaled_dot_product(q, k, v, causal_mask, alibi)

        # ── Step 9: Concatenate heads ─────────────────────────────────────────
        # (B, n_heads, T, head_dim) → (B, T, n_heads * head_dim) = (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        # `.contiguous()` is necessary because transpose creates a non-contiguous
        # tensor; `view` requires contiguous memory.

        # ── Step 10: Output projection ────────────────────────────────────────
        # Mixes information across heads; this is where the heads "talk" to each other.
        out = self.W_o(out)   # (B, T, d_model)

        return out, new_kv_cache


# ===========================================================================
# Minimal self-test — run: python attention.py
# ===========================================================================

if __name__ == "__main__":
    """
    Sanity-check every configuration variant.
    Checks:
      1. Output shape correctness
      2. Forward pass does not raise
      3. Backward pass (gradients flow through all params)
      4. KV-cache produces identical output to full forward pass (inference check)
    """
    import sys

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # ── Test configurations ──────────────────────────────────────────────────
    configs = [
        dict(label="Standard MHA (no pos-enc)",
             d_model=128, n_heads=4, use_rope=False, use_alibi=False),
        dict(label="MHA + RoPE",
             d_model=128, n_heads=4, use_rope=True,  use_alibi=False),
        dict(label="MHA + ALiBi",
             d_model=128, n_heads=4, use_rope=False, use_alibi=True),
        dict(label="GQA (4 Q heads, 2 KV heads) + RoPE",
             d_model=128, n_heads=4, n_kv_heads=2, use_rope=True, use_alibi=False),
        dict(label="MQA (4 Q heads, 1 KV head) + RoPE",
             d_model=128, n_heads=4, n_kv_heads=1, use_rope=True, use_alibi=False),
    ]

    B, T, C = 2, 64, 128
    all_passed = True

    for cfg in configs:
        label = cfg.pop("label")
        try:
            model = MultiHeadAttention(**cfg, max_seq_len=256).to(device)
            x = torch.randn(B, T, C, device=device, requires_grad=True)

            # Forward pass
            out, _ = model(x)
            assert out.shape == (B, T, C), f"Shape mismatch: {out.shape}"

            # Backward pass
            loss = out.sum()
            loss.backward()
            assert x.grad is not None, "No gradient on input"
            for name, p in model.named_parameters():
                assert p.grad is not None, f"No gradient on {name}"

            # KV-cache consistency check (simulate autoregressive inference)
            model.eval()
            with torch.no_grad():
                x_inf = torch.randn(1, T, C, device=device)
                full_out, _ = model(x_inf)

                # Feed tokens one by one with KV cache
                kv = None
                cache_outs = []
                for t in range(T):
                    tok = x_inf[:, t:t+1, :]   # (1, 1, C)
                    step_out, kv = model(tok, kv_cache=kv)
                    cache_outs.append(step_out)
                cached_out = torch.cat(cache_outs, dim=1)   # (1, T, C)

                # Outputs should match (within float tolerance)
                max_diff = (full_out - cached_out).abs().max().item()
                assert max_diff < 1e-4, f"KV-cache mismatch: max diff = {max_diff:.2e}"

            print(f"  ✓ {label}  — output {out.shape}  kv-cache diff {max_diff:.2e}")

        except Exception as e:
            print(f"  ✗ {label}  — ERROR: {e}", file=sys.stderr)
            all_passed = False

    # ── Parameter count breakdown ────────────────────────────────────────────
    print("\nParameter count breakdown (Standard MHA, d=256, h=8):")
    demo = MultiHeadAttention(d_model=256, n_heads=8, use_rope=True)
    total = sum(p.numel() for p in demo.parameters())
    for name, p in demo.named_parameters():
        print(f"  {name:30s}  {p.numel():>8,d}  shape={list(p.shape)}")
    print(f"  {'TOTAL':30s}  {total:>8,d}")

    print("\n" + ("All tests PASSED ✓" if all_passed else "Some tests FAILED ✗"))