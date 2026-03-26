"""
feedforward.py — Position-wise Feed-Forward Network (FFN)
==========================================================
Author  : NanoGPT Lab
Standard: Vaswani et al. (2017) §3.3  +  GPT-2/3, PaLM, LLaMA, Mistral extensions

Mathematical foundation
-----------------------
The FFN sub-layer applies two learned affine transforms with a non-linearity
in between. For input x ∈ R^{B × T × d_model}:

    FFN(x) = W₂  ·  act( W₁ x + b₁ )  + b₂

    where  W₁ ∈ R^{d_ff × d_model},  W₂ ∈ R^{d_model × d_ff}
    and    d_ff = 4 · d_model  (canonical expansion ratio from the original paper)

The 4× expansion gives the network a high-dimensional "thinking space" per
token before projecting back. Each token is processed independently (the
"position-wise" qualifier) — there is zero cross-token mixing here; that
responsibility belongs entirely to attention.

Why GELU and not ReLU?
-----------------------
ReLU(x) = max(0, x)  — a hard gate: dead below 0, linear above.
    Problem 1: "Dying ReLU" — neurons whose pre-activation is always negative
               receive zero gradient and never update.
    Problem 2: The kink at 0 is non-smooth, which can destabilise deep nets.

GELU(x) = x · Φ(x)  where Φ is the standard normal CDF
    = x · ½ [1 + erf(x / √2)]

This is a *soft* gate: the input is multiplied by the probability that it is
positive under a unit Gaussian.  Near 0 the gate is smooth and differentiable;
for large negative x it approaches 0; for large positive x it approaches x.
GELU consistently outperforms ReLU on language model benchmarks (Hendrycks &
Gimpel 2016) and is the default in GPT-2, BERT, RoBERTa, and most modern LLMs.

We provide three implementations of GELU:
  - Exact    : x · ½ [1 + erf(x/√2)]         — numerically gold standard
  - FastApprox: x · σ(1.702 x)                — sigmoid approximation, ~2× faster
  - TanhApprox: Hendrycks original tanh formula — GPT-2 originally used this

For training you should almost always use the exact form (PyTorch's built-in
F.gelu with no extra cost); the approximations are shown for educational depth.

Variants implemented beyond the assignment spec
-----------------------------------------------
1. SwiGLU  — used in PaLM, LLaMA 1/2/3, Mistral.
             FFN(x) = (W₁x ⊙ σ(W_gate x)) W₂
             The gating mechanism gives the network multiplicative interactions.
             State of the art for decoder-only LLMs as of 2024.

2. GeGLU   — GELU variant of gated linear units (Noam Shazeer 2020).
             FFN(x) = GELU(W₁x) ⊙ (W_gate x) · W₂
             Strong results on T5 and UL2.

3. Mixture of Experts (MoE) stub — shows the structural extension where
             multiple FFN experts compete via a learned router, with only the
             top-k active per token. This is how GPT-4, Mixtral, and
             DeepSeek-V2 scale efficiently.

4. Configurable expansion ratio — modern research uses ratios other than 4×:
             PaLM uses 8/3 × d_model for SwiGLU (to keep param count equal),
             GPT-NeoX uses 4× with a separate bias-free design.
"""

import math
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Activation functions — implemented from first principles
# ===========================================================================

class GELUExact(nn.Module):
    """
    GELU(x) = x · Φ(x)  where Φ(x) = ½ [1 + erf(x / √2)]

    This is the exact form.  torch.special.erf is a vectorised CUDA kernel.
    Gradient: GELU'(x) = Φ(x) + x · φ(x)
    where φ(x) = (1/√2π) exp(−x²/2) is the standard normal PDF.
    The gradient is always non-zero (no dying neuron problem).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELUFastApprox(nn.Module):
    """
    GELU(x) ≈ x · sigmoid(1.702 · x)

    Derived by fitting a sigmoid to the CDF Φ(x).
    Error vs exact: < 0.001 for |x| < 4.
    Used in some inference kernels where erf is expensive.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class GELUTanhApprox(nn.Module):
    """
    GELU(x) ≈ 0.5 · x · [1 + tanh(√(2/π) · (x + 0.044715 · x³))]

    Original approximation from Hendrycks & Gimpel (2016).
    Used in GPT-2's original release (OpenAI codebase).
    Slightly less accurate than exact but vectorises well on older hardware.
    """
    _SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)   # ≈ 0.7978845608

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inner = self._SQRT_2_OVER_PI * (x + 0.044715 * x.pow(3))
        return 0.5 * x * (1.0 + torch.tanh(inner))


class SiLU(nn.Module):
    """
    SiLU(x) = x · σ(x)   (Sigmoid Linear Unit, also called Swish-1)

    Used as the activation inside SwiGLU.
    Smooth, non-monotonic, no dying units.
    torch.nn.SiLU is available, but we implement it explicitly for clarity.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


# Map string names → activation modules (used by FeedForward factory)
ACTIVATIONS: dict[str, type] = {
    "gelu":        GELUExact,
    "gelu_fast":   GELUFastApprox,
    "gelu_tanh":   GELUTanhApprox,
    "relu":        nn.ReLU,
    "silu":        SiLU,
}


# ===========================================================================
# Standard FeedForward (assignment spec + extensions)
# ===========================================================================

class FeedForward(nn.Module):
    """
    Position-wise FFN as used in GPT-2 / GPT-3.

        FFN(x) = dropout( W₂ · act( W₁ x ) )

    Parameters
    ----------
    d_model        : input / output dimension
    expansion      : hidden dim multiplier  (default 4, as in the original paper)
    activation     : which activation to use.  'gelu' is default and best.
    dropout        : dropout probability applied after W₂ (before residual add)
    bias           : whether linear layers include bias terms
                     (False follows PaLM / LLaMA design — often just as good,
                      fewer parameters, and cleaner gradient flow)
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        d_ff = d_model * expansion   # e.g. 128 × 4 = 512

        # ── Layers ──────────────────────────────────────────────────────────
        # Expand: d_model → d_ff
        self.W1  = nn.Linear(d_model, d_ff, bias=bias)
        # Activate
        self.act = ACTIVATIONS[activation]()
        # Contract: d_ff → d_model
        self.W2  = nn.Linear(d_ff, d_model, bias=bias)
        # Regularise
        self.drop = nn.Dropout(dropout)

        # ── Weight init ──────────────────────────────────────────────────────
        # Same convention as attention.py:
        #   W1 gets standard 0.02 normal (input projection)
        #   W2 gets scaled  0.02/√2 normal (residual-path projection)
        nn.init.normal_(self.W1.weight, std=0.02)
        nn.init.normal_(self.W2.weight, std=0.02 / math.sqrt(2))
        if bias:
            nn.init.zeros_(self.W1.bias)
            nn.init.zeros_(self.W2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, d_model)
        Returns (B, T, d_model)

        Each token position is processed independently by the same W1, W2.
        The batch and sequence dimensions are treated as a flat batch of
        d_model-dimensional vectors — this is what "position-wise" means.
        """
        x = self.W1(x)    # (B, T, d_ff)     — expand
        x = self.act(x)   # (B, T, d_ff)     — non-linearity
        x = self.W2(x)    # (B, T, d_model)  — contract
        x = self.drop(x)  # regularise
        return x


# ===========================================================================
# SwiGLU FeedForward  — LLaMA / Mistral / PaLM architecture
# ===========================================================================

class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU (Shazeer 2020):

        FFN_SwiGLU(x) = ( SiLU(W₁ x)  ⊙  (W_gate x) )  W₂

    Two separate up-projections (W₁ and W_gate) whose outputs are multiplied
    element-wise. The SiLU branch acts as a *learned gate* that controls how
    much of the linear W_gate signal passes through.

    Why does this work better?
    --------------------------
    The multiplicative interaction (⊙) allows the network to learn
    feature-dependent non-linear interactions that a plain activation
    cannot express. Empirically, SwiGLU models consistently reach lower
    perplexity than GELU models at matched parameter counts (Chowdhery et al.
    PaLM 2022; Touvron et al. LLaMA 2023).

    Parameter parity note
    ---------------------
    SwiGLU needs 3 weight matrices (W₁, W_gate, W₂) vs standard FFN's 2.
    To keep total parameters equal to a standard 4× FFN, the hidden dim is
    reduced to (8/3) × d_model ≈ 2.67 × d_model. LLaMA rounds this to the
    nearest multiple of 256.

    Parameters
    ----------
    d_model       : input / output dimension
    hidden_dim    : inner dimension. If None, defaults to round_to_256(8/3 × d_model)
    dropout       : applied after output projection
    bias          : include bias terms
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            # Match parameter count of standard 4× FFN while using 3 matrices
            # 8/3 × d_model, rounded to nearest multiple of 256
            raw = int(d_model * 8 / 3)
            hidden_dim = ((raw + 255) // 256) * 256

        self.W1    = nn.Linear(d_model, hidden_dim, bias=bias)   # gate-signal branch
        self.Wgate = nn.Linear(d_model, hidden_dim, bias=bias)   # SiLU branch
        self.W2    = nn.Linear(hidden_dim, d_model, bias=bias)   # output projection
        self.drop  = nn.Dropout(dropout)
        self.silu  = SiLU()

        nn.init.normal_(self.W1.weight,    std=0.02)
        nn.init.normal_(self.Wgate.weight, std=0.02)
        nn.init.normal_(self.W2.weight,    std=0.02 / math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU(W₁ x) ⊙ (W_gate x) — the ⊙ is element-wise multiplication.
        Think of SiLU(W₁ x) as a learned, input-dependent gate.
        """
        gate   = self.silu(self.W1(x))    # (B, T, hidden_dim)  soft gate ∈ (0, ∞)
        linear = self.Wgate(x)            # (B, T, hidden_dim)  linear features
        x = gate * linear                 # (B, T, hidden_dim)  gated features
        x = self.W2(x)                    # (B, T, d_model)
        x = self.drop(x)
        return x


# ===========================================================================
# GeGLU FeedForward  — T5 / UL2 architecture
# ===========================================================================

class GeGLUFeedForward(nn.Module):
    """
    GeGLU (Noam Shazeer 2020):

        FFN_GeGLU(x) = ( GELU(W₁ x)  ⊙  (W_gate x) )  W₂

    Same structure as SwiGLU but uses GELU instead of SiLU as the gate
    activation. Strong alternative when GELU is already well-tuned.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            raw = int(d_model * 8 / 3)
            hidden_dim = ((raw + 255) // 256) * 256

        self.W1    = nn.Linear(d_model, hidden_dim, bias=bias)
        self.Wgate = nn.Linear(d_model, hidden_dim, bias=bias)
        self.W2    = nn.Linear(hidden_dim, d_model, bias=bias)
        self.drop  = nn.Dropout(dropout)
        self.gelu  = GELUExact()

        nn.init.normal_(self.W1.weight,    std=0.02)
        nn.init.normal_(self.Wgate.weight, std=0.02)
        nn.init.normal_(self.W2.weight,    std=0.02 / math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate   = self.gelu(self.W1(x))
        linear = self.Wgate(x)
        x = gate * linear
        x = self.W2(x)
        x = self.drop(x)
        return x


# ===========================================================================
# Mixture of Experts (MoE) — structural stub (GPT-4 / Mixtral / DeepSeek)
# ===========================================================================

class MoEFeedForward(nn.Module):
    """
    Sparse Mixture of Experts FFN (Shazeer et al. 2017, Fedus et al. Switch 2021).

    Instead of one FFN applied to all tokens, we have E expert FFNs and a
    lightweight router that assigns each token to the top-k experts.

        router_logits = x W_router                      (B·T, E)
        gates, indices = topk(softmax(router_logits), k) (B·T, k)
        out = Σ_{i in top-k} gates_i · Expert_i(x)

    Only k/E fraction of expert parameters are active per token, enabling
    huge total capacity with constant compute per token.

    This stub uses k=2, E=8 (same as Mixtral 8×7B).
    Auxiliary load-balancing loss (not shown here) is needed in real training
    to prevent all tokens routing to the same expert.

    Parameters
    ----------
    d_model       : embedding dimension
    n_experts     : total number of expert FFNs  (E)
    n_active      : experts activated per token  (k)
    expansion     : each expert's inner-dim multiplier
    dropout       : applied inside each expert
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        n_active: int = 2,
        expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        assert n_active <= n_experts
        self.n_experts = n_experts
        self.n_active  = n_active

        # Router: maps token embedding → expert logits
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Expert pool — each is a standard SwiGLU FFN
        self.experts = nn.ModuleList([
            SwiGLUFeedForward(d_model, dropout=dropout)
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat = x.view(B * T, C)               # (B·T, C)

        # ── Router ───────────────────────────────────────────────────────────
        logits = self.router(x_flat)             # (B·T, E)
        probs  = F.softmax(logits, dim=-1)       # (B·T, E)  — routing distribution
        gates, indices = probs.topk(self.n_active, dim=-1)  # (B·T, k) each
        # Re-normalise the top-k gates so they sum to 1
        gates = gates / gates.sum(dim=-1, keepdim=True)     # (B·T, k)

        # ── Sparse expert dispatch ────────────────────────────────────────────
        out = torch.zeros_like(x_flat)           # accumulator (B·T, C)
        for k_idx in range(self.n_active):
            expert_ids = indices[:, k_idx]       # (B·T,) — which expert for this slot
            for e_id in range(self.n_experts):
                # Tokens assigned to this expert in this slot
                token_mask = (expert_ids == e_id)
                if not token_mask.any():
                    continue
                tokens_in  = x_flat[token_mask]          # (n_e, C)
                tokens_out = self.experts[e_id](
                    tokens_in.unsqueeze(0)               # fake batch dim
                ).squeeze(0)                             # (n_e, C)
                out[token_mask] += gates[token_mask, k_idx].unsqueeze(-1) * tokens_out

        return out.view(B, T, C)


# ===========================================================================
# Factory function — clean API used by transformer_block.py
# ===========================================================================

def build_ffn(
    variant: str,
    d_model: int,
    expansion: int = 4,
    activation: str = "gelu",
    dropout: float = 0.0,
    bias: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Construct an FFN module by name. Centralises all variant selection.

    variant choices
    ---------------
    'standard'   : FeedForward with configurable activation (default, GPT-2 style)
    'swiglu'     : SwiGLU — LLaMA / Mistral style  (recommended for new projects)
    'geglu'      : GeGLU  — T5 style
    'moe'        : Mixture of Experts stub

    Example (in transformer_block.py)
    ----------------------------------
    self.ffn = build_ffn('swiglu', d_model=512, dropout=0.1)
    """
    variant = variant.lower()
    if variant == "standard":
        return FeedForward(d_model, expansion=expansion,
                           activation=activation, dropout=dropout, bias=bias)
    elif variant == "swiglu":
        return SwiGLUFeedForward(d_model, dropout=dropout, bias=bias, **kwargs)
    elif variant == "geglu":
        return GeGLUFeedForward(d_model, dropout=dropout, bias=bias, **kwargs)
    elif variant == "moe":
        return MoEFeedForward(d_model, dropout=dropout, **kwargs)
    else:
        raise ValueError(f"Unknown FFN variant: '{variant}'. "
                         f"Choose from: standard, swiglu, geglu, moe")


# ===========================================================================
# Self-test — run: python feedforward.py
# ===========================================================================

if __name__ == "__main__":
    import sys

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}\n")

    B, T, C = 2, 64, 128

    # ── 1. Activation function numerical comparison ──────────────────────────
    print("── Activation function comparison ──────────────────────────────")
    x_act = torch.linspace(-3, 3, 7)
    exact      = GELUExact()(x_act)
    fast_approx = GELUFastApprox()(x_act)
    tanh_approx = GELUTanhApprox()(x_act)
    torch_gelu  = F.gelu(x_act)
    print(f"  {'x':>6}  {'Exact':>8}  {'FastSig':>8}  {'TanhApp':>8}  {'Torch':>8}")
    for i, xi in enumerate(x_act.tolist()):
        print(f"  {xi:6.2f}  {exact[i]:8.5f}  {fast_approx[i]:8.5f}  "
              f"{tanh_approx[i]:8.5f}  {torch_gelu[i]:8.5f}")
    max_err_fast = (exact - fast_approx).abs().max().item()
    max_err_tanh = (exact - tanh_approx).abs().max().item()
    print(f"\n  Max |fast_approx − exact|  = {max_err_fast:.6f}")
    print(f"  Max |tanh_approx − exact|  = {max_err_tanh:.6f}")

    # ── 2. Module forward / backward / shape checks ──────────────────────────
    print("\n── Module forward/backward/shape tests ─────────────────────────")
    configs = [
        dict(label="Standard  GELU  (4×)",  variant="standard", activation="gelu"),
        dict(label="Standard  ReLU  (4×)",  variant="standard", activation="relu"),
        dict(label="Standard  SiLU  (4×)",  variant="standard", activation="silu"),
        dict(label="SwiGLU    (8/3×)",      variant="swiglu"),
        dict(label="GeGLU     (8/3×)",      variant="geglu"),
        dict(label="MoE       (8 exp, k=2)", variant="moe"),
    ]

    all_passed = True
    for cfg in configs:
        label   = cfg.pop("label")
        variant = cfg.pop("variant")
        try:
            ffn = build_ffn(variant, d_model=C, **cfg).to(device)
            x   = torch.randn(B, T, C, device=device, requires_grad=True)
            out = ffn(x)

            assert out.shape == (B, T, C), f"Shape mismatch: {out.shape}"

            loss = out.sum()
            loss.backward()
            assert x.grad is not None

            params = sum(p.numel() for p in ffn.parameters())
            print(f"  ✓ {label:30s}  out={tuple(out.shape)}  params={params:,}")
        except Exception as e:
            print(f"  ✗ {label:30s}  ERROR: {e}", file=sys.stderr)
            all_passed = False

    # ── 3. Parameter count breakdown ─────────────────────────────────────────
    print("\n── Parameter breakdown (d_model=256) ───────────────────────────")
    for variant, label in [("standard","Standard (4×)"), ("swiglu","SwiGLU (8/3×)")]:
        ffn = build_ffn(variant, d_model=256)
        total = sum(p.numel() for p in ffn.parameters())
        print(f"\n  {label}  — total params: {total:,}")
        for name, p in ffn.named_parameters():
            print(f"    {name:25s}  {p.numel():>7,}  shape={list(p.shape)}")

    # ── 4. Expansion ratio ablation (matches Phase 4 spirit) ─────────────────
    print("\n── Expansion ratio effect on parameter count (d_model=128) ────")
    print(f"  {'Expansion':>12}  {'d_ff':>6}  {'Params':>8}")
    for exp in [1, 2, 4, 8]:
        ffn = FeedForward(d_model=128, expansion=exp)
        params = sum(p.numel() for p in ffn.parameters())
        print(f"  {exp:>12}×  {128*exp:>6}  {params:>8,}")

    print("\n" + ("All tests PASSED ✓" if all_passed else "Some tests FAILED ✗"))