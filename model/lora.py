"""
lora.py — Low-Rank Adaptation (LoRA) for NanoGPT
==================================================
Author  : NanoGPT Lab
Paper   : Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
          https://arxiv.org/abs/2106.09685

Mathematical formulation
------------------------
For a frozen pre-trained weight W₀ ∈ R^{d × k}, LoRA parameterises the
weight update as a rank-r factorisation:

    h = W₀ x  +  ΔW x · (α / r)
      = W₀ x  +  B A x · (α / r)

    A ∈ R^{r × k}  — down-projection  (random normal init)
    B ∈ R^{d × r}  — up-projection    (zero init → ΔW = 0 at step 0)

Only A and B are trainable.  W₀ is frozen (requires_grad = False).

After fine-tuning, adapters can be merged into the base weights with zero
inference overhead:

    W_merged = W₀ + (α / r) · B · A

Modules implemented
-------------------
LoRALinear          — drop-in replacement for nn.Linear with LoRA adapters
LoRAConfig          — all LoRA hyperparameters in one dataclass
apply_lora          — inject LoRA into every target projection of a NanoGPT
merge_lora          — bake adapter weights into base weights for deployment
unmerge_lora        — reverse a merge (useful for switching between adapters)
save_lora           — save only the adapter weights (tiny checkpoint)
load_lora           — load adapter weights onto a base model
print_trainable_params — audit which parameters are trainable

Design decisions
----------------
1. LoRALinear wraps an existing nn.Linear rather than replacing it.
   This means we can inject LoRA into an already-instantiated NanoGPT
   loaded from a Phase 4 checkpoint without rebuilding the model.

2. The base linear weight is frozen at construction time.  Calling
   apply_lora() on a model also calls freeze_base_model() first so
   you cannot accidentally leave base weights trainable.

3. Rank ablation is easy: pass rank=1,2,4,8,16 to LoRAConfig and
   re-run.  The α/r scaling means you never need to retune lr.

4. All four attention projections (W_q, W_k, W_v, W_o) are targeted
   by default.  You can restrict to W_q + W_v (original paper ablation)
   by setting target_modules={"W_q", "W_v"} in LoRAConfig.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Optional, Set, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class LoRAConfig:
    """
    All LoRA hyperparameters in one place.

    Parameters
    ----------
    rank           : r — rank of the low-rank update matrices A, B.
                     Higher rank = more expressiveness, more params.
                     r=4 is the default from the original paper and works
                     for most domain adaptation tasks.

    alpha          : α — scaling numerator.  The adapter contribution is
                     scaled by α/r.  Setting alpha=rank gives scale=1.0,
                     meaning the adapter is weighted equally to the frozen
                     base at full rank-r capacity.  alpha=2*rank gives
                     scale=2.0, amplifying the adapter.

    dropout        : Dropout on the LoRA path only.  Regularises the adapter
                     without touching the frozen base.  Use 0.05–0.1 for
                     small fine-tuning datasets.

    target_modules : Set of module name suffixes to inject LoRA into.
                     Default targets all four attention projections.
                     Use {"W_q", "W_v"} for the minimal ablation from
                     the original paper.

    bias           : 'none'   — no bias trained (default, matches base model)
                     'lora'   — train LoRA bias terms
                     'all'    — unfreeze all biases in the model
    """
    rank:           int         = 4
    alpha:          float       = 4.0        # scale = alpha/rank = 1.0 by default
    dropout:        float       = 0.0
    target_modules: Set[str]    = field(default_factory=lambda: {
                                      "W_q", "W_k", "W_v", "W_o"
                                  })
    bias:           str         = "none"     # 'none' | 'lora' | 'all'

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank must be > 0, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if self.bias not in {"none", "lora", "all"}:
            raise ValueError(f"bias must be one of {{'none','lora','all'}}, got {self.bias}")

    @property
    def scale(self) -> float:
        """α / r  — the LoRA scaling factor applied at forward time."""
        return self.alpha / self.rank

    def to_dict(self) -> dict:
        import dataclasses
        d = dataclasses.asdict(self)
        d["target_modules"] = list(self.target_modules)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LoRAConfig":
        d = d.copy()
        d["target_modules"] = set(d.get("target_modules", ["W_q","W_k","W_v","W_o"]))
        return cls(**d)


# ===========================================================================
# LoRALinear — the core module
# ===========================================================================

class LoRALinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds a LoRA adapter path.

    Forward pass:
        out = x @ W₀ᵀ  +  (x @ Aᵀ @ Bᵀ) · (α / r)
              └────────┘    └─────────────────────────┘
              frozen base         trainable adapter

    The base weight W₀ is immediately frozen on construction
    (requires_grad = False).

    Parameters
    ----------
    base_layer : the original nn.Linear to wrap
    rank       : r, rank of A and B
    alpha      : α, scaling numerator
    dropout    : dropout probability on the LoRA path
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank:       int,
        alpha:      float,
        dropout:    float = 0.0,
    ) -> None:
        super().__init__()

        self.rank   = rank
        self.alpha  = alpha
        self.scale  = alpha / rank   # α/r applied at every forward call

        # ── Frozen base layer ────────────────────────────────────────────────
        self.base   = base_layer
        # Freeze immediately — no gradients will flow into W₀
        for p in self.base.parameters():
            p.requires_grad_(False)

        d_out, d_in = base_layer.weight.shape   # (out_features, in_features)

        # ── LoRA matrices ─────────────────────────────────────────────────────
        # A ∈ R^{r × d_in}   down-projection  (random Kaiming init)
        # B ∈ R^{d_out × r}  up-projection    (zero init → ΔW = 0 at start)
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))  # zero init

        # ── Dropout on the LoRA path only ─────────────────────────────────────
        self.lora_drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # ── State flag: has this adapter been merged into base? ───────────────
        self._merged = False

        # ── Initialise A with Kaiming uniform (same as nn.Linear default) ─────
        # This matches the distribution the base model expects as input scale.
        # B is already zero — the product BA = 0 at step 0.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def weight(self) -> torch.Tensor:
        """Expose base weight for compatibility (e.g., weight tying checks)."""
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    # ── Delta W computation ───────────────────────────────────────────────────

    def _lora_delta(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the LoRA adapter contribution: (x @ Aᵀ @ Bᵀ) · (α/r)

        Shape trace (batch dims collapsed to ...):
            x          : (..., d_in)
            x @ Aᵀ     : (..., r)         — project to low-rank subspace
            ... @ Bᵀ   : (..., d_out)     — project back to output dim
            · scale    : (..., d_out)     — weighted contribution
        """
        x_drop = self.lora_drop(x)                   # dropout on input to adapter
        down   = F.linear(x_drop, self.lora_A)       # (..., r)
        up     = F.linear(down,   self.lora_B)        # (..., d_out)
        return up * self.scale                        # (..., d_out)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._merged:
            # Adapter has been baked into base weights — single linear call
            return self.base(x)

        # Base path (frozen) + LoRA adapter path (trainable)
        base_out  = self.base(x)          # W₀ x  (+ bias if present)
        lora_out  = self._lora_delta(x)   # B A x · (α/r)
        return base_out + lora_out

    # ── Merge / Unmerge ───────────────────────────────────────────────────────

    def merge(self) -> None:
        """
        Bake the LoRA adapter into the base weight for zero-overhead inference.

            W_merged = W₀ + (α/r) · B · A

        After calling merge(), forward() executes a single matrix multiply.
        The adapter parameters are kept in memory (for potential unmerge)
        but contribute nothing to the computation.

        Call this before deployment / ONNX export.
        """
        if self._merged:
            return   # idempotent
        # Compute ΔW = (α/r) · B · A ∈ R^{d_out × d_in}
        delta_W = self.scale * (self.lora_B @ self.lora_A)
        # Add in-place to the frozen weight (safe: no grad needed here)
        self.base.weight.data += delta_W
        self._merged = True

    def unmerge(self) -> None:
        """
        Reverse a previous merge().  Useful when you want to:
          - Switch between multiple LoRA adapters on the same base model
          - Continue training after a temporary merge
        """
        if not self._merged:
            return
        delta_W = self.scale * (self.lora_B @ self.lora_A)
        self.base.weight.data -= delta_W
        self._merged = False

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}, scale={self.scale:.3f}, "
                f"merged={self._merged}")


# ===========================================================================
# Model-level utilities
# ===========================================================================

def freeze_base_model(model: nn.Module) -> None:
    """
    Freeze ALL parameters of a model.

    Call this before apply_lora() so the base model is definitely frozen.
    apply_lora() will then selectively unfreeze only the LoRA adapters.
    """
    for p in model.parameters():
        p.requires_grad_(False)


def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    Inject LoRA adapters into every nn.Linear whose name ends with a
    string in config.target_modules.

    This mutates the model in-place and returns it.

    Algorithm
    ---------
    Walk the module tree with named_modules().
    For each module whose name suffix matches a target, replace it with
    a LoRALinear wrapping the original module.
    Use _set_module() to do the replacement without breaking the module tree.

    Parameters
    ----------
    model  : the NanoGPT instance (should already be loaded from checkpoint)
    config : LoRAConfig with rank, alpha, dropout, target_modules

    Returns
    -------
    model : the same model object, mutated
    """
    # Step 1: freeze everything first
    freeze_base_model(model)

    # Step 2: find all target linear layers
    # We collect (parent_module, child_name, child_module) triples
    replacements: List[Tuple[nn.Module, str, str]] = []

    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Check if the leaf name matches any target suffix
        leaf_name = full_name.split(".")[-1]
        if leaf_name not in config.target_modules:
            continue
        # Navigate to parent
        parts  = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        replacements.append((parent, parts[-1], full_name))

    if not replacements:
        raise ValueError(
            f"No target modules found. Targets: {config.target_modules}\n"
            f"Available linears: "
            + str([n for n, m in model.named_modules() if isinstance(m, nn.Linear)])
        )

    # Step 3: replace each target with a LoRALinear
    n_replaced = 0
    for parent, attr_name, full_name in replacements:
        original = getattr(parent, attr_name)
        lora_layer = LoRALinear(
            base_layer = original,
            rank       = config.rank,
            alpha      = config.alpha,
            dropout    = config.dropout,
        )
        setattr(parent, attr_name, lora_layer)
        n_replaced += 1

    # Step 4: handle bias unfreezing per config
    if config.bias == "all":
        for name, p in model.named_parameters():
            if "bias" in name:
                p.requires_grad_(True)
    elif config.bias == "lora":
        for name, p in model.named_parameters():
            if "lora_" in name and "bias" in name:
                p.requires_grad_(True)
    # config.bias == "none": all biases remain frozen (default)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA applied  |  "
        f"{n_replaced} layers wrapped  |  "
        f"rank={config.rank}  alpha={config.alpha}  scale={config.scale:.3f}\n"
        f"              |  "
        f"trainable={n_trainable:,} / {n_total:,}  "
        f"({100*n_trainable/n_total:.3f}%)"
    )
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA adapters in the model into their base weights.
    Call this before deployment to eliminate inference overhead.
    Returns the same model, mutated.
    """
    n_merged = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            n_merged += 1
    print(f"Merged {n_merged} LoRA adapters into base weights.")
    return model


def unmerge_lora(model: nn.Module) -> nn.Module:
    """Reverse all merges in the model. Useful for adapter switching."""
    n_unmerged = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
            n_unmerged += 1
    print(f"Unmerged {n_unmerged} LoRA adapters.")
    return model


# ===========================================================================
# Checkpoint I/O — save/load ONLY the adapter weights
# ===========================================================================

def save_lora(model: nn.Module, config: LoRAConfig, path: str, step: int = 0,
              loss: Optional[float] = None) -> None:
    """
    Save only the LoRA adapter weights — not the base model.

    The resulting file is tiny (a few hundred KB for rank=4 on 6M model)
    compared to the full model checkpoint (tens of MB).  The base weights
    are never duplicated.

    File format
    -----------
    {
        "config"  : LoRAConfig.to_dict()        — for reconstruction
        "adapters": { param_name: tensor }      — only lora_A, lora_B, biases
        "step"    : int
        "loss"    : float | None
    }
    """
    adapter_state = {
        name: param.data
        for name, param in model.named_parameters()
        if param.requires_grad   # only trainable = LoRA params
    }
    payload = {
        "config":   config.to_dict(),
        "adapters": adapter_state,
        "step":     step,
        "loss":     loss,
    }
    torch.save(payload, path)
    size_kb = sum(t.numel() * t.element_size() for t in adapter_state.values()) / 1024
    print(f"LoRA adapters saved  →  {path}  "
          f"({len(adapter_state)} tensors, {size_kb:.1f} KB, step={step})")


def load_lora(model: nn.Module, path: str,
              device: str = "cpu") -> Tuple[nn.Module, LoRAConfig]:
    """
    Load LoRA adapter weights onto an already-constructed model.

    The model must already have LoRA layers injected (via apply_lora).
    This function only updates the lora_A and lora_B tensors.

    Returns
    -------
    model  : model with adapter weights loaded
    config : the LoRAConfig that was used when saving
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    config  = LoRAConfig.from_dict(payload["config"])

    # Build a mapping of param name → parameter for quick lookup
    param_map = dict(model.named_parameters())
    missing, unexpected = [], []

    for name, saved_tensor in payload["adapters"].items():
        if name in param_map:
            param_map[name].data.copy_(saved_tensor.to(device))
        else:
            unexpected.append(name)

    for name, p in param_map.items():
        if p.requires_grad and name not in payload["adapters"]:
            missing.append(name)

    step = payload.get("step", 0)
    loss = payload.get("loss", None)
    print(f"LoRA adapters loaded ←  {path}  (step={step}  loss={loss})")
    if missing:
        print(f"  WARNING: missing adapter tensors: {missing}")
    if unexpected:
        print(f"  WARNING: unexpected tensors (ignored): {unexpected}")

    return model, config


# ===========================================================================
# Diagnostics
# ===========================================================================

def print_trainable_params(model: nn.Module, verbose: bool = True) -> Dict[str, int]:
    """
    Print a clear audit of trainable vs frozen parameters.

    Returns a dict: {'trainable': N, 'frozen': M, 'total': N+M}
    """
    trainable_params = {}
    frozen_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param.numel()
        else:
            frozen_count += param.numel()

    n_trainable = sum(trainable_params.values())
    n_total     = n_trainable + frozen_count

    if verbose:
        print(f"\n{'Parameter':60s} {'Count':>10}  Trainable")
        print("─" * 76)
        for name, count in trainable_params.items():
            print(f"  {name:58s} {count:>10,}  ✓")
        print("─" * 76)
        print(f"  {'Trainable (LoRA)':58s} {n_trainable:>10,}  ✓")
        print(f"  {'Frozen (base)':58s} {frozen_count:>10,}  ✗")
        print(f"  {'Total':58s} {n_total:>10,}")
        print(f"\n  Adapter overhead: {100*n_trainable/n_total:.4f}% of total parameters")

    return {"trainable": n_trainable, "frozen": frozen_count, "total": n_total}


def lora_summary(model: nn.Module) -> None:
    """Print a table of all LoRA layers with their shapes and merge state."""
    print(f"\n{'Layer':55s} {'A shape':>14}  {'B shape':>14}  {'Merged':>7}")
    print("─" * 95)
    n_lora = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            A_shape = tuple(module.lora_A.shape)
            B_shape = tuple(module.lora_B.shape)
            print(f"  {name:53s} {str(A_shape):>14}  {str(B_shape):>14}  "
                  f"{'Yes' if module._merged else 'No':>7}")
            n_lora += 1
    print("─" * 95)
    print(f"  {n_lora} LoRA layers total")


# ===========================================================================
# Self-test — run: python lora.py
# ===========================================================================

if __name__ == "__main__":
    import sys
    import tempfile, os

    # We test LoRA in isolation (no NanoGPT import needed)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}\n")

    all_passed = True

    # ── 0. Build a minimal transformer-like model to apply LoRA on ───────────
    class TinyAttention(nn.Module):
        """Minimal stand-in for one attention layer."""
        def __init__(self, d: int = 64):
            super().__init__()
            self.W_q = nn.Linear(d, d, bias=False)
            self.W_k = nn.Linear(d, d, bias=False)
            self.W_v = nn.Linear(d, d, bias=False)
            self.W_o = nn.Linear(d, d, bias=False)
        def forward(self, x):
            return self.W_o(self.W_v(self.W_q(x) + self.W_k(x)))

    class TinyModel(nn.Module):
        def __init__(self, d: int = 64, n_layers: int = 3):
            super().__init__()
            self.layers = nn.ModuleList([TinyAttention(d) for _ in range(n_layers)])
            self.head   = nn.Linear(d, 10, bias=False)
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.head(x)

    D, B, T = 64, 2, 16
    model   = TinyModel(d=D, n_layers=3).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Base model: {total_p:,} parameters\n")

    # ── 1. LoRALinear unit test ───────────────────────────────────────────────
    print("── LoRALinear unit test ────────────────────────────────────────")
    try:
        base_lin = nn.Linear(D, D, bias=False).to(device)
        lora_lin = LoRALinear(base_lin, rank=4, alpha=4.0).to(device)

        x = torch.randn(B, T, D, device=device)

        # Forward before merge
        out_before = lora_lin(x)
        assert out_before.shape == (B, T, D)

        # B is zero-init → at step 0, lora output = base output
        base_out = base_lin(x)
        diff_init = (out_before - base_out).abs().max().item()
        print(f"  ✓  Zero-init check: |lora − base| = {diff_init:.2e}  (should be 0)")
        assert diff_init < 1e-6, "LoRA init is not zero!"

        # Verify only lora_A, lora_B are trainable
        trainable = [n for n, p in lora_lin.named_parameters() if p.requires_grad]
        frozen    = [n for n, p in lora_lin.named_parameters() if not p.requires_grad]
        assert set(trainable) == {"lora_A", "lora_B"}, f"Wrong trainable: {trainable}"
        print(f"  ✓  Trainable: {trainable}  |  Frozen: {frozen}")

        # Gradient flows through A and B only
        out_before.sum().backward()
        assert lora_lin.lora_A.grad is not None
        assert lora_lin.lora_B.grad is not None
        assert base_lin.weight.grad is None   # frozen: no gradient
        print(f"  ✓  Gradients: A.grad={lora_lin.lora_A.grad.norm():.4f}  "
              f"B.grad={lora_lin.lora_B.grad.norm():.4f}  "
              f"W₀.grad=None ✓")

    except Exception as e:
        print(f"  ✗  LoRALinear: {e}", file=sys.stderr)
        all_passed = False

    # ── 2. apply_lora on TinyModel ────────────────────────────────────────────
    print("\n── apply_lora on TinyModel ─────────────────────────────────────")
    try:
        model = TinyModel(d=D, n_layers=3).to(device)
        cfg   = LoRAConfig(rank=4, alpha=4.0, dropout=0.0,
                           target_modules={"W_q", "W_k", "W_v", "W_o"})
        apply_lora(model, cfg)

        stats = print_trainable_params(model, verbose=False)
        assert stats["trainable"] > 0, "No trainable params after apply_lora"
        assert stats["frozen"]    > 0, "No frozen params after apply_lora"
        expected_lora_params = 3 * 4 * (4*D + D*4)   # 3 layers × 4 projections × (A+B)
        print(f"  ✓  Trainable: {stats['trainable']:,}  "
              f"(expected ≈ {expected_lora_params:,})")
        print(f"  ✓  Frozen:    {stats['frozen']:,}")
        print(f"  ✓  Overhead:  {100*stats['trainable']/stats['total']:.3f}%")

    except Exception as e:
        print(f"  ✗  apply_lora: {e}", file=sys.stderr)
        all_passed = False

    # ── 3. Forward pass after LoRA injection ──────────────────────────────────
    print("\n── Forward pass after LoRA injection ───────────────────────────")
    try:
        x   = torch.randn(B, T, D, device=device)
        out = model(x)
        assert out.shape == (B, T, 10)
        loss = out.sum()
        loss.backward()
        # Check only LoRA params got gradients
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"LoRA param {name} has no gradient"
            else:
                assert p.grad is None, f"Frozen param {name} has gradient!"
        print(f"  ✓  Forward shape {tuple(out.shape)}  loss={loss.item():.4f}")
        print(f"  ✓  Gradient isolation verified (base frozen, adapters trainable)")
    except Exception as e:
        print(f"  ✗  Forward: {e}", file=sys.stderr)
        all_passed = False

    # ── 4. Merge / Unmerge consistency ────────────────────────────────────────
    print("\n── Merge / Unmerge consistency ─────────────────────────────────")
    try:
        # Simulate some training: randomise lora_B so ΔW ≠ 0
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    nn.init.normal_(module.lora_B, std=0.01)

        model.eval()
        x_test = torch.randn(1, 8, D, device=device)
        with torch.no_grad():
            out_before_merge = model(x_test).clone()

        # Merge and re-evaluate
        merge_lora(model)
        with torch.no_grad():
            out_after_merge = model(x_test)
        diff_merge = (out_before_merge - out_after_merge).abs().max().item()
        print(f"  ✓  Merge consistency: |before − after| = {diff_merge:.2e}  "
              f"(should be ≈ 0)")
        assert diff_merge < 1e-5, f"Merge broke output! diff={diff_merge}"

        # Unmerge and re-evaluate
        unmerge_lora(model)
        with torch.no_grad():
            out_after_unmerge = model(x_test)
        diff_unmerge = (out_before_merge - out_after_unmerge).abs().max().item()
        print(f"  ✓  Unmerge consistency: |original − unmerged| = {diff_unmerge:.2e}")
        assert diff_unmerge < 1e-5, f"Unmerge broke output! diff={diff_unmerge}"

    except Exception as e:
        print(f"  ✗  Merge/Unmerge: {e}", file=sys.stderr)
        all_passed = False

    # ── 5. Checkpoint round-trip ──────────────────────────────────────────────
    print("\n── Checkpoint round-trip ───────────────────────────────────────")
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name

        save_lora(model, cfg, ckpt_path, step=1000, loss=2.34)

        # Build a fresh model with same base weights, apply LoRA, then load
        model2 = TinyModel(d=D, n_layers=3).to(device)
        # Give model2 the same base weights as model (simulate loading from checkpoint)
        base_sd = {n: p.data for n, p in model.named_parameters() if not p.requires_grad}
        for n, p in model2.named_parameters():
            if n in base_sd:
                p.data.copy_(base_sd[n])
        apply_lora(model2, cfg)
        model2, loaded_cfg = load_lora(model2, ckpt_path, device=str(device))

        model.eval(); model2.eval()
        with torch.no_grad():
            o1 = model(x_test)
            o2 = model2(x_test)
        diff_ckpt = (o1 - o2).abs().max().item()
        os.unlink(ckpt_path)

        # Check saved file size
        print(f"  ✓  Checkpoint diff: {diff_ckpt:.2e}")
        print(f"  ✓  Loaded config: rank={loaded_cfg.rank}  alpha={loaded_cfg.alpha}")
        assert diff_ckpt < 1e-5, f"Checkpoint round-trip failed: {diff_ckpt}"
    except Exception as e:
        print(f"  ✗  Checkpoint: {e}", file=sys.stderr)
        all_passed = False

    # ── 6. Rank ablation: trainable param count across ranks ──────────────────
    print("\n── Rank ablation: parameter count across ranks ──────────────────")
    print(f"  {'Rank':>6}  {'Trainable':>12}  {'% of total':>12}  {'Scale (α/r)':>12}")
    for r in [1, 2, 4, 8, 16, 32]:
        m_abl = TinyModel(d=D, n_layers=3)
        apply_lora(m_abl, LoRAConfig(rank=r, alpha=float(r)))
        s = print_trainable_params(m_abl, verbose=False)
        scale = r / r   # alpha=rank by default → scale=1
        print(f"  {r:>6}  {s['trainable']:>12,}  {100*s['trainable']/s['total']:>11.3f}%  "
              f"{scale:>12.2f}")

    # ── 7. lora_summary ───────────────────────────────────────────────────────
    print()
    lora_summary(model)

    print("\n" + ("All tests PASSED ✓" if all_passed else "Some tests FAILED ✗"))