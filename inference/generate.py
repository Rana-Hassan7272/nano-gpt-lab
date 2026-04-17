"""
inference/generate.py — NanoGPT Inference Engine
=================================================
Author  : NanoGPT Lab
Phase   : 6, Step 1

Four decoding strategies implemented as clean, standalone functions:
  1. greedy_decode        — deterministic argmax
  2. temperature_sample   — softmax with temperature scaling
  3. top_k_sample         — vocabulary truncation to k candidates
  4. top_p_sample         — nucleus sampling (adaptive vocabulary)
  5. generate             — unified entry point used by the API

Mathematical background
-----------------------
All strategies operate on the same raw logits ℓ ∈ R^V produced by the LM head.
They differ in how they convert logits to a sampling distribution:

  Greedy:      token = argmax(ℓ)                       — no randomness
  Temperature: p_i = exp(ℓ_i / T) / Σ exp(ℓ_j / T)   — full vocab, rescaled
  Top-k:       zero all but the k largest ℓ_i, then temperature-sample
  Top-p:       zero all ℓ_i outside the smallest set S where Σ_{i∈S} p_i ≥ p

These are composable: top_k then top_p is a common production combination.
Temperature is always applied before top-k/top-p filtering in this engine.

KV-cache
--------
All generators use the model's KV-cache for O(T) inference instead of O(T²).
Prefill: run the full prompt through the model once to warm all layer caches.
Decode: feed one token at a time; each step is O(1) in sequence length.

Streaming
---------
All generators are Python generators (yield each token as it is produced).
The FastAPI server uses this to stream tokens to the client via SSE.
"""

import math
import time
from typing import Generator, Optional, Tuple, List

import torch
import torch.nn.functional as F


# ===========================================================================
# Logit processor utilities
# ===========================================================================

def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Divide logits by temperature before softmax.

    T → 0  :  argmax (deterministic)
    T = 1  :  model's true distribution
    T → ∞  :  uniform distribution

    We clamp to 1e-8 to avoid division by zero.
    """
    return logits / max(temperature, 1e-8)


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Set all logits below the k-th largest to -inf.

    Only the k most probable tokens remain in the distribution.
    This eliminates the long tail of near-zero probability tokens
    that can produce incoherent output.

    Computational note: torch.topk is O(V log k), not O(V log V).
    """
    k = min(k, logits.size(-1))
    # Find the value of the k-th largest logit
    kth_value = logits.topk(k, dim=-1).values[..., -1, None]  # (B, 1)
    # Mask everything below this threshold
    return logits.masked_fill(logits < kth_value, float("-inf"))


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling: keep the smallest vocabulary subset whose cumulative
    probability mass exceeds p.  (Holtzman et al. 2020)

    Why this is better than fixed top-k:
    - On a confident step (e.g. after 'The sky is'), the top token might
      have prob 0.97 — top-k=40 unnecessarily adds 39 low-quality options.
    - On an uncertain step, many tokens have similar probability — top-k=40
      may be too restrictive and cause repetition.
    Nucleus sampling adapts: it uses 1 token when the model is confident and
    many tokens when the model is uncertain.

    Algorithm:
    1. Sort tokens by descending probability
    2. Compute cumulative probabilities
    3. Remove tokens once cumulative prob exceeds p (shift by 1 to keep ≥1 token)
    4. Unsort back to original vocabulary order
    """
    # Sort logits descending and compute softmax probs
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cum_probs    = sorted_probs.cumsum(dim=-1)

    # Tokens to remove: those where cumulative prob (excluding self) > p
    # Shift right by 1 so we always keep at least the top token
    remove = cum_probs - sorted_probs > p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

    # Unsort: scatter filtered logits back to original vocabulary positions
    logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    return logits


def _sample_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Sample one token per batch element from logit scores.
    Returns shape (B, 1).
    """
    probs = F.softmax(logits, dim=-1)   # (B, V)
    return torch.multinomial(probs, num_samples=1)   # (B, 1)


# ===========================================================================
# Decoding strategy 1 — Greedy
# ===========================================================================

def greedy_decode(
    model,
    idx:     torch.Tensor,       # (B, T) prompt token ids
    max_new: int = 200,
    device:  Optional[torch.device] = None,
) -> Generator[torch.Tensor, None, torch.Tensor]:
    """
    Greedy decoding: always select the highest-probability next token.

        token_t = argmax_v  P(v | t_0 … t_{t-1})

    Properties:
    - Deterministic: same prompt → same output every time
    - Fast: no sampling overhead
    - Limitation: tends toward repetition and "safe" continuations
    - Does NOT maximise sequence probability:
        argmax P(token_t | context) ≠ argmax P(full sequence)
      Beam search is needed for true sequence-level maximisation,
      but beam search is rarely better for open-ended generation.

    Yields each new token tensor (B, 1) as it is generated.
    """
    model.eval()
    if device is not None:
        idx = idx.to(device)

    # Prefill: warm the KV cache with the full prompt
    with torch.no_grad():
        _, _, kv_caches = model(idx)
    cur = idx[:, -1:]   # (B, 1) — last prompt token, first decode input

    for _ in range(max_new):
        with torch.no_grad():
            logits, _, kv_caches = model(cur, kv_caches=kv_caches)
        logits    = logits[:, -1, :]                        # (B, V)
        next_tok  = logits.argmax(dim=-1, keepdim=True)    # (B, 1)
        cur       = next_tok
        idx       = torch.cat([idx, next_tok], dim=1)
        yield next_tok


# ===========================================================================
# Decoding strategy 2 — Temperature sampling
# ===========================================================================

def temperature_sample(
    model,
    idx:         torch.Tensor,
    max_new:     int   = 200,
    temperature: float = 1.0,
    device:      Optional[torch.device] = None,
) -> Generator[torch.Tensor, None, torch.Tensor]:
    """
    Sample from the full vocabulary after rescaling logits by temperature T.

        p_i ∝ exp(ℓ_i / T)

    T < 1  →  sharpens distribution  →  more focused, repetitive
    T = 1  →  model's true distribution
    T > 1  →  flattens distribution   →  more creative, less coherent

    Practical guidance:
    - Creative writing:    T = 0.8 – 1.1
    - Code / factual:      T = 0.2 – 0.5
    - Max diversity:       T = 1.2 – 1.5  (risk of incoherence)

    Yields each new token tensor (B, 1) as it is generated.
    """
    model.eval()
    if device is not None:
        idx = idx.to(device)

    with torch.no_grad():
        _, _, kv_caches = model(idx)
    cur = idx[:, -1:]

    for _ in range(max_new):
        with torch.no_grad():
            logits, _, kv_caches = model(cur, kv_caches=kv_caches)
        logits   = logits[:, -1, :]
        logits   = _apply_temperature(logits, temperature)
        next_tok = _sample_from_logits(logits)
        cur      = next_tok
        idx      = torch.cat([idx, next_tok], dim=1)
        yield next_tok


# ===========================================================================
# Decoding strategy 3 — Top-k sampling
# ===========================================================================

def top_k_sample(
    model,
    idx:         torch.Tensor,
    max_new:     int   = 200,
    k:           int   = 40,
    temperature: float = 1.0,
    device:      Optional[torch.device] = None,
) -> Generator[torch.Tensor, None, torch.Tensor]:
    """
    Sample from the top-k most probable tokens only.  (Fan et al. 2018)

    After filtering to top-k, temperature is applied and we sample.
    This eliminates the long vocabulary tail that produces incoherent tokens,
    while retaining randomness among the plausible candidates.

    k=1       →  greedy (deterministic)
    k=vocab   →  pure temperature sampling
    k=40      →  standard default, good for language generation

    Yields each new token tensor (B, 1) as it is generated.
    """
    model.eval()
    if device is not None:
        idx = idx.to(device)

    with torch.no_grad():
        _, _, kv_caches = model(idx)
    cur = idx[:, -1:]

    for _ in range(max_new):
        with torch.no_grad():
            logits, _, kv_caches = model(cur, kv_caches=kv_caches)
        logits   = logits[:, -1, :]
        logits   = _apply_temperature(logits, temperature)
        logits   = _apply_top_k(logits, k)
        next_tok = _sample_from_logits(logits)
        cur      = next_tok
        idx      = torch.cat([idx, next_tok], dim=1)
        yield next_tok


# ===========================================================================
# Decoding strategy 4 — Top-p (nucleus) sampling
# ===========================================================================

def top_p_sample(
    model,
    idx:         torch.Tensor,
    max_new:     int   = 200,
    p:           float = 0.9,
    temperature: float = 1.0,
    device:      Optional[torch.device] = None,
) -> Generator[torch.Tensor, None, torch.Tensor]:
    """
    Nucleus sampling: sample from the smallest set of tokens whose cumulative
    probability exceeds p.  (Holtzman et al. "The Curious Case of Neural Text
    Degeneration", 2020)

    Unlike top-k, the nucleus size adapts per step:
    - Confident step: nucleus may be 1-3 tokens
    - Uncertain step: nucleus may be 50-100 tokens

    p=0.95  →  covers 95% of probability mass per step
    p=0.9   →  covers 90%  (slightly more focused)
    p=1.0   →  pure temperature sampling (no nucleus filter)

    Yields each new token tensor (B, 1) as it is generated.
    """
    model.eval()
    if device is not None:
        idx = idx.to(device)

    with torch.no_grad():
        _, _, kv_caches = model(idx)
    cur = idx[:, -1:]

    for _ in range(max_new):
        with torch.no_grad():
            logits, _, kv_caches = model(cur, kv_caches=kv_caches)
        logits   = logits[:, -1, :]
        logits   = _apply_temperature(logits, temperature)
        logits   = _apply_top_p(logits, p)
        next_tok = _sample_from_logits(logits)
        cur      = next_tok
        idx      = torch.cat([idx, next_tok], dim=1)
        yield next_tok


# ===========================================================================
# Unified generate() — used by the FastAPI server
# ===========================================================================

def generate(
    model,
    idx:         torch.Tensor,
    max_new:     int            = 200,
    strategy:    str            = "top_p",
    temperature: float          = 0.8,
    top_k:       Optional[int]  = 40,
    top_p:       Optional[float]= 0.9,
    device:      Optional[torch.device] = None,
    stream:      bool           = False,
) -> Tuple[str, dict]:
    """
    Unified generation entry point.

    Parameters
    ----------
    model       : NanoGPT instance (already .eval(), on device)
    idx         : (1, T) prompt token ids
    max_new     : tokens to generate
    strategy    : 'greedy' | 'temperature' | 'top_k' | 'top_p'
    temperature : softmax temperature
    top_k       : k for top-k sampling (ignored if strategy != 'top_k')
    top_p       : p for nucleus sampling (ignored if strategy != 'top_p')
    device      : target device
    stream      : if True, returns a generator of token tensors

    Returns
    -------
    tokens  : (1, T + max_new) full sequence including prompt
    meta    : dict with timing, tokens_per_second, strategy used
    """
    t0 = time.perf_counter()

    strategy_map = {
        "greedy":      lambda: greedy_decode(model, idx, max_new, device),
        "temperature": lambda: temperature_sample(model, idx, max_new, temperature, device),
        "top_k":       lambda: top_k_sample(model, idx, max_new, top_k or 40, temperature, device),
        "top_p":       lambda: top_p_sample(model, idx, max_new, top_p or 0.9, temperature, device),
    }

    if strategy not in strategy_map:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         f"Choose from: {list(strategy_map.keys())}")

    generator = strategy_map[strategy]()

    if stream:
        return generator, {}

    # Collect all tokens
    all_new = []
    for tok in generator:
        all_new.append(tok)

    elapsed = time.perf_counter() - t0
    full_seq = torch.cat([idx] + all_new, dim=1)

    meta = {
        "strategy":          strategy,
        "tokens_generated":  len(all_new),
        "elapsed_seconds":   round(elapsed, 3),
        "tokens_per_second": round(len(all_new) / max(elapsed, 1e-6), 1),
        "temperature":       temperature,
        "top_k":             top_k,
        "top_p":             top_p,
    }
    return full_seq, meta


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    """
    Smoke-test all four strategies with a tiny mock model.
    Tests: output shape, no crash, streaming works, timing metadata present.
    """
    import sys

    class MockModel(torch.nn.Module):
        """Tiny mock that returns random logits — no real weights needed."""
        def __init__(self, vocab: int = 100, d: int = 32):
            super().__init__()
            self.vocab = vocab
            self.linear = torch.nn.Linear(d, vocab)
        def forward(self, idx, targets=None, kv_caches=None):
            B, T = idx.shape
            logits = self.linear(torch.randn(B, T, 32))
            return logits, None, kv_caches   # kv_caches passthrough

    torch.manual_seed(42)
    model  = MockModel(vocab=100)
    prompt = torch.randint(0, 100, (1, 8))
    print("Testing all decoding strategies:\n")

    all_passed = True
    for strat, kwargs in [
        ("greedy",      {}),
        ("temperature", {"temperature": 0.7}),
        ("top_k",       {"top_k": 20, "temperature": 0.8}),
        ("top_p",       {"top_p": 0.9, "temperature": 0.8}),
    ]:
        try:
            seq, meta = generate(model, prompt, max_new=20, strategy=strat, **kwargs)
            assert seq.shape == (1, 28), f"Shape mismatch: {seq.shape}"
            assert meta["tokens_generated"] == 20
            print(f"  ✓  {strat:15s}  shape={tuple(seq.shape)}  "
                  f"{meta['tokens_per_second']:.0f} tok/s")
        except Exception as e:
            print(f"  ✗  {strat}: {e}", file=sys.stderr)
            all_passed = False

    # Test streaming
    try:
        gen, _ = generate(model, prompt, max_new=10, strategy="top_p", stream=True)
        toks = list(gen)
        assert len(toks) == 10
        print(f"  ✓  streaming      yielded {len(toks)} tokens one-by-one")
    except Exception as e:
        print(f"  ✗  streaming: {e}", file=sys.stderr)
        all_passed = False

    print("\n" + ("All tests PASSED ✓" if all_passed else "Some tests FAILED ✗"))