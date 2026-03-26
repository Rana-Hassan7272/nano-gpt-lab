from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.nanogpt import NanoGPT, NanoGPTConfig


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cpu")

    # Tiny fake setup for fast CPU verification only.
    cfg = NanoGPTConfig(
        vocab_size=300,
        context_len=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        ffn_variant="standard",
        norm_type="rmsnorm",
        pos_encoding="rope",
        dropout=0.0,
        use_flash=False,
    )
    model = NanoGPT(cfg).to(device)
    model.train()

    batch_size = 4
    seq_len = 32
    idx = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)

    # Forward + shape check.
    logits, loss, _ = model(idx, targets)
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)

    # Backward + one optimizer step.
    optimizer = model.configure_optimizer(learning_rate=3e-4, weight_decay=0.01, device_type="cpu")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Small generation sanity check.
    model.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    with torch.no_grad():
        out = model.generate(prompt, max_new=8, greedy=True)
    assert out.shape == (1, 16)

    print("OK: Phase 1 Step 7 CPU verification passed")
    print(f"device={device}")
    print(f"input_shape={tuple(idx.shape)} logits_shape={tuple(logits.shape)}")
    print(f"loss={loss.item():.4f} grad_norm={float(grad_norm):.4f}")
    print(f"generation_shape={tuple(out.shape)}")


if __name__ == "__main__":
    main()
