from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import build_dataloader
from model.tokenizer import BPETokenizer


def main() -> None:
    train_bin = PROJECT_ROOT / "data" / "train.bin"
    val_bin = PROJECT_ROOT / "data" / "val.bin"
    tokenizer_path = PROJECT_ROOT / "data" / "tokenizer.json"

    assert train_bin.exists(), f"Missing file: {train_bin}"
    assert val_bin.exists(), f"Missing file: {val_bin}"
    assert tokenizer_path.exists(), f"Missing file: {tokenizer_path}"

    context_len = 32
    batch_size = 4

    train_loader = build_dataloader(
        bin_path=train_bin,
        context_len=context_len,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = build_dataloader(
        bin_path=val_bin,
        context_len=context_len,
        batch_size=batch_size,
        shuffle=False,
    )

    x_train, y_train = next(iter(train_loader))
    x_val, y_val = next(iter(val_loader))

    assert x_train.shape == (batch_size, context_len)
    assert y_train.shape == (batch_size, context_len)
    assert x_val.shape == (batch_size, context_len)
    assert y_val.shape == (batch_size, context_len)

    tokenizer = BPETokenizer.load(tokenizer_path)
    decoded_x = tokenizer.decode(x_train[0].tolist())
    decoded_y = tokenizer.decode(y_train[0].tolist())

    print("OK: Phase 2 Step 4 data pipeline verification passed")
    print(f"train batch shapes: x={tuple(x_train.shape)} y={tuple(y_train.shape)}")
    print(f"val batch shapes:   x={tuple(x_val.shape)} y={tuple(y_val.shape)}")
    print(f"dtypes: x={x_train.dtype} y={y_train.dtype}")
    print("decoded sample x[0][:80]:", decoded_x[:80].replace("\n", "\\n"))
    print("decoded sample y[0][:80]:", decoded_y[:80].replace("\n", "\\n"))


if __name__ == "__main__":
    main()
