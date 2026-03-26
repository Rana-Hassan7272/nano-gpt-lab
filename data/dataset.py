from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TokenChunkDataset(Dataset):
    """
    Dataset over token-id binary files for next-token prediction.

    Each sample returns:
      - x: tokens[i : i + context_len]
      - y: tokens[i + 1 : i + 1 + context_len]
    """

    def __init__(self, bin_path: str | Path, context_len: int) -> None:
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Token binary not found: {self.bin_path}")
        if context_len < 1:
            raise ValueError("context_len must be >= 1")

        self.context_len = context_len
        self.tokens = np.memmap(self.bin_path, dtype=np.int32, mode="r")
        if self.tokens.shape[0] <= context_len:
            raise ValueError(
                f"Not enough tokens ({self.tokens.shape[0]}) for context_len={context_len}"
            )

    def __len__(self) -> int:
        # Need context_len + 1 tokens per sample because targets are shifted by 1.
        return int(self.tokens.shape[0] - self.context_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range: {idx}")

        x_np = np.array(self.tokens[idx : idx + self.context_len], dtype=np.int64)
        y_np = np.array(self.tokens[idx + 1 : idx + 1 + self.context_len], dtype=np.int64)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        return x, y


def build_dataloader(
    bin_path: str | Path,
    context_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    dataset = TokenChunkDataset(bin_path=bin_path, context_len=context_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )
