from __future__ import annotations

import math

import torch


class WarmupCosineScheduler:
    """
    Learning-rate schedule used by modern GPT training:
      1) Linear warmup for first N steps
      2) Cosine decay from base_lr to min_lr after warmup
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if not (0.0 <= min_lr_ratio <= 1.0):
            raise ValueError("min_lr_ratio must be in [0, 1]")

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max(max_steps, warmup_steps + 1)
        self.min_lr_ratio = min_lr_ratio
        self.step_num = 0
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def _scale(self, step: int) -> float:
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return float(step) / float(self.warmup_steps)

        decay_steps = self.max_steps - self.warmup_steps
        progress = (step - self.warmup_steps) / float(decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self) -> None:
        self.step_num += 1
        scale = self._scale(self.step_num)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale

    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])
