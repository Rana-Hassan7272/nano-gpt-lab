from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

Pair = Tuple[int, int]


@dataclass(frozen=True)
class BPETokenizerConfig:
    vocab_size: int


class BPETokenizer:
    """Simple byte-level BPE tokenizer built from scratch."""

    def __init__(self) -> None:
        self.base_vocab_size = 256
        self.merges: List[Pair] = []
        self.token_bytes: Dict[int, bytes] = {i: bytes([i]) for i in range(self.base_vocab_size)}
        self.config = BPETokenizerConfig(vocab_size=self.base_vocab_size)

    def train(self, text: str, vocab_size: int) -> None:
        if vocab_size < self.base_vocab_size:
            raise ValueError(f"vocab_size must be >= {self.base_vocab_size}")

        token_ids = list(text.encode("utf-8"))
        self.merges = []
        self.token_bytes = {i: bytes([i]) for i in range(self.base_vocab_size)}

        next_token_id = self.base_vocab_size
        target_merges = vocab_size - self.base_vocab_size

        for _ in range(target_merges):
            pair_counts = self._count_pairs(token_ids)
            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break

            self.merges.append(best_pair)
            self.token_bytes[next_token_id] = (
                self.token_bytes[best_pair[0]] + self.token_bytes[best_pair[1]]
            )
            token_ids = self._merge_tokens(token_ids, best_pair, next_token_id)
            next_token_id += 1

        self.config = BPETokenizerConfig(vocab_size=self.base_vocab_size + len(self.merges))

    def encode(self, text: str) -> List[int]:
        tokens = list(text.encode("utf-8"))
        for i, pair in enumerate(self.merges):
            merged_token_id = self.base_vocab_size + i
            tokens = self._merge_tokens(tokens, pair, merged_token_id)
        return tokens

    def decode(self, token_ids: Iterable[int]) -> str:
        byte_stream = bytearray()
        for token_id in token_ids:
            token_bytes = self.token_bytes.get(token_id)
            if token_bytes is None:
                raise ValueError(f"Unknown token id: {token_id}")
            byte_stream.extend(token_bytes)
        return byte_stream.decode("utf-8", errors="strict")

    def save(self, path: str | Path) -> None:
        payload = {
            "vocab_size": self.config.vocab_size,
            "merges": [[a, b] for a, b in self.merges],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls()
        tokenizer.merges = [tuple(pair) for pair in payload.get("merges", [])]

        tokenizer.token_bytes = {i: bytes([i]) for i in range(tokenizer.base_vocab_size)}
        for i, (left, right) in enumerate(tokenizer.merges):
            token_id = tokenizer.base_vocab_size + i
            tokenizer.token_bytes[token_id] = tokenizer.token_bytes[left] + tokenizer.token_bytes[right]

        tokenizer.config = BPETokenizerConfig(
            vocab_size=payload.get("vocab_size", tokenizer.base_vocab_size + len(tokenizer.merges))
        )
        return tokenizer

    @staticmethod
    def _count_pairs(tokens: List[int]) -> Counter[Pair]:
        if len(tokens) < 2:
            return Counter()
        return Counter(zip(tokens, tokens[1:]))

    @staticmethod
    def _merge_tokens(tokens: List[int], pair: Pair, new_token_id: int) -> List[int]:
        merged: List[int] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_token_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged
