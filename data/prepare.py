from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.tokenizer import BPETokenizer


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_indices(n_tokens: int, train_ratio: float = 0.9) -> Tuple[int, int]:
    train_n = int(n_tokens * train_ratio)
    val_n = n_tokens - train_n
    return train_n, val_n


def save_bin(tokens: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save as int32 raw binary for fast mmap loading later
    tokens.astype(np.int32).tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TinyShakespeare with BPE tokenizer")
    parser.add_argument("--input", type=str, default="data/input.txt", help="raw text file path")
    parser.add_argument("--out_dir", type=str, default="data", help="output directory")
    parser.add_argument("--vocab_size", type=int, default=8000, help="BPE vocab size")
    parser.add_argument(
        "--train_subset_chars",
        type=int,
        default=0,
        help="If >0, use only the first N characters to TRAIN the tokenizer (encode uses full text).",
    )
    parser.add_argument("--train_ratio", type=float, default=0.9, help="train split ratio")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    assert input_path.exists(), f"Input file not found: {input_path}"

    print(f"Loading text from {input_path} ...")
    text = load_text(input_path)
    print(f"Loaded {len(text):,} characters")

    # Train tokenizer (optionally on a smaller subset for speed)
    train_text = text
    if args.train_subset_chars and args.train_subset_chars > 0:
        train_text = text[: args.train_subset_chars]
        print(f"Training on subset: first {len(train_text):,} characters (for speed)")
    print(f"Training BPE tokenizer (vocab_size={args.vocab_size}) ...")
    tokenizer = BPETokenizer()
    tokenizer.train(train_text, vocab_size=args.vocab_size)
    tok_path = out_dir / "tokenizer.json"
    tokenizer.save(tok_path)
    print(f"Saved tokenizer to {tok_path}")

    # Encode full corpus
    print("Encoding full corpus ...")
    ids = tokenizer.encode(text)
    ids_np = np.array(ids, dtype=np.int32)
    print(f"Total tokens: {ids_np.size:,}")

    # Train/val split
    train_n, val_n = split_indices(ids_np.size, args.train_ratio)
    train_ids = ids_np[:train_n]
    val_ids = ids_np[train_n:]
    print(f"Split: train={train_ids.size:,}  val={val_ids.size:,}  "
          f"({args.train_ratio*100:.1f}% / {(1-args.train_ratio)*100:.1f}%)")

    # Save binaries
    train_bin = out_dir / "train.bin"
    val_bin = out_dir / "val.bin"
    save_bin(train_ids, train_bin)
    save_bin(val_ids, val_bin)
    print(f"Wrote: {train_bin} ({train_ids.nbytes/1e6:.2f} MB)")
    print(f"Wrote: {val_bin}   ({val_ids.nbytes/1e6:.2f} MB)")

    # Quick decode sanity check (first 200 tokens of val)
    sample_len = min(200, val_ids.size)
    try:
        sample_text = tokenizer.decode(val_ids[:sample_len].tolist())
        print("Decode sanity check (first ~200 val tokens):")
        print(sample_text.replace("\n", "\\n")[:200])
    except Exception as e:
        print(f"Decode check skipped due to error: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
