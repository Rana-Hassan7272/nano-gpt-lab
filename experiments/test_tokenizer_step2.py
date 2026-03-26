from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.tokenizer import BPETokenizer


def main() -> None:
    sample_text = (
        "To be, or not to be, that is the question.\n"
        "Whether tis nobler in the mind to suffer."
    )

    tokenizer = BPETokenizer()
    tokenizer.train(sample_text, vocab_size=300)

    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == sample_text, "Round-trip encode/decode failed"

    out_path = Path("results/tokenizer_step2.json")
    tokenizer.save(out_path)
    reloaded = BPETokenizer.load(out_path)
    re_encoded = reloaded.encode(sample_text)
    re_decoded = reloaded.decode(re_encoded)
    assert re_decoded == sample_text, "Reloaded tokenizer round-trip failed"

    print("OK: Step 2 tokenizer test passed")
    print(f"Trained vocab_size={tokenizer.config.vocab_size}")
    print(f"Encoded length={len(encoded)}")
    print(f"Model file={out_path.resolve()}")


if __name__ == "__main__":
    main()
