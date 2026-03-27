from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.lora import LoRAConfig, apply_lora, load_lora
from model.nanogpt import NanoGPT
from model.tokenizer import BPETokenizer


@torch.no_grad()
def generate_text(
    model: NanoGPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new: int,
    temperature: float,
    top_k: int,
) -> str:
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new=max_new,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(out[0].tolist())


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    ckpt_path = Path(args.base_checkpoint)
    adapter_path = Path(args.adapter_path)
    tok_path = Path(args.tokenizer_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {ckpt_path}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

    tokenizer = BPETokenizer.load(tok_path)

    # Base model generation
    base_model = NanoGPT.from_checkpoint(str(ckpt_path), device=str(device)).to(device).eval()
    base_text = generate_text(
        base_model,
        tokenizer,
        prompt=args.prompt,
        max_new=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # LoRA model generation
    lora_model = NanoGPT.from_checkpoint(str(ckpt_path), device=str(device)).to(device)
    lora_cfg = LoRAConfig(
        rank=args.rank,
        alpha=args.alpha,
        dropout=0.0,
        target_modules=set(args.target_modules),
        bias=args.bias,
    )
    apply_lora(lora_model, lora_cfg)
    lora_model, loaded_cfg = load_lora(lora_model, str(adapter_path), device=str(device))
    lora_model.eval()
    lora_text = generate_text(
        lora_model,
        tokenizer,
        prompt=args.prompt,
        max_new=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / "before_after.txt"
    txt_path.write_text(
        "PROMPT:\n"
        + args.prompt
        + "\n\nBASE_OUTPUT:\n"
        + base_text
        + "\n\nLORA_OUTPUT:\n"
        + lora_text
        + "\n",
        encoding="utf-8",
    )

    json_path = out_dir / "before_after.json"
    payload = {
        "prompt": args.prompt,
        "base_checkpoint": str(ckpt_path),
        "adapter_path": str(adapter_path),
        "lora_config_loaded": loaded_cfg.to_dict(),
        "generation_args": {
            "max_new": args.max_new,
            "temperature": args.temperature,
            "top_k": args.top_k,
        },
        "base_output": base_text,
        "lora_output": lora_text,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved: {txt_path}")
    print(f"Saved: {json_path}")
    print("\n--- BASE (preview) ---")
    print(base_text[:500].replace("\n", "\\n"))
    print("\n--- LORA (preview) ---")
    print(lora_text[:500].replace("\n", "\\n"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base vs LoRA generation outputs")
    parser.add_argument("--base_checkpoint", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer.json")
    parser.add_argument("--out_dir", type=str, default="results/lora")
    parser.add_argument("--prompt", type=str, default="O moon,")
    parser.add_argument("--max_new", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)

    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["W_q", "W_k", "W_v", "W_o"],
        help="Must match LoRA training target modules",
    )
    parser.add_argument("--bias", type=str, default="none", choices=["none", "lora", "all"])
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

