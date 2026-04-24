from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.lora import LoRAConfig, apply_lora, load_lora
from model.nanogpt import NanoGPT
from model.tokenizer import BPETokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_with_optional_comments(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Allow JSONC-style comments in prompt files for authoring convenience.
        cleaned = re.sub(r"//.*?$", "", raw, flags=re.MULTILINE)
        return json.loads(cleaned)


def generate_once(
    model: NanoGPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new: int,
    temperature: float,
    top_k: int,
    seed: int,
    device: torch.device,
) -> str:
    set_seed(seed)
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new=max_new,
        temperature=temperature,
        top_k=top_k,
    )
    text = tokenizer.decode(out[0].tolist())
    # Return only generated continuation to keep comparison focused.
    return text[len(prompt) :]


def load_model_pair(
    checkpoint: Path,
    adapter: Path,
    rank: int,
    alpha: float,
    target_modules: List[str],
    bias: str,
    device: torch.device,
) -> Tuple[NanoGPT, NanoGPT]:
    baseline = NanoGPT.from_checkpoint(str(checkpoint), device=str(device)).to(device).eval()

    lora = NanoGPT.from_checkpoint(str(checkpoint), device=str(device)).to(device)
    cfg = LoRAConfig(
        rank=rank,
        alpha=alpha,
        dropout=0.0,
        target_modules=set(target_modules),
        bias=bias,
    )
    apply_lora(lora, cfg)
    lora, _ = load_lora(lora, str(adapter), device=str(device))
    lora.eval()
    return baseline, lora


def infer_lora_hparams_from_adapter(adapter_path: Path) -> Dict[str, Any]:
    payload = torch.load(str(adapter_path), map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    if not isinstance(cfg, dict):
        return {}
    return {
        "rank": cfg.get("rank"),
        "alpha": cfg.get("alpha"),
        "target_modules": cfg.get("target_modules"),
        "bias": cfg.get("bias"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate blinded prompt benchmark outputs for baseline vs LoRA.")
    p.add_argument("--prompt_set", type=str, default="results/eval/prompt_eval_set.json")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--adapter_path", type=str, required=True)
    p.add_argument("--tokenizer_path", type=str, default="data/tokenizer.json")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_new", type=int, default=None, help="Override max_new from prompt set generation_config")
    p.add_argument("--temperature", type=float, default=None, help="Override temperature from prompt set generation_config")
    p.add_argument("--top_k", type=int, default=None, help="Override top_k from prompt set generation_config")
    p.add_argument("--seed", type=int, default=None, help="Override seed base from prompt set generation_config")
    p.add_argument("--rank", type=int, default=None, help="Optional override; otherwise inferred from adapter")
    p.add_argument("--alpha", type=float, default=None, help="Optional override; otherwise inferred from adapter")
    p.add_argument(
        "--target_modules",
        nargs="+",
        default=None,
        help="Optional override; otherwise inferred from adapter (or defaults)",
    )
    p.add_argument(
        "--bias",
        type=str,
        default=None,
        choices=["none", "lora", "all"],
        help="Optional override; otherwise inferred from adapter (or 'none')",
    )
    p.add_argument("--output_json", type=str, default="results/eval/prompt_benchmark_outputs.json")
    p.add_argument("--ratings_template_json", type=str, default="results/eval/prompt_benchmark_ratings_template.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prompt_set_path = Path(args.prompt_set)
    checkpoint = Path(args.checkpoint)
    adapter = Path(args.adapter_path)
    tokenizer_path = Path(args.tokenizer_path)

    for path, label in [
        (prompt_set_path, "prompt set"),
        (checkpoint, "checkpoint"),
        (adapter, "adapter"),
        (tokenizer_path, "tokenizer"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    prompt_set = load_json_with_optional_comments(prompt_set_path)
    prompts = prompt_set.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompt set must contain a non-empty 'prompts' list")

    gen_cfg = prompt_set.get("generation_config", {})
    max_new = int(args.max_new if args.max_new is not None else gen_cfg.get("max_new", 80))
    temperature = float(args.temperature if args.temperature is not None else gen_cfg.get("temperature", 0.8))
    top_k = int(args.top_k if args.top_k is not None else gen_cfg.get("top_k", 40))
    base_seed = int(args.seed if args.seed is not None else gen_cfg.get("seed", 42))

    device = torch.device(args.device)
    tokenizer = BPETokenizer.load(tokenizer_path)

    inferred = infer_lora_hparams_from_adapter(adapter)
    rank = int(args.rank if args.rank is not None else (inferred.get("rank") or 4))
    alpha = float(args.alpha if args.alpha is not None else (inferred.get("alpha") or float(rank)))
    target_modules = (
        args.target_modules
        if args.target_modules is not None
        else (inferred.get("target_modules") or ["W_q", "W_k", "W_v", "W_o"])
    )
    bias = str(args.bias if args.bias is not None else (inferred.get("bias") or "none"))

    baseline_model, lora_model = load_model_pair(
        checkpoint=checkpoint,
        adapter=adapter,
        rank=rank,
        alpha=alpha,
        target_modules=target_modules,
        bias=bias,
        device=device,
    )

    outputs: List[Dict[str, Any]] = []
    ratings_template: List[Dict[str, Any]] = []
    mapping: Dict[str, Dict[str, str]] = {}

    for i, item in enumerate(prompts):
        pid = str(item.get("id", f"p{i+1:03d}"))
        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            continue

        seed = base_seed + i
        baseline_text = generate_once(
            baseline_model, tokenizer, prompt, max_new, temperature, top_k, seed, device
        )
        lora_text = generate_once(
            lora_model, tokenizer, prompt, max_new, temperature, top_k, seed, device
        )

        # Blind order to reduce rater bias.
        if random.Random(seed).random() < 0.5:
            variant_a, variant_b = baseline_text, lora_text
            mapping[pid] = {"A": "baseline", "B": "lora"}
        else:
            variant_a, variant_b = lora_text, baseline_text
            mapping[pid] = {"A": "lora", "B": "baseline"}

        outputs.append(
            {
                "id": pid,
                "prompt": prompt,
                "intent": item.get("intent"),
                "tags": item.get("tags", []),
                "seed": seed,
                "variant_a": variant_a,
                "variant_b": variant_b,
            }
        )
        ratings_template.append(
            {
                "id": pid,
                "winner": None,  # "A" | "B" | "tie"
                "scores": {
                    "A": {
                        "style_adherence": None,
                        "coherence": None,
                        "relevance": None,
                        "fluency": None,
                    },
                    "B": {
                        "style_adherence": None,
                        "coherence": None,
                        "relevance": None,
                        "fluency": None,
                    },
                },
                "notes": "",
            }
        )

    payload = {
        "metadata": prompt_set.get("metadata", {}),
        "generation_config": {
            "max_new": max_new,
            "temperature": temperature,
            "top_k": top_k,
            "seed_base": base_seed,
            "strategy": "top_k",
            "source": "cli_override" if any(
                x is not None for x in [args.max_new, args.temperature, args.top_k, args.seed]
            ) else "prompt_set_generation_config",
        },
        "models": {
            "baseline_checkpoint": str(checkpoint),
            "lora_adapter": str(adapter),
            "rank": rank,
            "alpha": alpha,
            "target_modules": target_modules,
            "bias": bias,
        },
        "outputs": outputs,
        "blind_mapping": mapping,
    }

    ratings_payload = {
        "metadata": {
            "instructions": "Fill winner and 1-5 scores for each dimension. Keep A/B labels blinded.",
            "scale": "1-5",
            "dimensions": ["style_adherence", "coherence", "relevance", "fluency"],
        },
        "ratings": ratings_template,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rating_json = Path(args.ratings_template_json)
    rating_json.parent.mkdir(parents=True, exist_ok=True)
    rating_json.write_text(json.dumps(ratings_payload, indent=2), encoding="utf-8")

    print(f"Saved outputs: {out_json}")
    print(f"Saved rating template: {rating_json}")
    print(f"Total prompts generated: {len(outputs)}")


if __name__ == "__main__":
    main()

