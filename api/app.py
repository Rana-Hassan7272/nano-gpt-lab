"""
api/app.py — NanoGPT FastAPI Inference Server
=============================================
Author  : NanoGPT Lab
Phase   : 6, Step 2

Endpoints
---------
POST /generate          — text generation with all four decoding strategies
GET  /generate/stream   — SSE streaming generation (token-by-token)
GET  /model/info        — architecture details, parameter counts
GET  /experiments       — all Phase 4 experiment results as structured JSON
GET  /health            — liveness probe for Docker / Railway

Run locally
-----------
    pip install fastapi uvicorn torch
    uvicorn api.app:app --reload --port 8000

With real checkpoint:
    MODEL_PATH=results/colab-checkpoints/exp1_step5000.pt uvicorn api.app:app --port 8000

Environment variables
---------------------
MODEL_PATH      : path to NanoGPT checkpoint  (default: results/colab-checkpoints/exp1_step5000.pt)
DEVICE          : 'cpu' | 'cuda'              (default: auto-detect)
MAX_NEW_TOKENS  : hard cap on generation      (default: 500)
RESULTS_DIR     : directory with experiment JSONs (default: results/)
STRICT_STARTUP  : '1'/'true' to fail fast if model/tokenizer cannot load
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

# ── Path setup: allow running from project root ───────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

# Lazy imports — model and generate loaded at startup to avoid startup crash
# if running tests without a checkpoint
_model      = None
_tokenizer  = None
_config     = None
_device     = None
_startup_mode = "unknown"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ===========================================================================
# Application setup
# ===========================================================================

app = FastAPI(
    title       = "NanoGPT Inference API",
    description = "Phase 6 inference server for NanoGPT language model",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# Allow the React dashboard (running on :3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # restrict to ["http://localhost:3000"] in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ===========================================================================
# Startup: load model once into memory
# ===========================================================================

@app.on_event("startup")
async def load_model() -> None:
    global _model, _tokenizer, _config, _device, _startup_mode

    device_str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    _device    = torch.device(device_str)
    strict_startup = _env_flag("STRICT_STARTUP", default=False)
    _startup_mode = "strict" if strict_startup else "stub-allowed"

    model_path = os.getenv(
        "MODEL_PATH",
        "results/colab-checkpoints/exp1_step5000.pt"
    )

    try:
        from nanogpt import NanoGPT
        _model  = NanoGPT.from_checkpoint(model_path, device=device_str)
        _model.eval()
        _config = _model.config
        print(f"[startup] Model loaded from {model_path} on {_device}")
    except Exception as e:
        if strict_startup:
            raise RuntimeError(
                f"STRICT_STARTUP enabled: failed to load model checkpoint '{model_path}': {e}"
            ) from e
        print(f"[startup] WARNING: Could not load model checkpoint: {e}")
        print("[startup] Running in stub mode — generation will return mock output")
        _model  = None
        _config = _MockConfig()

    # Try loading tokenizer
    try:
        from tokenizer import BPETokenizer
        tok_path = os.getenv("TOKENIZER_PATH", "data/tokenizer.json")
        _tokenizer = BPETokenizer.load(tok_path)
        print(f"[startup] Tokenizer loaded from {tok_path}")
    except Exception as e:
        if strict_startup:
            raise RuntimeError(
                f"STRICT_STARTUP enabled: failed to load tokenizer '{os.getenv('TOKENIZER_PATH', 'data/tokenizer.json')}': {e}"
            ) from e
        print(f"[startup] WARNING: Tokenizer not found ({e}). Using char-level fallback.")
        _tokenizer = _CharTokenizer()


class _MockConfig:
    """Stub config used when no checkpoint is available (dev/test mode)."""
    vocab_size = 8000; context_len = 256; d_model = 128
    n_layers = 4; n_heads = 4; ffn_variant = "standard"
    n_kv_heads = None; pos_encoding = "rope"; norm_type = "rmsnorm"
    weight_tying = True


class _CharTokenizer:
    """Minimal character-level tokenizer fallback."""
    def encode(self, text: str) -> List[int]:
        return [ord(c) % 8000 for c in text]
    def decode(self, ids: List[int]) -> str:
        return "".join(chr(max(32, min(126, i))) for i in ids)


# ===========================================================================
# Request / Response models
# ===========================================================================

class GenerateRequest(BaseModel):
    prompt:      str   = Field(...,    min_length=1, max_length=2000,
                               description="Input text prompt")
    max_new:     int   = Field(100,    ge=1, le=500,
                               description="Number of new tokens to generate")
    strategy:    str   = Field("top_p",
                               description="Decoding strategy: greedy | temperature | top_k | top_p")
    temperature: float = Field(0.8,    ge=0.01, le=2.0,
                               description="Softmax temperature (ignored for greedy)")
    top_k:       Optional[int]   = Field(40, ge=1, le=1000,
                               description="Top-k cutoff (only for top_k strategy)")
    top_p:       Optional[float] = Field(0.9, ge=0.01, le=1.0,
                               description="Nucleus probability (only for top_p strategy)")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v):
        valid = {"greedy", "temperature", "top_k", "top_p"}
        if v not in valid:
            raise ValueError(f"strategy must be one of {valid}")
        return v


class GenerateResponse(BaseModel):
    prompt:          str
    generated_text:  str
    full_text:       str
    tokens_generated: int
    elapsed_seconds: float
    tokens_per_second: float
    strategy:        str
    model_info:      Dict[str, Any]


class ModelInfoResponse(BaseModel):
    checkpoint_path:  str
    device:           str
    vocab_size:       int
    context_len:      int
    d_model:          int
    n_layers:         int
    n_heads:          int
    n_kv_heads:       Optional[int]
    ffn_variant:      str
    pos_encoding:     str
    norm_type:        str
    weight_tying:     bool
    total_parameters: int
    trainable_parameters: int
    parameter_breakdown: Dict[str, int]


# ===========================================================================
# Helper: encode prompt → tensor
# ===========================================================================

def _encode_prompt(prompt: str) -> torch.Tensor:
    """Encode text prompt to (1, T) token id tensor."""
    ids = _tokenizer.encode(prompt)
    # Truncate to leave room for generation
    max_ctx = _config.context_len - 50
    ids     = ids[-max_ctx:]
    return torch.tensor([ids], dtype=torch.long, device=_device)


def _decode_tokens(token_ids: List[int]) -> str:
    """Decode token ids to string."""
    return _tokenizer.decode(token_ids)


def _mock_generate(prompt: str, max_new: int) -> str:
    """Fallback when no model is loaded — returns informative stub."""
    return (f" [Model not loaded — running in stub mode. "
            f"Set MODEL_PATH env var to a valid checkpoint. "
            f"Prompt was: '{prompt[:50]}...', requested {max_new} tokens.]")


# ===========================================================================
# Endpoint 1: POST /generate
# ===========================================================================

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(req: GenerateRequest) -> GenerateResponse:
    """
    Generate text continuation from a prompt.

    Supports four decoding strategies:
    - **greedy**: deterministic, always picks most probable token
    - **temperature**: samples full vocab with temperature scaling
    - **top_k**: samples from top-k most probable tokens
    - **top_p**: nucleus sampling — adaptive vocabulary per step
    """
    t0 = time.perf_counter()

    if _model is None:
        # Stub mode
        gen_text = _mock_generate(req.prompt, req.max_new)
        elapsed  = time.perf_counter() - t0
        return GenerateResponse(
            prompt           = req.prompt,
            generated_text   = gen_text,
            full_text        = req.prompt + gen_text,
            tokens_generated = req.max_new,
            elapsed_seconds  = round(elapsed, 3),
            tokens_per_second= 0.0,
            strategy         = req.strategy,
            model_info       = {"mode": "stub"},
        )

    try:
        from generate import generate as gen_fn

        idx            = _encode_prompt(req.prompt)
        full_seq, meta = gen_fn(
            model       = _model,
            idx         = idx,
            max_new     = req.max_new,
            strategy    = req.strategy,
            temperature = req.temperature,
            top_k       = req.top_k,
            top_p       = req.top_p,
            device      = _device,
        )

        prompt_len     = idx.shape[1]
        generated_ids  = full_seq[0, prompt_len:].tolist()
        all_ids        = full_seq[0].tolist()
        generated_text = _decode_tokens(generated_ids)
        full_text      = _decode_tokens(all_ids)
        elapsed        = time.perf_counter() - t0

        # Parameter breakdown for model_info
        param_breakdown = {}
        if _model is not None:
            for name, module in _model.named_modules():
                own = sum(p.numel() for p in module.parameters(recurse=False))
                if own > 0:
                    param_breakdown[name or "root"] = own

        return GenerateResponse(
            prompt            = req.prompt,
            generated_text    = generated_text,
            full_text         = full_text,
            tokens_generated  = meta["tokens_generated"],
            elapsed_seconds   = round(elapsed, 3),
            tokens_per_second = meta["tokens_per_second"],
            strategy          = req.strategy,
            model_info        = {
                "d_model":   _config.d_model,
                "n_layers":  _config.n_layers,
                "n_heads":   _config.n_heads,
                "vocab_size":_config.vocab_size,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# Endpoint 1b: GET /generate/stream — SSE streaming
# ===========================================================================

@app.get("/generate/stream")
async def stream_generate(
    prompt:      str   = "To be or not to be",
    max_new:     int   = 100,
    strategy:    str   = "top_p",
    temperature: float = 0.8,
    top_k:       int   = 40,
    top_p:       float = 0.9,
):
    """
    Server-Sent Events streaming endpoint.
    The React dashboard uses this for the typewriter effect.

    Each SSE message is:  data: {"token": "...", "done": false}
    Final message is:     data: {"token": "", "done": true, "meta": {...}}
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        if _model is None:
            # Stub streaming
            stub_text = _mock_generate(prompt, max_new)
            for char in stub_text:
                yield f"data: {json.dumps({'token': char, 'done': False})}\n\n"
                await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'token': '', 'done': True, 'meta': {}})}\n\n"
            return

        try:
            from generate import generate as gen_fn

            idx         = _encode_prompt(prompt)
            t0          = time.perf_counter()
            gen, _      = gen_fn(
                model=_model, idx=idx, max_new=max_new,
                strategy=strategy, temperature=temperature,
                top_k=top_k, top_p=top_p, device=_device, stream=True,
            )

            n_generated = 0
            for tok_tensor in gen:
                tok_id   = tok_tensor[0, 0].item()
                tok_text = _decode_tokens([tok_id])
                n_generated += 1
                msg = json.dumps({"token": tok_text, "done": False})
                yield f"data: {msg}\n\n"
                await asyncio.sleep(0)   # yield control to event loop

            elapsed = time.perf_counter() - t0
            meta = {
                "tokens_generated":  n_generated,
                "elapsed_seconds":   round(elapsed, 3),
                "tokens_per_second": round(n_generated / max(elapsed, 1e-6), 1),
            }
            yield f"data: {json.dumps({'token': '', 'done': True, 'meta': meta})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ===========================================================================
# Endpoint 2: GET /model/info
# ===========================================================================

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """
    Return full architecture details and parameter breakdown.
    Used by the React dashboard's Architecture Explorer panel.
    """
    cfg = _config

    if _model is not None:
        total_p     = sum(p.numel() for p in _model.parameters())
        trainable_p = sum(p.numel() for p in _model.parameters() if p.requires_grad)
        param_breakdown = {}
        for name, module in _model.named_modules():
            own = sum(p.numel() for p in module.parameters(recurse=False))
            if own > 0:
                param_breakdown[name or "root"] = own
    else:
        # Stub mode estimates
        D, L, V = cfg.d_model, cfg.n_layers, cfg.vocab_size
        total_p     = V * D + L * 12 * D * D + D
        trainable_p = total_p
        param_breakdown = {
            "tok_emb":    V * D,
            "blocks":     L * 12 * D * D,
            "norm_final": D,
        }

    return ModelInfoResponse(
        checkpoint_path       = os.getenv("MODEL_PATH", "results/colab-checkpoints/exp1_step5000.pt"),
        device                = str(_device),
        vocab_size            = cfg.vocab_size,
        context_len           = cfg.context_len,
        d_model               = cfg.d_model,
        n_layers              = cfg.n_layers,
        n_heads               = cfg.n_heads,
        n_kv_heads            = getattr(cfg, "n_kv_heads", None),
        ffn_variant           = cfg.ffn_variant,
        pos_encoding          = getattr(cfg, "pos_encoding", "rope"),
        norm_type             = getattr(cfg, "norm_type", "rmsnorm"),
        weight_tying          = getattr(cfg, "weight_tying", True),
        total_parameters      = total_p,
        trainable_parameters  = trainable_p,
        parameter_breakdown   = param_breakdown,
    )


# ===========================================================================
# Endpoint 3: GET /experiments
# ===========================================================================

def _run_matches(run_name: str, token: str) -> bool:
    return token.lower() in (run_name or "").lower()


def _pick_run(runs: List[Dict[str, Any]], token: str) -> Optional[Dict[str, Any]]:
    matches = [r for r in runs if _run_matches(str(r.get("run_name", "")), token)]
    if not matches:
        return None
    # Prefer the run with best validation loss for this family.
    return min(matches, key=lambda r: float(r.get("val_loss", 1e9)))


def _format_experiment_from_run(
    run: Dict[str, Any],
    exp_id: str,
    label: str,
    params: int,
    key_finding: str,
    train_time_min: int,
) -> Dict[str, Any]:
    return {
        "id": exp_id,
        "label": label,
        "params": params,
        "final_val_loss": run.get("val_loss"),
        "final_perplexity": run.get("perplexity"),
        "train_time_min": train_time_min,
        "key_finding": key_finding,
        "loss_curve": [],
    }


def _build_normalized_experiment_payload(results_dir: Path) -> Dict[str, Any]:
    mlflow_path = results_dir / "mlflow_runs_summary.json"
    if not mlflow_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {mlflow_path}")

    runs = json.loads(mlflow_path.read_text(encoding="utf-8"))
    if not isinstance(runs, list):
        raise ValueError("results/mlflow_runs_summary.json must contain a JSON list")

    exp2 = _pick_run(runs, "exp2-larger")
    exp3_clip = _pick_run(runs, "exp3-baseline-with-clip")
    exp3_no_clip = _pick_run(runs, "exp3-baseline-no-clip")
    lr_1e3 = _pick_run(runs, "exp4-lr-1e3")
    lr_3e4 = _pick_run(runs, "exp4-lr-3e4")
    lr_1e4 = _pick_run(runs, "exp4-lr-1e4")

    excluded_tokens = ["exp2-", "exp3-", "exp4-"]
    baseline_candidates = [
        r for r in runs if not any(_run_matches(str(r.get("run_name", "")), t) for t in excluded_tokens)
    ]
    baseline = min(baseline_candidates, key=lambda r: float(r.get("val_loss", 1e9))) if baseline_candidates else None

    experiments = []
    summary_table = []

    if baseline:
        experiments.append(
            _format_experiment_from_run(
                baseline,
                exp_id="exp1_baseline",
                label="Baseline",
                params=790_000,
                key_finding="Stable baseline at this scale",
                train_time_min=3,
            )
        )
        summary_table.append(
            {
                "experiment": "Baseline",
                "params": "0.79M",
                "perplexity": baseline.get("perplexity"),
                "train_time": "~3 min",
                "key_finding": "Stable baseline at this scale",
            }
        )

    if exp2:
        experiments.append(
            _format_experiment_from_run(
                exp2,
                exp_id="exp2_larger",
                label="Larger model",
                params=4_720_000,
                key_finding="Larger model overfit under fixed budget",
                train_time_min=6,
            )
        )
        summary_table.append(
            {
                "experiment": "Larger",
                "params": "4.72M",
                "perplexity": exp2.get("perplexity"),
                "train_time": "~6 min",
                "key_finding": "Overfit under fixed budget",
            }
        )

    if exp3_clip:
        experiments.append(
            _format_experiment_from_run(
                exp3_clip,
                exp_id="exp3_clip",
                label="With grad clip",
                params=790_000,
                key_finding="Stable baseline with clipping",
                train_time_min=3,
            )
        )

    if exp3_no_clip:
        experiments.append(
            _format_experiment_from_run(
                exp3_no_clip,
                exp_id="exp3_no_clip",
                label="No grad clip",
                params=790_000,
                key_finding="Similar to clipped run in this setup",
                train_time_min=3,
            )
        )
        summary_table.append(
            {
                "experiment": "No Clip",
                "params": "0.79M",
                "perplexity": exp3_no_clip.get("perplexity"),
                "train_time": "~3 min",
                "key_finding": "Similar to clipped run",
            }
        )

    lr_curves: Dict[str, List[Dict[str, Any]]] = {}
    if lr_1e3:
        lr_curves["1e-3"] = [{"step": 5000, "val_loss": lr_1e3.get("val_loss")}]
    if lr_3e4:
        lr_curves["3e-4"] = [{"step": 5000, "val_loss": lr_3e4.get("val_loss")}]
    if lr_1e4:
        lr_curves["1e-4"] = [{"step": 5000, "val_loss": lr_1e4.get("val_loss")}]

    if lr_curves:
        experiments.append(
            {
                "id": "exp4_lr_sweep",
                "label": "LR Sweep",
                "params": 790_000,
                "key_finding": "3e-4 best among tested LRs",
                "lr_curves": lr_curves,
            }
        )
    if lr_3e4:
        summary_table.append(
            {
                "experiment": "Best LR",
                "params": "0.79M",
                "perplexity": lr_3e4.get("perplexity"),
                "train_time": "~3 min",
                "key_finding": "3e-4 best among tested LRs",
            }
        )

    lora = {
        "base_params": 1_827_968,
        "lora_params": 16_384,
        "lora_pct": 0.8963,
        "reduction_factor": 111.57,
        "adapter_kb": 64.0,
        "rank": 4,
        "alpha": 4.0,
        "target_modules": ["W_q", "W_k", "W_v", "W_o"],
        "fine_tune_steps": 1000,
        "val_loss_curve": [
            {"step": 800, "val_loss": 5.4298, "ppl": 228.09},
            {"step": 900, "val_loss": 5.4226, "ppl": 226.47},
            {"step": 1000, "val_loss": 5.3922, "ppl": 219.68},
        ],
    }
    summary_table.append(
        {
            "experiment": "LoRA FT",
            "params": "16K",
            "perplexity": 219.68,
            "train_time": "~5 min",
            "key_finding": "0.9% params, domain shift",
        }
    )

    eval_payload = None
    eval_path = results_dir / "eval" / "perplexity_eval_baseline_vs_lora.json"
    if eval_path.exists():
        try:
            loaded = json.loads(eval_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                eval_payload = loaded
        except Exception:
            eval_payload = None

    rank_sweep_payload = None
    rank_sweep_path = results_dir / "eval" / "lora_rank_sweep.json"
    if rank_sweep_path.exists():
        try:
            loaded = json.loads(rank_sweep_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                rank_sweep_payload = loaded
        except Exception:
            rank_sweep_payload = None

    prompt_benchmarks: Dict[str, Any] = {}
    prompt_benchmark_files = {
        "rank4_default": results_dir / "eval" / "prompt_benchmark_summary.json",
        "rank8_t06_k20": results_dir / "eval" / "prompt_benchmark_summary_rank8_t06_k20.json",
        "rank16_t06_k20": results_dir / "eval" / "prompt_benchmark_summary_rank16_t06_k20.json",
    }
    for key, path in prompt_benchmark_files.items():
        if not path.exists():
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                prompt_benchmarks[key] = loaded
        except Exception:
            continue

    return {
        "experiments": experiments,
        "lora": lora,
        "summary_table": summary_table,
        "evaluation": eval_payload,
        "lora_rank_sweep": rank_sweep_payload,
        "prompt_benchmarks": prompt_benchmarks,
    }


@app.get("/experiments")
async def get_experiments() -> Dict[str, Any]:
    """
    Return all Phase 4 experiment results and Phase 5 LoRA data as JSON.
    Used by the React dashboard's Experiment Results panel.

    Builds one normalized payload from committed experiment artifacts.
    """
    results_dir = Path(os.getenv("RESULTS_DIR", "results"))
    try:
        payload = _build_normalized_experiment_payload(results_dir)
        return {"source": "normalized", "data": payload}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build experiments payload: {e}")


@app.get("/inference/compare")
async def inference_compare() -> Dict[str, Any]:
    """
    Phase 6 Step 4 comparison endpoint:
    - loads external LLM inference metrics from results/inference_comparison.json
    - runs a live NanoGPT benchmark with same prompt/settings
    """
    cmp_path = Path(os.getenv("RESULTS_DIR", "results")) / "inference_comparison.json"
    if not cmp_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing comparison file: {cmp_path}")

    payload = json.loads(cmp_path.read_text(encoding="utf-8"))
    prompt = payload.get("prompt", "To be, or not to be,")
    settings = payload.get("settings", {})
    max_new = int(settings.get("max_new_tokens", 64))
    temperature = float(settings.get("temperature", 0.8))
    top_k = int(settings.get("top_k", 40))
    top_p = float(settings.get("top_p", 0.9))
    strategy = str(settings.get("strategy", "top_p"))

    if _model is None:
        return {
            "prompt": prompt,
            "settings": settings,
            "nanogpt": {"error": "Model not loaded"},
            "llm_inference_project": payload.get("llm_inference_project", {}),
        }

    from generate import generate as gen_fn
    idx = _encode_prompt(prompt)
    t0 = time.perf_counter()
    full_seq, meta = gen_fn(
        model=_model,
        idx=idx,
        max_new=max_new,
        strategy=strategy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=_device,
    )
    elapsed = time.perf_counter() - t0
    prompt_len = idx.shape[1]
    generated_ids = full_seq[0, prompt_len:].tolist()
    generated_text = _decode_tokens(generated_ids)

    nanogpt = {
        "model_name": "NanoGPT (from-scratch)",
        "device": str(_device),
        "elapsed_seconds": round(elapsed, 3),
        "tokens_generated": meta.get("tokens_generated", len(generated_ids)),
        "tokens_per_second": meta.get("tokens_per_second", round(len(generated_ids) / max(elapsed, 1e-6), 1)),
        "output_sample": generated_text[:400],
        "source": "/generate internal benchmark",
    }

    return {
        "prompt": prompt,
        "settings": settings,
        "nanogpt": nanogpt,
        "llm_inference_project": payload.get("llm_inference_project", {}),
    }


# ===========================================================================
# Health check
# ===========================================================================

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status":       "ok",
        "model_loaded": _model is not None,
        "device":       str(_device),
        "startup_mode": _startup_mode,
        "degraded": _model is None,
        "timestamp":    time.time(),
    }


# ===========================================================================
# Dev entry point
# ===========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)