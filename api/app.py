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
    global _model, _tokenizer, _config, _device

    device_str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    _device    = torch.device(device_str)

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

# Hardcoded Phase 4 + Phase 5 results as fallback when no JSON files exist
_HARDCODED_RESULTS = {
    "experiments": [
        {
            "id":          "exp1_baseline",
            "label":       "Baseline (1.2M)",
            "params":      1_827_968,
            "n_layers":    4,
            "n_heads":     4,
            "d_model":     128,
            "lr":          3e-4,
            "grad_clip":   True,
            "max_steps":   5000,
            "final_perplexity": 28.4,
            "train_time_min":   18,
            "key_finding": "Stable training, good baseline",
            "loss_curve": [
                {"step": 0,    "train_loss": 9.12, "val_loss": 9.10},
                {"step": 500,  "train_loss": 6.43, "val_loss": 6.51},
                {"step": 1000, "train_loss": 5.21, "val_loss": 5.35},
                {"step": 1500, "train_loss": 4.58, "val_loss": 4.74},
                {"step": 2000, "train_loss": 4.12, "val_loss": 4.30},
                {"step": 2500, "train_loss": 3.81, "val_loss": 4.01},
                {"step": 3000, "train_loss": 3.58, "val_loss": 3.78},
                {"step": 3500, "train_loss": 3.41, "val_loss": 3.62},
                {"step": 4000, "train_loss": 3.28, "val_loss": 3.50},
                {"step": 4500, "train_loss": 3.18, "val_loss": 3.41},
                {"step": 5000, "train_loss": 3.10, "val_loss": 3.34},
            ],
        },
        {
            "id":          "exp2_larger",
            "label":       "Larger (6M)",
            "params":      6_000_000,
            "n_layers":    6,
            "n_heads":     6,
            "d_model":     256,
            "lr":          3e-4,
            "grad_clip":   True,
            "max_steps":   5000,
            "final_perplexity": 22.1,
            "train_time_min":   32,
            "key_finding": "Better perplexity with more capacity",
            "loss_curve": [
                {"step": 0,    "train_loss": 9.15, "val_loss": 9.11},
                {"step": 500,  "train_loss": 5.98, "val_loss": 6.09},
                {"step": 1000, "train_loss": 4.72, "val_loss": 4.89},
                {"step": 1500, "train_loss": 4.05, "val_loss": 4.22},
                {"step": 2000, "train_loss": 3.61, "val_loss": 3.81},
                {"step": 2500, "train_loss": 3.31, "val_loss": 3.52},
                {"step": 3000, "train_loss": 3.08, "val_loss": 3.31},
                {"step": 3500, "train_loss": 2.90, "val_loss": 3.15},
                {"step": 4000, "train_loss": 2.76, "val_loss": 3.02},
                {"step": 4500, "train_loss": 2.65, "val_loss": 2.93},
                {"step": 5000, "train_loss": 2.56, "val_loss": 2.85},
            ],
        },
        {
            "id":          "exp3_no_clip",
            "label":       "No Grad Clip",
            "params":      1_827_968,
            "n_layers":    4,
            "n_heads":     4,
            "d_model":     128,
            "lr":          3e-4,
            "grad_clip":   False,
            "max_steps":   5000,
            "final_perplexity": None,
            "train_time_min":   None,
            "key_finding": "Gradient explosion — training diverged",
            "loss_curve": [
                {"step": 0,    "train_loss": 9.12, "val_loss": 9.10},
                {"step": 500,  "train_loss": 6.51, "val_loss": 6.62},
                {"step": 1000, "train_loss": 5.34, "val_loss": 5.48},
                {"step": 1500, "train_loss": 7.82, "val_loss": 8.10},
                {"step": 2000, "train_loss": 18.4, "val_loss": 19.2},
                {"step": 2500, "train_loss": None, "val_loss": None},
            ],
        },
        {
            "id":          "exp4_lr_sweep",
            "label":       "LR Sweep",
            "params":      1_827_968,
            "n_layers":    4,
            "n_heads":     4,
            "d_model":     128,
            "grad_clip":   True,
            "max_steps":   5000,
            "key_finding": "3e-4 optimal; 1e-3 diverges; 1e-4 too slow",
            "lr_curves": {
                "1e-3": [
                    {"step": 0,    "val_loss": 9.12},
                    {"step": 500,  "val_loss": 6.21},
                    {"step": 1000, "val_loss": 8.43},
                    {"step": 1500, "val_loss": 14.2},
                    {"step": 2000, "val_loss": None},
                ],
                "3e-4": [
                    {"step": 0,    "val_loss": 9.10},
                    {"step": 500,  "val_loss": 6.51},
                    {"step": 1000, "val_loss": 5.35},
                    {"step": 2000, "val_loss": 4.30},
                    {"step": 3000, "val_loss": 3.78},
                    {"step": 4000, "val_loss": 3.50},
                    {"step": 5000, "val_loss": 3.34},
                ],
                "1e-4": [
                    {"step": 0,    "val_loss": 9.11},
                    {"step": 500,  "val_loss": 7.82},
                    {"step": 1000, "val_loss": 6.94},
                    {"step": 2000, "val_loss": 5.91},
                    {"step": 3000, "val_loss": 5.14},
                    {"step": 4000, "val_loss": 4.62},
                    {"step": 5000, "val_loss": 4.24},
                ],
            },
        },
    ],
    "lora": {
        "base_params":  1_827_968,
        "lora_params":  16_384,
        "lora_pct":     0.8963,
        "reduction_factor": 111.57,
        "adapter_kb":   64.0,
        "rank":         4,
        "alpha":        4.0,
        "target_modules": ["W_q", "W_k", "W_v", "W_o"],
        "fine_tune_steps": 1000,
        "val_loss_curve": [
            {"step": 800,  "val_loss": 5.4298, "ppl": 228.09},
            {"step": 900,  "val_loss": 5.4226, "ppl": 226.47},
            {"step": 1000, "val_loss": 5.3922, "ppl": 219.68},
        ],
    },
    "summary_table": [
        {"experiment": "Baseline",   "params": "1.2M", "perplexity": 28.4,    "train_time": "18 min", "key_finding": "Stable training"},
        {"experiment": "Larger",     "params": "6M",   "perplexity": 22.1,    "train_time": "32 min", "key_finding": "Better perplexity"},
        {"experiment": "No Clip",    "params": "1.2M", "perplexity": "∞",     "train_time": "stopped","key_finding": "Exploding gradients"},
        {"experiment": "Best LR",    "params": "1.2M", "perplexity": 26.8,    "train_time": "18 min", "key_finding": "3e-4 optimal"},
        {"experiment": "LoRA FT",    "params": "16K",  "perplexity": 219.68,  "train_time": "~5 min", "key_finding": "0.9% params, domain shift"},
    ],
}


@app.get("/experiments")
async def get_experiments() -> Dict[str, Any]:
    """
    Return all Phase 4 experiment results and Phase 5 LoRA data as JSON.
    Used by the React dashboard's Experiment Results panel.

    Tries to load from results/ directory first; falls back to hardcoded data.
    """
    results_dir = Path(os.getenv("RESULTS_DIR", "results"))

    # Try to load real experiment JSONs from disk
    loaded = {}
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                loaded[json_file.stem] = json.load(f)
        except Exception:
            pass

    if loaded:
        return {"source": "disk", "data": loaded}

    # Fall back to hardcoded Phase 4/5 results
    return {"source": "hardcoded", "data": _HARDCODED_RESULTS}


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
        "timestamp":    time.time(),
    }


# ===========================================================================
# Dev entry point
# ===========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)