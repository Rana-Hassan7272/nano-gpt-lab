

# NanoGPT # NanoGPT Lab

End-to-end GPT-style project built from scratch with PyTorch, including custom tokenizer, transformer training, LoRA fine-tuning, inference API, dashboard, Docker setup, and cloud deployment.

## Live Deployment

- Frontend (Vercel): [https://nano-gpt-lab.vercel.app/](https://nano-gpt-lab.vercel.app/)
- Backend API (Render): [https://nano-gpt-lab.onrender.com](https://nano-gpt-lab.onrender.com)
- API Docs (Swagger): [https://nano-gpt-lab.onrender.com/docs](https://nano-gpt-lab.onrender.com/docs)
- Health Check: [https://nano-gpt-lab.onrender.com/health](https://nano-gpt-lab.onrender.com/health)
# README SOURCE PACK FOR CLAUDE
# Project: NanoGPT Lab
# Author intent: Build a complete GPT-style model pipeline from scratch, train it, experiment systematically, add LoRA fine-tuning, deploy API + dashboard, and present professionally.
# Please transform this source pack into a polished README with clear design, strong narrative, and clean formatting.

---

## 1) PROJECT IDENTITY

Project Name:
NanoGPT Lab

Tagline:
Build, train, fine-tune, serve, and deploy a GPT-style language model from scratch using pure PyTorch.

Short Elevator Pitch:
NanoGPT Lab is a full-stack LLM engineering project that starts from first-principles transformer implementation and ends with production-style deployment. It includes custom tokenizer, transformer architecture, training infrastructure, controlled experiments, LoRA fine-tuning, FastAPI inference service, React dashboard, Dockerization, and cloud deployment.

Portfolio Positioning:
This project demonstrates end-to-end AI engineering capability:
- deep model understanding (attention, FFN, normalization, positional methods),
- practical training infra (scheduling, clipping, AMP, MLflow),
- evaluation and experimentation mindset,
- parameter-efficient fine-tuning (LoRA),
- system integration (API + frontend),
- deployment and product packaging.

---

## 2) CORE GOALS

Primary Goal:
Implement a GPT-style language model from scratch without relying on high-level model libraries for core architecture.

Secondary Goals:
- Build reliable data pipeline and reproducible configs.
- Train and compare multiple experiment settings.
- Add LoRA fine-tuning for parameter efficiency.
- Serve model through an API.
- Expose model behavior and experiment outputs in a dashboard.
- Deploy backend + frontend in cloud.
- Document professionally for recruiters and engineers.

---

## 3) PHASE-WISE DELIVERY SUMMARY

### Phase 1: Architecture from Scratch
Delivered:
- Custom BPE tokenizer implementation.
- Multi-head causal self-attention implementation.
- Feedforward network implementation.
- Transformer block composition (residual + norm + attention + FFN).
- Full NanoGPT model assembly.
- CPU sanity tests.

### Phase 2: Data Pipeline
Delivered:
- Raw text preparation script.
- BPE tokenizer training and persistence.
- Corpus tokenization.
- Train/validation split and binary serialization.
- PyTorch Dataset/DataLoader using memmap binaries.
- Decode sanity checks.

### Phase 3: Training Infrastructure
Delivered:
- YAML config-based training control.
- Trainer loop:
  - forward pass,
  - CE loss,
  - backward pass,
  - gradient clipping,
  - optimizer step,
  - LR scheduler step,
  - validation evaluation,
  - checkpoint save.
- Warmup + cosine scheduler.
- Mixed precision support.
- MLflow metric logging.

### Phase 4: Training Experiments
Delivered:
- Multiple config files for planned ablations:
  - baseline,
  - larger model,
  - no-clip vs clip,
  - LR sweep.
- Dashboard/API supports rendering experiment outputs.
- Result summary artifacts available in results JSON workflows.

### Phase 5: LoRA Fine-tuning
Delivered:
- LoRA module implementation (`LoRALinear`, adapter inject/merge/unmerge/save/load).
- LoRA trainer script.
- Parameter-efficiency tracking.
- Fine-tuning pipeline integration and reporting.
- Mathematical LoRA explanation document.

### Phase 6: API + Dashboard
Delivered:
- Inference engine with:
  - greedy,
  - temperature,
  - top-k,
  - top-p.
- FastAPI backend with generation + model info + experiments endpoints.
- Streaming generation endpoint (SSE).
- React dashboard with:
  - generation playground,
  - experiment visualizations,
  - architecture explorer,
  - inference comparison panel.
- Environment-variable based frontend API URL support.

### Phase 7: Polish + Deploy + Document
Delivered so far:
- Docker setup:
  - API Dockerfile,
  - dashboard Dockerfile,
  - compose stack for API + dashboard + MLflow.
- Cloud deployment:
  - Backend deployed on Render.
  - Frontend deployed on Vercel.
- Deployment troubleshooting done:
  - JSON ignore issue fixed,
  - Vercel output config alignment fixed,
  - runtime env setup stabilized.

---

## 4) REPOSITORY STRUCTURE (ACTUAL)

Use this in README as project tree summary:

- `model/`
  - `tokenizer.py`
  - `attention.py`
  - `feedforward.py`
  - `transformer_block.py`
  - `nanogpt.py`
  - `lora.py`
- `data/`
  - `prepare.py`
  - `dataset.py`
  - `tokenizer.json` (generated/persisted tokenizer)
- `training/`
  - `trainer.py`
  - `scheduler.py`
  - `lora_trainer.py`
- `inference/`
  - `generate.py`
- `api/`
  - `app.py`
- `dashboard/`
  - `dashboard.jsx`
  - `main.jsx`
  - `vite.config.js`
  - `package.json`
- `configs/`
  - `train_config.yaml`
  - `exp2_larger.yaml`
  - `exp3_clip.yaml`
  - `exp3_no_clip.yaml`
  - `exp4_lr_1e3.yaml`
  - `exp4_lr_3e4.yaml`
  - `exp4_lr_1e4.yaml`
- `experiments/`
  - test scripts and export utilities
  - LoRA explanation markdown
- `results/`
  - checkpoints
  - experiment summary JSON(s)
  - inference comparison JSON
- Infra files:
  - `Dockerfile.api`
  - `dashboard/Dockerfile`
  - `docker-compose.yml`
  - `.dockerignore`
  - `vercel.json`
  - `.gitignore`
  - `requirements.txt`

---

## 5) TECHNICAL ARCHITECTURE OVERVIEW

### Model Layer
- Decoder-only transformer.
- Causal masking ensures autoregressive behavior.
- Token embeddings + positional strategy support.
- Stacked transformer blocks.
- Final normalization + LM head.
- Weight tying supported (token embedding and head).

### Attention Layer
Implemented features include:
- Standard multi-head attention.
- Causal mask.
- Optional RoPE.
- Optional ALiBi.
- Optional grouped-query style behavior.
- Optional flash-attention path when available.
- KV-cache support for inference acceleration.

### Feedforward Layer
- Standard FFN with expansion and activation options.
- Variants include modern gated alternatives.
- Regularization via dropout.

### Block/Stack Layer
- Residual paths.
- Normalization options.
- Configurable architecture toggles.
- Stack abstraction for repeated blocks.

### Tokenization Layer
- Byte-level BPE from scratch.
- Train/encode/decode/save/load.
- Merge learning logic from corpus statistics.

### Data Layer
- Corpus -> token IDs -> train.bin/val.bin.
- Memory-mapped dataset for efficient training reads.
- Context window chunking with shifted targets.

### Training Layer
- Config-driven training.
- AdamW optimizer grouping.
- Warmup + cosine decay.
- Gradient clipping and grad norm logging.
- AMP support.
- Validation estimation loop.
- Checkpointing.
- MLflow instrumentation.

### Fine-tuning Layer
- LoRA adapter insertion over target linear layers.
- Freeze base parameters, train adapters only.
- Adapter serialization and reload.
- Merge adapters into base weights for inference.

### Inference Layer
- Unified generator API.
- Strategy-specific decoding controls.
- Streaming-friendly token generation.
- Metrics (tok/s, elapsed, token count).

### Serving Layer
- FastAPI endpoints for generation and metadata.
- Streaming endpoint via Server-Sent Events.
- CORS-enabled for frontend integration.

### Frontend Layer
- React + Vite.
- Generation controls and stream output.
- Experiment charts and summaries.
- Architecture visualization panel.
- Inference comparison panel.

### Deployment Layer
- Local orchestration with Docker Compose.
- Cloud backend on Render.
- Cloud frontend on Vercel.
- Env-driven API target wiring for frontend.

---

## 6) API DESIGN (DOCUMENT THESE ENDPOINTS)

Main endpoints:
- `POST /generate`
- `GET /generate/stream`
- `GET /model/info`
- `GET /experiments`
- `GET /inference/compare`
- `GET /health`

Recommended README endpoint table columns:
- Endpoint
- Method
- Purpose
- Key Parameters
- Response highlights

Generation request fields to document:
- `prompt`
- `max_new`
- `strategy` (`greedy`, `temperature`, `top_k`, `top_p`)
- `temperature`
- `top_k`
- `top_p`

---

## 7) DASHBOARD CAPABILITIES TO DOCUMENT

### Generation Playground
- Prompt input.
- Decoding strategy toggles.
- Temperature/top-k/top-p controls.
- Streaming output view.
- Throughput stats.

### Experiment Results
- Loss curves (train vs val).
- LR sweep behavior.
- Summary tables.
- LoRA efficiency panel.

### Architecture Explorer
- Parameter distribution chart.
- Layer tree view.
- Data flow sequence.

### Inference Comparison
- NanoGPT vs external inference metrics loaded from result JSON.
- Relative latency/throughput context.

---

## 8) TRAINING & EXPERIMENT CONFIGURATION

Document config philosophy:
- One YAML per experiment.
- Reproducibility via explicit hyperparameters.
- Controlled sweeps for specific variables.

Key fields to show in README:
- model:
  - layers
  - heads
  - d_model
  - context length
  - dropout
- training:
  - batch size
  - learning rate
  - max steps
  - warmup
  - grad clip
  - eval/checkpoint intervals
- data split
- mlflow tracking URI/experiment name

---

## 9) ACHIEVEMENTS TO HIGHLIGHT (HR + ENGINEERING)

### Engineering Achievements
- End-to-end implementation from model internals to deployment.
- Multiple training and decoding strategies exposed to product UI.
- Structured experiment system, not ad-hoc runs.
- LoRA integration with adapter lifecycle management.
- Production-style API + dashboard integration.
- Containerized local stack and cloud deployment.

### Learning/Research Achievements
- Practical understanding of:
  - causal attention math,
  - normalization and residual design,
  - optimizer scheduling,
  - gradient stability,
  - decoding behavior trade-offs,
  - parameter-efficient fine-tuning.

### Product Achievements
- Live deploy with interactive streaming generation.
- Visual experiment storytelling in dashboard.
- Architecture interpretability view for non-ML stakeholders.

### Portfolio Signal
- Demonstrates ability to own full lifecycle:
  ideation -> implementation -> training -> evaluation -> serving -> deployment -> documentation.

---

## 10) SKILLS SHOWCASED (MAKE THIS EXPLICIT)

### ML / DL
- Transformer internals
- Attention mechanisms
- Tokenization (BPE)
- Training loops in PyTorch
- LR scheduling
- Mixed precision
- Model checkpointing
- LoRA fine-tuning
- Text generation decoding strategies

### Data & Experimentation
- Data preprocessing pipeline
- Binary dataset handling
- Ablation studies
- Metric tracking with MLflow
- Reproducible config management

### Backend / Systems
- FastAPI
- Async streaming (SSE)
- API design
- Runtime configuration via env vars
- Dockerization
- Cloud deployment on Render

### Frontend / Visualization
- React + Vite
- Recharts visualization
- Real-time generation UX
- Cross-service integration with backend

### DevOps / Delivery
- Docker Compose orchestration
- Multi-service local stack
- Frontend deployment on Vercel
- Production troubleshooting across CI/CD and runtime config

---

## 11) DEPLOYMENT DETAILS TO INCLUDE

### Local deployment (one command)
`docker compose up --build`

Expose:
- API: `http://localhost:8000`
- Dashboard: `http://localhost:5173`
- MLflow: `http://localhost:5000`

### Cloud deployment (current)
Backend:
- Render web service
- Start command using uvicorn or Docker CMD
- CPU mode for compatibility

Frontend:
- Vercel static deployment from `dashboard`
- Env var:
  - `VITE_API_URL=<backend-url>`

### Deployment lessons learned (include as a "Troubleshooting" section)
- Broad `*.json` in `.gitignore` can break frontend deploy by excluding package metadata.
- Vercel root directory vs output directory mismatch can produce blank screen.
- Serving raw `.jsx` in prod causes MIME type error (`text/jsx`).
- Need explicit build/output settings and/or `vercel.json` alignment.

---

## 12) RESULTS SECTION GUIDANCE

Important:
Use real metrics from your artifacts where available.
If exact numbers change, keep structure but update values.

### Suggested result blocks:
1) Baseline model performance
2) Larger model comparison
3) Clipping vs no clipping
4) LR sweep outcomes
5) LoRA parameter efficiency outcomes
6) Inference throughput/latency notes (local vs cloud)

### Suggested narrative:
- What changed
- Why it matters
- What decision it informed

### Suggested visuals:
- Train/val loss chart
- LR sweep chart
- Experiment comparison table
- LoRA parameter reduction chart
- Architecture parameter distribution chart
- Inference throughput bar comparison

---

## 13) TABLES TO INCLUDE IN README

### Table A: Feature Matrix
Columns:
- Module
- Implemented
- Description
- Status

Rows:
- Tokenizer
- Attention
- FFN
- Transformer Stack
- Training Loop
- Scheduler
- AMP
- MLflow
- LoRA
- API
- Streaming
- Dashboard
- Docker
- Cloud Deploy

### Table B: Experiment Summary
Columns:
- Experiment
- Model Size
- Key Config
- Final Validation Loss / Perplexity
- Time
- Finding

### Table C: Decoding Strategy Comparison
Columns:
- Strategy
- Deterministic?
- Diversity
- Typical Use Case
- Parameters

Rows:
- Greedy
- Temperature
- Top-k
- Top-p

### Table D: Deployment Matrix
Columns:
- Layer
- Platform
- URL
- Notes

Rows:
- Backend
- Frontend
- MLflow (local)

---

## 14) IMAGE / DIAGRAM PLAN (HIGH PRIORITY)

Please include placeholders in README for these images:

1) `docs/images/architecture-overview.png`
Caption:
"End-to-end architecture: tokenizer -> model -> trainer -> API -> dashboard"

2) `docs/images/training-loss-curves.png`
Caption:
"Training and validation loss over steps"

3) `docs/images/lr-sweep.png`
Caption:
"Learning rate sweep and convergence behavior"

4) `docs/images/lora-efficiency.png`
Caption:
"LoRA trainable parameter efficiency vs full fine-tuning"

5) `docs/images/dashboard-generation.png`
Caption:
"Live text generation playground with decoding controls"

6) `docs/images/dashboard-experiments.png`
Caption:
"Experiment analysis dashboard"

7) `docs/images/dashboard-architecture.png`
Caption:
"Architecture explorer with parameter distribution"

8) `docs/images/api-swagger.png`
Caption:
"FastAPI Swagger documentation"

9) `docs/images/deployment-overview.png`
Caption:
"Deployment topology: Vercel frontend + Render API"

10) `docs/images/inference-compare.png`
Caption:
"Inference comparison panel"

---

## 15) COMMANDS SECTION TO INCLUDE

### Environment setup
- Python env creation
- dependency install
- frontend install

### Data preparation
- run `data/prepare.py` command with args

### Training
- run baseline config
- run experiment configs

### LoRA fine-tuning
- command for `training/lora_trainer.py`

### API run
- command for uvicorn backend run

### Dashboard run
- command for vite dev server

### Docker run
- `docker compose up --build`

### Optional quality checks
- run experiment test scripts from `experiments/`

---

## 16) REPRODUCIBILITY SECTION CONTENT

Include:
- seed control strategy (if implemented / planned)
- exact config file references
- checkpoint naming conventions
- where logs are stored
- how to re-run experiments from scratch
- expected runtime notes (local CPU vs GPU/cloud)

Mention:
- cloud performance differs due to weaker CPU and network latency.
- streaming in deployed mode incurs per-token transport overhead.

---

## 17) LIMITATIONS & HONEST DISCLOSURE

Please include candid limitations:
- Tiny dataset limits general-purpose capabilities.
- Model behaves strongly in Shakespeare-like style domain.
- Cloud free-tier latency is slower than local machine.
- Inference speed is constrained by CPU deployment.
- Experiment metrics quality depends on compute budget and run length.

This honesty improves credibility in interviews.

---

## 18) ROADMAP / NEXT STEPS

Suggested roadmap entries:
- Quantization for faster CPU inference.
- Batch generation endpoint.
- Redis caching for repeated prompts.
- Better experiment artifact persistence and report automation.
- Authentication and rate limiting for production API.
- Add downloadable run reports from dashboard.
- Add CI workflow for lint/tests/build checks.
- Add unit + integration tests for API and inference.
- Introduce model registry and checkpoint versioning policy.
- Add proper prompt templates and evaluation harness.

---

## 19) VIDEO DEMO SECTION

Include:
- short walkthrough script
- what each tab proves
- prompt examples used
- expected behavior notes
- deployment proof links

Prompt examples to include:
- Shakespeare continuation
- Dialogue role-play continuation
- Same prompt with greedy vs top-p comparison

---

## 20) RECRUITER-FRIENDLY OUTCOME SUMMARY

Use a concise bullet list:
- Built GPT-style stack from tokenizer to deployment.
- Implemented and compared decoding and training stabilization methods.
- Added parameter-efficient LoRA fine-tuning.
- Exposed model through API and interactive frontend.
- Deployed full product and documented engineering decisions.

---

## 21) SUGGESTED README TOP-LEVEL TABLE OF CONTENTS

1. Project Overview
2. Why This Project Matters
3. Live Demo Links
4. System Architecture
5. Repository Structure
6. Core Components
   - Tokenizer
   - Model
   - Training
   - LoRA
   - Inference
   - API
   - Dashboard
7. Experiments and Results
8. Deployment
9. Quickstart
10. Detailed Usage
11. API Reference
12. Dashboard Guide
13. Reproducibility
14. Limitations
15. Roadmap
16. Skills Demonstrated
17. Acknowledgements

---

## 22) LIVE LINKS PLACEHOLDERS

Please include placeholders so I can replace later:

- Frontend URL: `<VERCEL_FRONTEND_URL>`
- Backend URL: `<RENDER_BACKEND_URL>`
- API Docs: `<RENDER_BACKEND_URL>/docs`
- Health: `<RENDER_BACKEND_URL>/health`
- Repository URL: `<GITHUB_REPO_URL>`
- Demo video URL: `<YOUTUBE_OR_DRIVE_LINK>`

---

## 23) RESULTS FILES TO REFERENCE IN README

Potential files to mention in "Artifacts":
- `results/checkpoints/step_500.pt`
- `results/checkpoints/step_1000.pt`
- `results/mlflow_runs_summary.json`
- `results/inference_comparison.json`

Mention that binary checkpoints are large and usually not included in lightweight distribution.

---

## 24) IMPORTANT IMPLEMENTATION NOTES TO INCLUDE

- Frontend reads backend URL from `VITE_API_URL`.
- API supports fallback behavior if model not loaded.
- Streaming endpoint uses SSE and emits incremental token updates.
- Deployment setup includes Docker and cloud-specific env var handling.
- `.gitignore` must not broadly ignore `*.json` to avoid breaking frontend deploy metadata.

---

## 25) DOCUMENTATION TONE & STYLE REQUEST TO CLAUDE

Please produce README in:
- polished professional engineering style,
- concise but technically deep,
- with clear sectioning, tables, and visuals placeholders,
- audience: recruiters, hiring managers, ML engineers, backend engineers.

Prioritize:
- clarity,
- reproducibility,
- measurable outcomes,
- end-to-end ownership narrative.

---

## 26) FINAL SUMMARY PARAGRAPH TO INCLUDE

NanoGPT Lab is not just a model script; it is a complete AI system project that demonstrates first-principles model engineering, disciplined experimentation, efficient fine-tuning, product integration, and real deployment. It showcases the ability to bridge deep learning theory and production delivery in one coherent portfolio-grade implementation.

---

# END OF SOURCE PACK
## Local Docker + MLOps Stack

Run everything locally with one command:

```bash
docker compose up --build
```

Services:
- API: `http://localhost:8000`
- Dashboard: `http://localhost:5173`
- MLflow: `http://localhost:5000`

## Deployment Architecture

- Local: Docker Compose (`api` + `dashboard` + `mlflow`)
- Cloud Backend: Render (FastAPI)
- Cloud Frontend: Vercel (React/Vite dashboard)
