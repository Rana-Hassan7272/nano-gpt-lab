# Project 5 — NanoGPT Lab
## Total time: 8 weeks

---

## Phase 1 — Architecture from Scratch
### Week 1-2

**Goal:** Every transformer component written by hand in pure PyTorch. No HuggingFace. No shortcuts.

**Step 1 — Project structure**
```
nanogpt-lab/
├── model/
│   ├── attention.py
│   ├── feedforward.py
│   ├── transformer_block.py
│   ├── nanogpt.py
│   └── tokenizer.py
├── training/
│   ├── trainer.py
│   ├── optimizer.py
│   └── scheduler.py
├── data/
│   ├── dataset.py
│   └── prepare.py
├── experiments/
├── inference/
├── api/
├── dashboard/
├── configs/
└── results/
Step 2 — Tokenizer first
Write model/tokenizer.py implementing basic Byte-Pair Encoding from scratch. This is the most underrated component. Understanding tokenization deeply separates you from 90% of candidates. It does not need to be perfect — a clean simple BPE that works is enough.
Step 3 — Multi-head attention
Write model/attention.py implementing exactly:
pythonclass MultiHeadAttention:
    # Q, K, V projection matrices — write these yourself
    # Scaled dot-product attention — implement the equation
    # Causal masking — so model cannot see future tokens
    # Multi-head splitting and concatenation
    # Output projection
Do not copy paste. Type every line. Understand every line. This is the component you will explain in every interview.
Step 4 — Feedforward block
Write model/feedforward.py:
pythonclass FeedForward:
    # Linear layer 1: d_model → 4 * d_model
    # GELU activation — implement it, understand why not ReLU
    # Linear layer 2: 4 * d_model → d_model
    # Dropout
Step 5 — Transformer block
Write model/transformer_block.py combining:

Layer norm before attention (pre-norm, not post-norm — know the difference)
Multi-head attention
Residual connection
Layer norm before feedforward
Feedforward block
Residual connection

Step 6 — Full NanoGPT model
Write model/nanogpt.py:

Token embedding table
Positional embedding
Stack of N transformer blocks
Final layer norm
Language model head (linear projection to vocabulary)

Step 7 — Verify architecture on CPU
Before any GPU training, run your model on your laptop with fake tiny data. Make sure shapes are correct. Make sure forward pass runs. Make sure no dimension errors. Fix everything locally. This saves hours of wasted Colab sessions.
Deliverable at end of Phase 1:
Complete transformer architecture running on CPU with correct shapes verified.

Phase 2 — Data Pipeline
Week 3
Goal: Clean data pipeline ready to feed your model.
Step 1 — Download TinyShakespeare
One line: wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
Small enough to train fast. Rich enough to show real language patterns. Perfect for your hardware.
Step 2 — Data preparation script
Write data/prepare.py that:

Loads raw text
Tokenizes using your BPE tokenizer
Creates train/validation split — 90% train, 10% validation
Saves as binary files for fast loading

Step 3 — Dataset class
Write data/dataset.py with a PyTorch Dataset class that:

Loads your tokenized data
Returns chunks of fixed context length
Handles batching correctly

Step 4 — Verify data pipeline
Load one batch on your laptop. Print shapes. Print a decoded sample to make sure tokenization is working. Fix all issues locally before going to Colab.
Deliverable at end of Phase 2:
Data pipeline verified locally, ready for Colab training.

Phase 3 — Training Infrastructure
Week 4
Goal: Professional training loop with all modern optimization techniques.
Step 1 — Configuration system
Write configs/train_config.yaml:
yamlmodel:
  n_layers: 4
  n_heads: 4
  d_model: 128
  context_length: 256
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 3e-4
  max_steps: 5000
  warmup_steps: 100
  grad_clip: 1.0
  
data:
  dataset: tinyshakespeare
  train_split: 0.9
```

Every experiment uses a different config file. This is how reproducibility works.

**Step 2 — Training loop**
Write `training/trainer.py` implementing:
- Forward pass
- Cross entropy loss calculation
- Backward pass
- Gradient clipping — clip at 1.0, record why this matters
- Optimizer step
- Learning rate scheduler step
- Periodic validation loss calculation
- Checkpoint saving every N steps

**Step 3 — Learning rate scheduler**
Write `training/scheduler.py` implementing:
- Linear warmup for first N steps
- Cosine decay after warmup

This is exactly how GPT-3 and every modern LLM is trained. Knowing this detail matters in interviews.

**Step 4 — Mixed precision training**
Add `torch.cuda.amp` automatic mixed precision to your training loop. This halves memory usage on Colab T4. Necessary to train even 6M parameter models on free GPU.

**Step 5 — MLflow logging**
Log to MLflow every 100 steps:
- Training loss
- Validation loss
- Perplexity
- Learning rate
- Gradient norm

**Deliverable at end of Phase 3:**
Complete training infrastructure ready to run on Colab.

---

## Phase 4 — Training Experiments on Colab
### Week 5

**Goal:** Four clean experiments with honest documented results.

**Before each Colab session:**
Upload your code to Google Drive. At start of session: mount Drive, install dependencies, run training. Save checkpoint back to Drive before session ends. Never lose work.

**Experiment 1 — Baseline small model**
Config: 4 layers, 4 heads, 128 d_model, 1.2M parameters
Train for 5000 steps. Record loss curve and final perplexity.
Expected time: 15-20 minutes on T4.

**Experiment 2 — Larger model**
Config: 6 layers, 6 heads, 256 d_model, 6M parameters
Train for 5000 steps. Record same metrics.
Expected time: 25-35 minutes on T4.

**Experiment 3 — Training stability**
Take your 1.2M config. Run twice — once with gradient clipping, once without. Show what happens to training when gradients explode. This graph is powerful evidence of deep understanding.

**Experiment 4 — Learning rate effect**
Run three times with learning rates 1e-3, 3e-4, 1e-4. Show convergence curves. Show which one diverges, which one is too slow, which one is right. This is textbook research methodology.

**Build your results table:**
```
Experiment | Params | Perplexity | Train time | Key finding
Baseline   | 1.2M   | 28.4       | 18 min     | stable training
Larger     | 6M     | 22.1       | 32 min     | better perplexity
No clip    | 1.2M   | diverged   | stopped    | exploding gradients
Best LR    | 1.2M   | 26.8       | 18 min     | 3e-4 optimal
Deliverable at end of Phase 4:
Four completed experiments, results table, loss curve plots saved.

Phase 5 — LoRA Fine-tuning Addition
Week 6
Goal: Add LoRA fine-tuning to cover the fine-tuning gap without new hardware.
Step 1 — Understand LoRA mathematically
LoRA freezes original weights and learns a low-rank update.
Write a one page explanation in your own words before writing any code, including:
- what is frozen (base weights)
- what is trained (LoRA matrices A and B)
- the forward update rule and scaling
- why rank=4 keeps trainable params small
Use the best Phase 4 baseline checkpoint (`exp1/step_5000.pt`) as the starting point for LoRA fine-tuning.
Step 2 — Implement LoRA layers
Write `model/lora.py` implementing production LoRA modules that can wrap `nn.Linear`.
Apply LoRA (rank=4) to attention projection linears:
- `W_q`, `W_k`, `W_v`, and `W_o` (or at minimum `W_q` and `W_v` as a smaller ablation)
Ensure only LoRA adapter parameters require gradients; base model parameters remain frozen.

**Step 3 — Fine-tune experiment**
Take your best trained checkpoint and apply LoRA adapters.
Fine-tune on a different small text dataset (poetry/news) for 1000 steps.
Reuse the Phase 2 tokenizer and build fine-tune `train.bin`/`val.bin` from the new corpus.
Generate samples before/after fine-tuning and compare them.

**Step 4 — Document parameter efficiency**
Record and display:
```
Full fine-tuning: 6M trainable parameters
LoRA fine-tuning: 48K trainable parameters (0.8% of total)
Quality difference: minimal
```

This one comparison makes senior engineers nod in interviews.

**Deliverable at end of Phase 5:**
Working LoRA implementation, fine-tuning experiment completed, parameter efficiency documented.

---

## Phase 6 — Inference API and Dashboard
### Week 7

**Goal:** Your trained model served as a proper API with a visual interface.

**Step 1 — Inference engine**
Write `inference/generate.py` implementing:
- Greedy decoding
- Temperature sampling
- Top-k sampling
- Top-p nucleus sampling

Each sampling method as a separate clean function. Show you understand the differences.

**Step 2 — FastAPI server**
Write `api/app.py` with endpoints:
```
POST /generate        — text generation with sampling params
GET  /model/info      — model size, architecture details
GET  /experiments     — return all experiment results as JSON
```

**Step 3 — React dashboard**
Build a clean dashboard with three sections:
- Text generation playground — type a prompt, see your model generate
- Experiment results — interactive charts of all training curves
- Architecture explorer — visual breakdown of model parameters by layer

**Step 4 — Connect LLM Inference project**
This is the bonus connection. Add one page to your dashboard comparing your NanoGPT inference speed against quantized Mistral. You now have a direct bridge between both projects. One portfolio story.

**Deliverable at end of Phase 6:**
Live inference API, React dashboard, generation playground working.

---

## Phase 7 — Polish, Deploy, Document
### Week 8

**Goal:** Both projects presented at professional level.

**Step 1 — Docker for NanoGPT**
Same pattern as LLM Inference:
```
docker-compose up
Starts API, dashboard, MLflow. Everything in one command.
Step 2 — Deploy
Deploy inference API to Railway or Hugging Face Spaces. Use your smallest checkpoint — 1.2M model runs on CPU inference fine.
Step 3 — Write technical blog post
Write one Medium article titled something like:
"What I learned building a GPT from scratch on a free GPU"
Cover your scaling experiment findings. Link to your GitHub. This partially substitutes for research publications and drives profile visibility.
Step 4 — READMEs for both projects
Every README must have:

One paragraph explaining what the project proves
Architecture diagram
Results table with real numbers
Setup instructions — one command
Live demo URL
Link to demo video

Step 5 — Demo videos
NanoGPT: Show model generating Shakespeare-style text, show training curves, show LoRA fine-tuning comparison.
LLM Inference: Show streaming response, show adaptive router decision, show benchmark dashboard.
Final deliverable:
Both projects complete, deployed, documented, with demo videos. Your full 5-project portfolio is ready.
