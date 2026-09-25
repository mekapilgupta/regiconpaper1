# Paper 1: Evaluation Invariance under Multilingual Benchmark Transformation

This repository contains the complete experimental framework to evaluate whether common low-resource multilingual benchmark transformations (Machine Translation, LLM Translation, Post-Editing, Paraphrasing, and Synthetic Query Generation) preserve the **scientific conclusions** of information retrieval evaluation.

---

## 📁 Repository Structure

```
paper1/
├── .env.example                               # Template for OpenRouter API keys & model selection
├── .env                                       # Active environment configuration
├── config.py                                  # Model endpoints, seeds, language maps & prompt templates
├── requirements.txt                           # Python dependencies
├── src/
│   ├── openrouter_client.py                   # Async / cached OpenRouter API client
│   ├── dataset_loader.py                      # IndicMSMARCO loader & deterministic split creator
│   ├── transformers.py                        # C0 to C5 query transformation engine
│   ├── judge.py                               # Multi-model LLM quality validation
│   ├── retrievers.py                          # 6 Retrieval Systems (BM25, BGE-M3, mE5-base, mE5-large, LaBSE, Hybrid-RRF)
│   └── evaluation.py                          # Invariance metrics, Kendall's Tau, sign-flips & IR metrics
├── run_pilot.py                               # Pilot Runner (~150 queries × 2 languages)
├── run_full.py                                # Scaled Full Runner (800+ queries/lang)
├── run_analysis.py                            # LaTeX publication tables & statistical reporting
└── paper1_evaluation_invariance_execution_protocol.md  # Official protocol document
```

---

## ⚙️ Configuration & Models (`.env`)

Edit your `.env` file to add your OpenRouter API key:

```bash
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Strictly Configured Transformation Models
MODEL_PRIMARY_TRANSFORM=openai/gpt-5.6-luna
MODEL_SECONDARY_TRANSFORM=qwen/qwen3.8-flash

# Multi-Model Quality Judges
MODEL_PRIMARY_JUDGE=google/gemini-3.8-flash
MODEL_SECONDARY_JUDGE=deepseek/deepseek-v4-pro
```

---

## 🧪 Experimental Conditions & Retrieval Suite

### 1. Benchmark Transformations (C0 - C5):
- **C0 — REF**: Benchmark-provided native query ($q_{ref}$)
- **C1 — MT**: Machine translation via IndicTrans2
- **C2 — LLM-TRANSLATE**: Direct zero-shot translation via `openai/gpt-5.6-luna`
- **C3 — MT+POSTEDIT**: Post-edited MT via `openai/gpt-5.6-luna`
- **C4 — PARAPHRASE**: In-language semantic paraphrase of $q_{ref}$
- **C5 — SYNTHETIC**: Passage-conditioned question generated from relevant document and answer

### 2. Retrieval Systems (6 Systems):
1. **BM25** (Lexical baseline)
2. **BGE-M3** (`BAAI/bge-m3`)
3. **mE5-base** (`intfloat/multilingual-e5-base`)
4. **mE5-large** (`intfloat/multilingual-e5-large`)
5. **LaBSE** (`sentence-transformers/LaBSE`)
6. **Hybrid-RRF** (Reciprocal Rank Fusion of BM25 + BGE-M3)

---

## 🚀 Execution Commands (Run on GPU Node)

Navigate to the `paper1` directory on your GPU compute node:

```bash
cd /home/kshitij22256/fineline/paper1
```

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Pilot Study (~150 queries × Assamese & Bengali)
```bash
python run_pilot.py --languages as,bn --sample_size 150 --device cuda
```

### Step 3: Generate LaTeX Tables & Invariance Report
```bash
python run_analysis.py
```

### Step 4: Run Full-Scale Experiment (800 queries / language)
```bash
python run_full.py --languages as,bn --device cuda
```

All results, performance breakdowns (nDCG@10, MRR@10, Recall@10), Kendall's $\tau$, pairwise sign flips, and publication LaTeX tables are written to [`outputs/`](file:///home/kshitij22256/fineline/paper1/outputs) and [`outputs/results/`](file:///home/kshitij22256/fineline/paper1/outputs/results).
