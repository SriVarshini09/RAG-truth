# Multi-Agent Hallucination Detection in RAG Systems

**Capstone Project** | RIT | Spring 2026  
Dataset: [RAGTruth](https://github.com/ParticleMedia/RAGTruth) (Wu et al., 2023)

---

## Overview

We propose a **zero-shot multi-agent pipeline** for detecting hallucinations in RAG-generated responses. The system chains three specialized agents without requiring any model fine-tuning:

1. **Agent 1 – Claim Extractor** (`agents/claim_extractor.py`): GPT-4o-mini decomposes responses into atomic factual claims.
2. **Agent 2 – Claim Verifier** (`agents/claim_verifier.py`): DeBERTa-v3-large NLI verifies each claim against the source passage (GPU-accelerated, batch inference).
3. **Agent 3 – Decision Aggregator** (`agents/decision_aggregator.py`): Rule-based logic aggregates verdicts into a binary hallucination label + span predictions.

---

## Dataset

**RAGTruth** — 17,790 human-annotated LLM responses across 3 tasks:

| Split | Records | Notes |
|-------|---------|-------|
| Train | 15,090 | Used for official LLaMA-2-13B baseline only |
| Test (full) | 2,700 | Full evaluation set |
| **Test (balanced)** | **600** | **Primary eval: 200/task, seed=42** |

Tasks: QA · Summary · Data2txt  
Models: GPT-4, GPT-3.5, Mistral-7B, LLaMA-2 (7B/13B/70B)

---

## Results (Track A — Hallucination Detection)

| Method | Precision | Recall | **F1** | QA F1 | Summary F1 | Data2txt F1 |
|--------|-----------|--------|--------|-------|------------|-------------|
| RAGTruth LLaMA-2-13B (official) | 0.7380 | 0.8320 | **0.7822** | 0.7149 | 0.7341 | 0.8358 |
| Self-Verification GPT-4o-mini (CKPT2) | — | — | **0.6791** | 0.5349 | 0.4818 | 0.8308 |
| **Multi-Agent Pipeline (ours)** | — | — | **TBD** | TBD | TBD | TBD |

> Results pending full 600-sample pipeline run.

---

## Project Structure

```
capstone project/
├── agents/
│   ├── claim_extractor.py      # Agent 1: GPT-4o-mini claim extraction
│   ├── claim_verifier.py       # Agent 2: DeBERTa NLI verification (GPU)
│   └── decision_aggregator.py  # Agent 3: Rule-based aggregation
├── pipeline/
│   ├── run_pipeline.py         # Full orchestration (600 samples, resume-safe)
│   └── run_ablation.py         # 4 ablation configurations
├── data/
│   ├── data_pipeline.py        # Load, validate, split RAGTruth data
│   ├── dataset_stats.ipynb     # Dataset visualization notebook
│   ├── splits/                 # train / test_full / test_balanced / dev
│   └── figures/                # 6 dataset visualization figures
├── logs/
│   ├── experiment_logger.py    # Auto-log runs to JSON + summary.csv
│   └── runs/                   # Per-run JSON logs
├── evaluation/
│   └── metrics.py              # Case-level F1, span F1, bootstrap CI
├── baselines/
│   └── run_gpt_baseline.py     # Self-Verification GPT-4o-mini baseline
├── notebooks/
│   └── CKPT3_Results.ipynb     # Full results, comparisons, error analysis
├── results/
│   ├── multi_agent_predictions.jsonl
│   └── ablation/
├── RAGTruth/dataset/           # Raw data (response.jsonl, source_info.jsonl)
├── requirements.txt
└── .env                        # OPENAI_API_KEY (not committed)
```

---

## Setup

```bash
git clone https://github.com/SriVarshini09/RAG-truth.git
cd RAG-truth
pip install -r requirements.txt
echo "OPENAI_API_KEY=your_key_here" > .env
```

### GPU (optional but recommended)
```bash
# NVIDIA RTX 5070 / Blackwell (sm_120) — needs CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128
# DeBERTa automatically uses GPU if CUDA is available
```

---

## Running the Pipeline

```bash
# Full 600-sample run
python pipeline/run_pipeline.py

# Resume interrupted run
python pipeline/run_pipeline.py

# Quick test (5 samples)
python pipeline/run_pipeline.py --limit 5 --no-resume

# Ablation studies (all 4 configs)
python pipeline/run_ablation.py --all

# Results notebook
jupyter notebook notebooks/CKPT3_Results.ipynb
```

---

## Baselines

```bash
# Self-Verification GPT-4o-mini (CKPT2 baseline)
python baselines/run_gpt_baseline.py
```

---

## Evaluation

```bash
python -c "
from pipeline.run_pipeline import quick_eval
quick_eval()
"
```

---

## Experiment Logging

```python
from logs.experiment_logger import ExperimentLogger
logger = ExperimentLogger()
logger.print_summary()  # Compare all logged runs
```
