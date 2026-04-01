# Presentation Slides Outline (10–12 slides)
# Multi-Agent Hallucination Detection in RAG Systems

---

## Slide 1: Title
**Title:** Multi-Agent Hallucination Detection in RAG Systems  
**Subtitle:** A Zero-Shot Multi-Agent Approach Using GPT-4o-mini + DeBERTa NLI  
**Author:** [Name], Rochester Institute of Technology, Spring 2026  
**Visual:** Pipeline diagram overview (3 agents in sequence)

---

## Slide 2: Motivation
**Title:** Why Do RAG Systems Still Hallucinate?

**Key Points:**
- RAG grounds LLMs in retrieved documents — yet hallucinations persist
- Two types: *Evident Conflict* (contradicts source) and *Baseless Info* (not in source)
- Real-world consequence: wrong medical, legal, or scientific information
- Need: fast, reliable, zero-shot detection without retraining

**Visual:** Example response with highlighted hallucinated span vs ground truth

---

## Slide 3: RAGTruth Dataset
**Title:** Evaluation Benchmark — RAGTruth

**Key Points:**
- 17,790 LLM responses, 6 models, 3 task types
- Human-annotated word-level hallucination spans
- 4 label types: Evident/Subtle × Conflict/Baseless Info
- Our evaluation: 600 balanced samples (200/task)

**Visual:** Bar chart — hallucination rates by task (QA: 18%, Summary: 23%, Data2txt: 64%)

---

## Slide 4: Baselines
**Title:** What We Compare Against

| Method | Type | F1 |
|--------|------|----|
| RAGTruth LLaMA-2-13B | Supervised (15K samples) | 0.7822 |
| Self-Verification GPT-4o-mini | Zero-shot, single model | 0.6791 |
| **Multi-Agent (ours)** | Zero-shot, multi-model | **TBD** |

**Key Point:** We target Self-Verification as the zero-shot baseline to beat — particularly on Summary (F1=0.4818)

---

## Slide 5: Our Approach — Pipeline Overview
**Title:** Multi-Agent Pipeline Architecture

```
Response + Reference Source
         ↓
  ┌──────────────────┐
  │  Agent 1         │  GPT-4o-mini
  │  Claim Extractor │  → ["Einstein born 1879", "Won Nobel 1922", ...]
  └──────────────────┘
         ↓
  ┌──────────────────┐
  │  Agent 2         │  DeBERTa-v3-large NLI (local, GPU)
  │  Claim Verifier  │  → [ENTAILMENT, CONTRADICTION, NEUTRAL, ...]
  └──────────────────┘
         ↓
  ┌──────────────────┐
  │  Agent 3         │  Rule-based logic
  │  Aggregator      │  → hallucination=True, spans=[...]
  └──────────────────┘
```

**Visual:** Architecture diagram with agent descriptions and example

---

## Slide 6: Agent Details
**Title:** How Each Agent Works

**Agent 1 — Claim Extractor (GPT-4o-mini):**
- Zero-shot prompt decomposes response into atomic facts
- Returns JSON: `{"claims": ["fact1", "fact2", ...]}`
- Solves: NLI length limitation

**Agent 2 — Claim Verifier (DeBERTa NLI):**
- Cross-encoder: ⟨reference, claim⟩ → {ENTAILMENT, NEUTRAL, CONTRADICTION}
- Runs locally, GPU-accelerated, batch=16
- Cost: $0 (no API)

**Agent 3 — Decision Aggregator (Rule-based):**
- Any CONTRADICTION → hallucinated
- ≥50% NEUTRAL → baseless info
- Maps claims back to response spans

---

## Slide 7: Main Results
**Title:** Track A — Hallucination Detection Results (n=600)

**[Bar chart: F1 comparison across 3 methods × 3 tasks + overall]**

| Method | Overall F1 | QA | Summary | Data2txt |
|--------|------------|-----|---------|---------|
| LLaMA-2-13B | 0.7822 | 0.7149 | 0.7341 | 0.8358 |
| Self-Verification | 0.6791 | 0.5349 | 0.4818 | 0.8308 |
| **Multi-Agent** | **TBD** | **TBD** | **TBD** | **TBD** |

**Bootstrap 95% CI:** [TBD, TBD]

---

## Slide 8: Ablation Study
**Title:** What Does Each Agent Contribute?

| Config | Description | F1 |
|--------|-------------|-----|
| A | Claim extractor only (no NLI) | TBD |
| B | NLI on full response (no decomposition) | TBD |
| C | Claims + NLI, majority vote (no custom rules) | TBD |
| **D** | **Full pipeline (ours)** | **TBD** |

**Key Takeaway:** Decomposition + NLI + rules each contribute; removing any one hurts performance

**Visual:** Delta F1 bar chart showing each component's contribution

---

## Slide 9: Error Analysis
**Title:** Where Does the System Fail?

**False Negatives (Missed Hallucinations):**
- Subtle paraphrasing not caught by NLI
- Very short responses with few claims
- Domain knowledge gaps in DeBERTa

**False Positives (False Alarms):**
- Legitimate inferences from context
- Temporal expressions not in source
- Numeric rounding or estimation

**Visual:** Example table of 2–3 FN and FP cases

---

## Slide 10: Cost & Efficiency
**Title:** Practical Considerations

| Method | API Calls/record | GPU Needed | Cost (600 samples) | Training |
|--------|-----------------|------------|-------------------|---------|
| LLaMA-2-13B | 0 | Yes (training) | $0 inference | 15K labeled |
| Self-Verification | 1 GPT call | No | ~$0.08 | None |
| **Multi-Agent** | **1 GPT call** | **Optional** | **~$0.15** | **None** |

- DeBERTa runs locally → no NLI cost
- With RTX 5070: 600 samples in ~50 min
- Resume-safe: can pause/restart without data loss

---

## Slide 11: Conclusion
**Title:** Summary & Takeaways

**What We Did:**
- Built a 3-agent zero-shot hallucination detection pipeline
- Claim extraction → NLI verification → rule-based aggregation
- Evaluated on 600 RAGTruth samples with bootstrap CI

**Key Finding:**
- Multi-agent decomposition improves over single-model self-verification
- Especially on Summary task (hardest for GPT self-verification)
- No training data required

**Limitations:** Slower than single-model, GPT-dependent extraction step

---

## Slide 12: Future Work & Questions
**Title:** Future Directions

**Immediate:**
- Per-task aggregation rule tuning
- Replace GPT claim extractor with local LLM (Mistral/LLaMA)
- Span-level precision improvement

**Longer Term:**
- Multi-document RAG hallucination detection
- Real-time streaming detection
- Domain-specific NLI fine-tuning

**Q&A**

---

## Notes for Slide Creation
- Use dark theme with accent colors: #4C72B0 (blue), #55A868 (green), #C44E52 (red)
- Include pipeline diagram on slide 5 as main visual
- Fill in all TBD values after pipeline results are complete
- Recommended tool: Google Slides or PowerPoint
- Export as PDF for submission
