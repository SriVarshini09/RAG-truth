# CKPT2 Technical Report Notes

**Student:** Tulasi Padamata  
**Project:** Multi-Agent Hallucination Detection in RAG Systems  
**Due:** March 13, 2026, 11:59 PM

---

## Formatting Requirements ✓ [10pts]

- **Font:** 11-point
- **Margins:** 1 inch all sides
- **Length:** Within 2 pages (excluding references)
- **File name:** `Padamata_Ckpt2.pdf`

---

## Section 1: Training/Test Data Demonstration [30pts]

### 1.1 Data Statistics

**RAGTruth Dataset Overview:**
- **Total Records:** 17,617 LLM-generated responses
- **Annotation:** Word-level hallucination spans (human-annotated)
- **Tasks:** 3 RAG tasks
  - Question Answering (MS MARCO): 5,767 samples (32.7%)
  - Summarization (CNN/DailyMail): 5,655 samples (32.1%)
  - Data-to-Text (Yelp): 6,195 samples (35.2%)
- **LLMs:** 6 models (GPT-4, ChatGPT, Llama-2-7B/13B/70B, Mistral-7B)
- **Label Types:** 4 hallucination categories
  - Evident Baseless Info
  - Evident Conflict
  - Subtle Baseless Info
  - Subtle Conflict

**Data Splits:**
| Split | Size | Percentage | Purpose |
|-------|------|------------|---------|
| Training | 14,047 | 79.7% | Fine-tuning supervised baselines |
| Development | 895 | 5.1% | 50 source IDs per task for hyperparameter tuning |
| Test | 2,675 | 15.2% | Held-out evaluation |

**Hallucination Rates by Task:**
- Data2txt: 68.6% (highest - sparse structured data leads to fabrication)
- Summary: 29.8%
- QA: 29.6%

### 1.2 Visualization of Training/Test Data

**Include these figures:**

1. **Figure 1: Task Type Distribution** (Bar chart)
   - Shows balanced distribution across 3 tasks
   - Caption: "RAGTruth dataset contains balanced samples across QA, Summarization, and Data-to-Text tasks"

2. **Figure 2: Hallucination Rate by Task** (Bar chart)
   - Data2txt (68.6%), Summary (29.8%), QA (29.6%)
   - Caption: "Data2txt exhibits the highest hallucination rate due to sparse structured data"

3. **Table 1: Sample Data Cards** (2-3 examples)
   - Show question/prompt, source passages, LLM response, annotated hallucination spans
   - Include label types for each span

### 1.3 Data Split Strategy

**Validation and Testing Approach:**

1. **Official RAGTruth Splits:** Used pre-defined train/test splits from the dataset
2. **Development Set:** 50 source IDs per task (150 total) sampled from training set
3. **Test Set:** Held-out 2,675 samples for final evaluation
4. **Balanced Sampling:** For baseline evaluation, we sample 200 per task (600 total) to ensure representative performance across all task types
5. **No Data Leakage:** Source IDs in dev set are excluded from training to prevent overfitting

**Rationale:** This strategy ensures:
- Fair comparison across task types
- Sufficient statistical power (600 samples)
- Reproducibility using official splits

---

## Section 2: Baseline Results [30pts]

### 2.1 Two-Track Evaluation Approach

**IMPORTANT:** We evaluate methods across two separate tracks:

**Track A: Hallucination Detection** (Primary Research Question)
- Task: Binary classification - does response contain hallucinations?
- Ground Truth: Human-annotated hallucination spans
- Metrics: Case-level F1, Span-level F1, Precision, Recall, Bootstrap 95% CI

**Track B: RAG QA Performance** (Supplementary, per Professor Feedback)
- Task: Generate answers from retrieved passages
- Ground Truth: None (RAGTruth is not a QA dataset)
- Metrics: Answerable rate, answer length distribution

**Note:** These tracks solve different tasks and cannot be directly compared.

### 2.2 Baseline Methods

**Table 2: Baseline Comparison**

| Track | Baseline | Method | Type | Reference |
|-------|----------|--------|------|-----------|
| A | RAGTruth LLaMA-2-13B | Fine-tuned to output hallucination spans | Supervised fine-tuning | Wu et al., 2023 |
| A | Self-Verification (GPT-4o-mini) | LLM checks output against source | Inference-time, zero-shot | Manakul et al., 2023 |
| A | Multi-Agent Pipeline (proposed) | GPT + DeBERTa NLI + rules | Inference-time, no training | CKPT3 |
| B | RAG System (GPT-4o-mini) | Generate answers from passages | Inference-time, zero-shot | This work |

### 2.3 Track A Results: Hallucination Detection

**Table 3: Self-Verification Baseline Performance (n=600)**

| Metric | Overall | QA | Summary | Data2txt |
|--------|---------|-----|---------|----------|
| Precision | 0.6189 | 0.4510 | 0.3837 | 0.8438 |
| Recall | 0.7523 | 0.6571 | 0.6471 | 0.8182 |
| **F1 Score** | **0.6791** | 0.5349 | 0.4818 | **0.8308** |
| 95% CI | [0.6307, 0.7240] | - | - | - |
| Samples | 600 | 200 | 200 | 200 |

**Key Findings:**
- Overall F1: 0.6791 with tight 95% CI [0.6307, 0.7240]
- Data2txt performs best (F1=0.8308) - structured data enables clear verification
- Summary performs worst (F1=0.4818) - confirmation bias in self-verification
- Span-level mean F1: 0.5709 (room for improvement in precise localization)

**Figure 3: Performance by Task Type** (Bar chart)
- Show Precision, Recall, F1 for each task
- Caption: "Self-verification baseline shows strong performance on Data2txt but struggles with Summary tasks due to confirmation bias"

**Figure 4: Confusion Matrix** (Heatmap)
- TN: 281, FP: 101, FN: 54, TP: 164
- Accuracy: 74.17%
- Caption: "Confusion matrix shows 101 false positives (over-detection) and 54 false negatives (missed hallucinations)"

**Figure 5: Bootstrap Confidence Interval** (Error bar plot)
- Mean F1: 0.6786, CI: [0.6307, 0.7240]
- Caption: "Bootstrap 95% confidence interval demonstrates statistical significance with 600 samples"

### 2.4 Track B Results: RAG QA Performance

**Table 4: RAG System Performance (n=100)**

| Metric | Value |
|--------|-------|
| Answerable Rate | 90.0% |
| Cannot Answer Rate | 10.0% |
| Average Answer Length | 74.1 words |
| Min/Max Length | 8 / 125 words |

**Answer Quality Distribution:**
- Long (>50 words): 70 (70.0%)
- Medium (20-50 words): 20 (20.0%)
- Cannot Answer: 10 (10.0%)

**Figure 6: Answer Quality Distribution** (Bar chart)
- Caption: "RAG system achieves 90% answerable rate with conservative behavior when passages lack information"

---

## Section 3: Codebase Description [20pts]

### 3.1 Repository Structure

```
capstone_project/
├── RAGTruth/baseline/          # Official baseline (cloned)
│   ├── prepare_dataset.py      # Data split generation
│   ├── dataset.py              # Prompt templates
│   ├── train.py                # LLaMA-2-13B fine-tuning
│   └── predict_and_evaluate.py # Inference + metrics
├── evaluation/                 # Unified evaluation module
│   ├── metrics.py              # Core metrics functions
│   ├── run_eval.py             # CLI evaluation runner
│   └── README.md               # Documentation
├── baselines/
│   ├── run_gpt_baseline.py     # Self-verification (Track A)
│   └── run_rag_baseline.py     # RAG generator (Track B)
├── scripts/
│   └── phase1_analysis.py      # Automated analysis
├── results/
│   ├── gpt_baseline_predictions.jsonl  # 600 predictions
│   └── phase1_full_report.json         # Complete report
└── CKPT2_Final_Submission.ipynb        # Comprehensive demo
```

### 3.2 Key Implementation Details

**Self-Verification Baseline** (`baselines/run_gpt_baseline.py`):
- Reuses RAGTruth prompt templates from official `dataset.py`
- Balanced sampling: 200 samples per task for representative evaluation
- GPT-4o-mini with temperature=0.0 for deterministic outputs
- JSON parsing for hallucination span extraction
- Retry logic (3 attempts) for API robustness

**RAG Baseline** (`baselines/run_rag_baseline.py`):
- Uses pre-retrieved passages from RAGTruth `reference` field
- GPT-4o-mini with temperature=0.3 for consistent answers
- Max 150 tokens to prevent verbose responses
- Conservative prompting: "If passages lack info, say 'I cannot answer'"

**Unified Evaluation Module** (`evaluation/metrics.py`):
- `case_level_metrics()`: Binary classification (P, R, F1)
- `per_task_metrics()`: Task-specific breakdown
- `span_level_f1()`: Character-overlap F1 for span localization
- `bootstrap_confidence_interval()`: Statistical validation (1000 iterations)
- `full_evaluation_report()`: Complete metrics in one call

**CLI Tool** (`evaluation/run_eval.py`):
```bash
python evaluation/run_eval.py \
    --predictions results/gpt_baseline_predictions.jsonl \
    --method "Self-Verification" \
    --output results/report.json
```

### 3.3 Reproducibility

All experiments are reproducible via:
1. **Notebook:** `CKPT2_Final_Submission.ipynb` (executed with all outputs)
2. **Scripts:** `scripts/phase1_analysis.py` for automated evaluation
3. **CLI:** `evaluation/run_eval.py` for consistent metrics
4. **GitHub:** `https://github.com/SriVarshini09/RAG-truth.git`

---

## Section 4: Response to Professor Feedback [10pts]

### **Response to Professor Feedback** (BOLD as required)

**Professor's Feedback:** "Datasets have been prepared well, and the API performance by GPT-4o mini has been reported. Please build the baseline RAG system to report the QA performance."

**Our Response:**

**We have implemented and evaluated the baseline RAG system on 100 QA test samples from RAGTruth. The system achieves a 90% answerable rate with an average answer length of 74 words, demonstrating effective passage-based question answering. The 10% "cannot answer" rate reflects conservative behavior when passages lack sufficient information, which is desirable to prevent hallucination.**

**To address the professor's feedback while maintaining research clarity, we established two separate evaluation tracks:**

1. **Track A (Hallucination Detection)** - Primary research question
   - Self-Verification baseline: F1=0.6791 (95% CI: [0.6307, 0.7240])
   - Evaluates ability to detect hallucinations in LLM responses
   - Uses human-annotated ground truth from RAGTruth

2. **Track B (RAG QA Performance)** - Supplementary, per professor's request
   - RAG System baseline: 90% answerable rate, 74.1 avg words
   - Evaluates answer generation quality
   - No ground truth available (RAGTruth is not a QA dataset)

**These tracks solve different tasks and cannot be directly compared. The multi-agent system (CKPT3) will be evaluated using Track A's protocol to improve on the self-verification baseline's performance, particularly on Summary tasks (F1=0.4818) where confirmation bias is problematic.**

**The RAG baseline complements the hallucination detection work by demonstrating that the end-to-end RAG pipeline functions correctly, addressing the professor's concern about QA performance reporting.**

---

## References

1. Wu, Y., Zhu, J., Xu, S., Shum, K., Niu, C., Zhong, R., Song, J., & Zhang, T. (2023). RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models. *arXiv preprint arXiv:2401.00396*.

2. Manakul, P., Liusie, A., & Gales, M. J. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. *arXiv preprint arXiv:2303.08896*.

---

## Figures and Tables Summary

**Required Visualizations:**

1. **Figure 1:** Task Type Distribution (bar chart)
2. **Figure 2:** Hallucination Rate by Task (bar chart)
3. **Figure 3:** Performance by Task Type (grouped bar chart)
4. **Figure 4:** Confusion Matrix (heatmap)
5. **Figure 5:** Bootstrap Confidence Interval (error bar)
6. **Figure 6:** RAG Answer Quality Distribution (bar chart)

**Required Tables:**

1. **Table 1:** Data Split Statistics
2. **Table 2:** Baseline Comparison
3. **Table 3:** Self-Verification Performance (Track A)
4. **Table 4:** RAG System Performance (Track B)

---

## Writing Tips

1. **Be concise:** 2-page limit means every sentence counts
2. **Use bold for professor feedback response:** Make it easy to find
3. **Reference figures/tables:** "As shown in Figure 3..."
4. **Quantify everything:** Use specific numbers, not vague descriptions
5. **Highlight key findings:** Use bold or italics for important results
6. **Connect to CKPT3:** Mention how baselines inform future work
7. **Professional tone:** Technical but accessible

---

## Grading Checklist

- [ ] 11-point font, 1-inch margins, ≤2 pages (excluding refs) [10pts]
- [ ] Data statistics (dimensions, samples, classes) [10pts]
- [ ] Data visualization (tables/plots) [10pts]
- [ ] Data split strategy explained [10pts]
- [ ] Baseline results table/figure [15pts]
- [ ] Numerical evaluation or empirical study [15pts]
- [ ] Codebase description with file names [20pts]
- [ ] Professor feedback response (in bold) [10pts]

**Total: 100 points**
