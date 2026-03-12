# Evaluation Module

Unified evaluation framework for hallucination detection methods in the RAG system project.

## Overview

This module provides consistent metrics computation and evaluation protocols for all hallucination detection baselines (Track A). It ensures fair comparison by using the same test set, ground truth, and metrics across all methods.

## Files

### `metrics.py`

Core metrics computation functions:

- **`case_level_metrics(records)`**: Computes binary classification metrics (precision, recall, F1)
- **`per_task_metrics(records)`**: Breaks down metrics by task type (QA, Summary, Data2txt)
- **`span_level_f1(pred_spans, gold_spans, response)`**: Character-overlap F1 for hallucination spans
- **`compute_span_level_metrics(records)`**: Aggregates span-level metrics across all records
- **`bootstrap_confidence_interval(records, n_iterations, confidence)`**: Bootstrap CI for F1 scores
- **`compute_confusion_matrix(records)`**: Confusion matrix for binary classification
- **`compute_classification_report(records)`**: Detailed classification report
- **`hallucination_type_breakdown(records)`**: Counts by hallucination type
- **`full_evaluation_report(records, method_name)`**: Complete evaluation report with all metrics

### `run_eval.py`

Command-line evaluation runner:

```bash
python evaluation/run_eval.py \
    --predictions results/gpt_baseline_predictions.jsonl \
    --method "Self-Verification GPT-4o-mini" \
    --output results/self_verification_report.json
```

**Arguments:**
- `--predictions`: Path to predictions JSONL file (required)
- `--method`: Method name for the report (default: "Method")
- `--output`: Path to save JSON report (optional)
- `--quiet`: Suppress console output

## Usage Examples

### 1. Compute Case-Level Metrics

```python
from evaluation.metrics import case_level_metrics

records = load_jsonl("results/predictions.jsonl")
metrics = case_level_metrics(records)

print(f"Precision: {metrics['precision']}")
print(f"Recall: {metrics['recall']}")
print(f"F1: {metrics['f1']}")
```

### 2. Per-Task Breakdown

```python
from evaluation.metrics import per_task_metrics

task_metrics = per_task_metrics(records)
for task, metrics in task_metrics.items():
    print(f"{task}: F1 = {metrics['f1']}")
```

### 3. Bootstrap Confidence Intervals

```python
from evaluation.metrics import bootstrap_confidence_interval

ci = bootstrap_confidence_interval(records, n_iterations=1000, confidence=0.95)
print(f"Mean F1: {ci['mean_f1']}")
print(f"95% CI: [{ci['ci_lower']}, {ci['ci_upper']}]")
```

### 4. Full Evaluation Report

```python
from evaluation.metrics import full_evaluation_report

report = full_evaluation_report(records, method_name="My Method")
# Returns dict with all metrics, confusion matrix, classification report, etc.
```

### 5. Command-Line Evaluation

```bash
# Evaluate predictions and save report
python evaluation/run_eval.py \
    --predictions results/gpt_baseline_predictions.jsonl \
    --method "Self-Verification" \
    --output results/report.json

# Quick evaluation without saving
python evaluation/run_eval.py \
    --predictions results/predictions.jsonl \
    --method "Multi-Agent"
```

## Input Format

Predictions JSONL file must contain records with:

```json
{
  "task_type": "QA",
  "response": "The LLM-generated response text",
  "labels": [
    {"start": 10, "end": 20, "text": "hallucinated span", "label_type": "Evident Baseless Info"}
  ],
  "pred": ["predicted hallucination span 1", "predicted hallucination span 2"],
  "question": "...",
  "reference": "..."
}
```

**Required fields:**
- `labels`: List of ground truth hallucination spans (can be empty)
- `pred`: List of predicted hallucination spans (can be empty)
- `response`: The LLM-generated response text
- `task_type`: One of "QA", "Summary", "Data2txt"

## Metrics Definitions

### Case-Level Metrics
- **Binary classification**: Does the response contain ANY hallucination?
- **y_true**: 1 if `len(labels) > 0`, else 0
- **y_pred**: 1 if `len(pred) > 0`, else 0
- **Metrics**: Precision, Recall, F1 (sklearn)

### Span-Level Metrics
- **Character-overlap F1**: How well do predicted spans match gold spans?
- **Precision**: `overlap_chars / pred_chars`
- **Recall**: `overlap_chars / gold_chars`
- **F1**: Harmonic mean of precision and recall

### Bootstrap Confidence Intervals
- **Method**: Resample records with replacement 1000 times
- **Compute**: F1 on each bootstrap sample
- **CI**: 2.5th and 97.5th percentiles for 95% CI

## Track A vs Track B

This evaluation module is designed for **Track A: Hallucination Detection** only.

**Track A methods** (use this module):
- RAGTruth LLaMA-2-13B
- Self-Verification GPT-4o-mini
- Multi-Agent Pipeline (CKPT3)

**Track B methods** (different evaluation):
- RAG System GPT-4o-mini (uses answerable rate, answer length)

Track B cannot use this module because it solves a different task (answer generation vs hallucination detection).

## Statistical Validation

All Track A methods should report:
1. **Point estimates**: Precision, Recall, F1
2. **Confidence intervals**: 95% bootstrap CI for F1
3. **Per-task breakdown**: Metrics for QA, Summary, Data2txt separately
4. **Span-level metrics**: Fine-grained hallucination localization

When comparing methods, use:
- **Paired significance tests** (if same test set)
- **Non-overlapping CIs** as evidence of significant difference
- **Effect size** (Cohen's d) for practical significance
