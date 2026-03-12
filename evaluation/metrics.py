import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from typing import List, Dict


def case_level_metrics(records: List[Dict]) -> Dict:
    y_true = [1 if len(r["labels"]) > 0 else 0 for r in records]
    y_pred = [1 if len(r.get("pred", [])) > 0 else 0 for r in records]
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_samples": len(records),
        "y_true": y_true,
        "y_pred": y_pred
    }


def per_task_metrics(records: List[Dict]) -> Dict:
    results = {}
    for task in ["QA", "Summary", "Data2txt"]:
        subset = [r for r in records if r.get("task_type") == task]
        if subset:
            metrics = case_level_metrics(subset)
            results[task] = metrics
    return results


def span_level_f1(pred_spans: List[str], gold_spans: List[Dict], response: str) -> float:
    if not gold_spans and not pred_spans:
        return 1.0
    if not gold_spans or not pred_spans:
        return 0.0
    
    gold_chars = set()
    for span in gold_spans:
        start = span.get("start", 0)
        end = span.get("end", 0)
        for i in range(start, end):
            gold_chars.add(i)
    
    pred_chars = set()
    for pred_text in pred_spans:
        idx = response.find(pred_text)
        if idx != -1:
            for i in range(idx, idx + len(pred_text)):
                pred_chars.add(i)
    
    if not gold_chars and not pred_chars:
        return 1.0
    if not gold_chars or not pred_chars:
        return 0.0
    
    overlap = len(gold_chars & pred_chars)
    precision = overlap / len(pred_chars) if pred_chars else 0
    recall = overlap / len(gold_chars) if gold_chars else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def compute_span_level_metrics(records: List[Dict]) -> Dict:
    span_f1_scores = []
    for r in records:
        pred_spans = r.get("pred", [])
        gold_spans = r.get("labels", [])
        response = r.get("response", "")
        f1 = span_level_f1(pred_spans, gold_spans, response)
        span_f1_scores.append(f1)
    
    return {
        "mean_span_f1": round(np.mean(span_f1_scores), 4),
        "median_span_f1": round(np.median(span_f1_scores), 4),
        "std_span_f1": round(np.std(span_f1_scores), 4)
    }


def bootstrap_confidence_interval(records: List[Dict], n_iterations: int = 1000, confidence: float = 0.95) -> Dict:
    f1_scores = []
    n_samples = len(records)
    
    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = [records[i] for i in indices]
        metrics = case_level_metrics(bootstrap_sample)
        f1_scores.append(metrics["f1"])
    
    f1_scores = np.array(f1_scores)
    alpha = (1 - confidence) / 2
    lower = np.percentile(f1_scores, alpha * 100)
    upper = np.percentile(f1_scores, (1 - alpha) * 100)
    
    return {
        "mean_f1": round(np.mean(f1_scores), 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "confidence": confidence
    }


def compute_confusion_matrix(records: List[Dict]) -> np.ndarray:
    y_true = [1 if len(r["labels"]) > 0 else 0 for r in records]
    y_pred = [1 if len(r.get("pred", [])) > 0 else 0 for r in records]
    return confusion_matrix(y_true, y_pred)


def compute_classification_report(records: List[Dict]) -> str:
    y_true = [1 if len(r["labels"]) > 0 else 0 for r in records]
    y_pred = [1 if len(r.get("pred", [])) > 0 else 0 for r in records]
    return classification_report(y_true, y_pred, target_names=["No Hallucination", "Hallucination"])


def hallucination_type_breakdown(records: List[Dict]) -> Dict:
    type_counts = {
        "Evident Baseless Info": 0,
        "Evident Conflict": 0,
        "Subtle Baseless Info": 0,
        "Subtle Conflict": 0
    }
    
    for r in records:
        for label in r.get("labels", []):
            label_type = label.get("label_type", "")
            if label_type in type_counts:
                type_counts[label_type] += 1
    
    return type_counts


def full_evaluation_report(records: List[Dict], method_name: str = "Method") -> Dict:
    case_metrics = case_level_metrics(records)
    task_metrics = per_task_metrics(records)
    span_metrics = compute_span_level_metrics(records)
    bootstrap_ci = bootstrap_confidence_interval(records)
    conf_matrix = compute_confusion_matrix(records)
    class_report = compute_classification_report(records)
    type_breakdown = hallucination_type_breakdown(records)
    
    return {
        "method": method_name,
        "case_level": case_metrics,
        "per_task": task_metrics,
        "span_level": span_metrics,
        "bootstrap_ci": bootstrap_ci,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report,
        "hallucination_types": type_breakdown
    }
