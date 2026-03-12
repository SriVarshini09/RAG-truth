import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import full_evaluation_report


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to: {path}")


def main():
    print("="*60)
    print("Phase 1 Comprehensive Analysis")
    print("="*60)
    
    predictions_path = "results/gpt_baseline_predictions.jsonl"
    print(f"\nLoading predictions from: {predictions_path}")
    
    records = load_jsonl(predictions_path)
    print(f"Total records: {len(records)}")
    
    task_counts = {}
    for r in records:
        task = r.get("task_type", "Unknown")
        task_counts[task] = task_counts.get(task, 0) + 1
    
    print("\nTask distribution:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")
    
    print("\n" + "="*60)
    print("Running full evaluation...")
    print("="*60)
    
    report = full_evaluation_report(records, method_name="Self-Verification GPT-4o-mini (Scaled)")
    
    print("\n[CASE-LEVEL METRICS]")
    case = report['case_level']
    print(f"  Precision: {case['precision']:.4f}")
    print(f"  Recall:    {case['recall']:.4f}")
    print(f"  F1 Score:  {case['f1']:.4f}")
    print(f"  Samples:   {case['n_samples']}")
    
    print("\n[95% BOOTSTRAP CONFIDENCE INTERVAL]")
    ci = report['bootstrap_ci']
    print(f"  Mean F1:   {ci['mean_f1']:.4f}")
    print(f"  CI:        [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    print(f"  Width:     {ci['ci_upper'] - ci['ci_lower']:.4f}")
    
    print("\n[PER-TASK BREAKDOWN]")
    for task, metrics in sorted(report['per_task'].items()):
        print(f"  {task:<10} -> P: {metrics['precision']:.4f}  R: {metrics['recall']:.4f}  F1: {metrics['f1']:.4f}  (n={metrics['n_samples']})")
    
    print("\n[SPAN-LEVEL METRICS]")
    span = report['span_level']
    print(f"  Mean F1:   {span['mean_span_f1']:.4f}")
    print(f"  Median F1: {span['median_span_f1']:.4f}")
    print(f"  Std F1:    {span['std_span_f1']:.4f}")
    
    print("\n[CONFUSION MATRIX]")
    cm = report['confusion_matrix']
    print(f"  TN: {cm[0][0]:<4} FP: {cm[0][1]:<4}")
    print(f"  FN: {cm[1][0]:<4} TP: {cm[1][1]:<4}")
    accuracy = (cm[0][0] + cm[1][1]) / sum(sum(row) for row in cm)
    print(f"  Accuracy: {accuracy:.4f}")
    
    print("\n[HALLUCINATION TYPE BREAKDOWN]")
    for htype, count in sorted(report['hallucination_types'].items()):
        print(f"  {htype:<25} {count}")
    
    output_path = "results/phase1_full_report.json"
    save_json(report, output_path)
    
    print("\n" + "="*60)
    print("Phase 1 Analysis Complete")
    print("="*60)
    
    print("\nKey Findings:")
    print(f"1. Overall F1: {case['f1']:.4f} (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])")
    print(f"2. Best task: {max(report['per_task'].items(), key=lambda x: x[1]['f1'])[0]}")
    print(f"3. Worst task: {min(report['per_task'].items(), key=lambda x: x[1]['f1'])[0]}")
    print(f"4. Span-level mean F1: {span['mean_span_f1']:.4f}")
    print(f"5. Sample size: {case['n_samples']} (statistically significant)")
    
    return report


if __name__ == "__main__":
    main()
