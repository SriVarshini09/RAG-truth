import argparse
import json
import os
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


def save_report(report, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {output_path}")


def print_summary(report):
    print("\n" + "="*60)
    print(f"EVALUATION REPORT: {report['method']}")
    print("="*60)
    
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
    
    print("\n[PER-TASK BREAKDOWN]")
    for task, metrics in report['per_task'].items():
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
    
    print("\n[HALLUCINATION TYPE BREAKDOWN]")
    for htype, count in report['hallucination_types'].items():
        print(f"  {htype:<25} {count}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation runner for hallucination detection methods")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL file")
    parser.add_argument("--method", default="Method", help="Method name for the report")
    parser.add_argument("--output", help="Path to save JSON report (optional)")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}")
        sys.exit(1)
    
    print(f"Loading predictions from: {args.predictions}")
    records = load_jsonl(args.predictions)
    print(f"Loaded {len(records)} records")
    
    print(f"\nRunning full evaluation for: {args.method}")
    report = full_evaluation_report(records, method_name=args.method)
    
    if not args.quiet:
        print_summary(report)
    
    if args.output:
        save_report(report, args.output)
    
    return report


if __name__ == "__main__":
    main()
