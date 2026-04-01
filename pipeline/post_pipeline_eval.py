"""
Post-Pipeline Evaluation
Run this after run_pipeline.py finishes to get full metrics,
log the run, and generate a comparison table.

Usage:
    python pipeline/post_pipeline_eval.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import full_evaluation_report
from logs.experiment_logger import ExperimentLogger

RESULTS_DIR = ROOT / "results"
PREDICTIONS_FILE = RESULTS_DIR / "multi_agent_predictions_final.jsonl"
GPT_PREDICTIONS_FILE = RESULTS_DIR / "gpt_baseline_predictions.jsonl"


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    print("=" * 70)
    print("Post-Pipeline Evaluation")
    print("=" * 70)

    if not PREDICTIONS_FILE.exists():
        print(f"[ERROR] Predictions file not found: {PREDICTIONS_FILE}")
        print("  Run: python pipeline/run_pipeline.py first.")
        return

    # ─── Load predictions ──────────────────────────────────────────────────
    ma_preds = load_jsonl(PREDICTIONS_FILE)
    print(f"Loaded {len(ma_preds)} multi-agent predictions")

    # ─── Full evaluation ───────────────────────────────────────────────────
    print("\nRunning full evaluation (bootstrap CI takes ~10s)...")
    t0 = time.time()
    report = full_evaluation_report(ma_preds, method_name="Multi-Agent Pipeline (GPT-4o-mini + GPT-4o-mini Verifier)")
    elapsed = time.time() - t0

    cl = report["case_level"]
    pt = report["per_task"]
    ci = report["bootstrap_ci"]
    sl = report["span_level"]

    print("\n" + "="*70)
    print("MULTI-AGENT PIPELINE RESULTS")
    print("="*70)
    print(f"  n_samples : {len(ma_preds)}")
    print(f"  Precision : {cl['precision']:.4f}")
    print(f"  Recall    : {cl['recall']:.4f}")
    print(f"  F1        : {cl['f1']:.4f}")
    print(f"  95% CI    : [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    print("\nPer-task F1:")
    for task in ["QA", "Summary", "Data2txt"]:
        m = pt.get(task, {})
        print(f"  {task:<10}: P={m.get('precision',0):.4f}  R={m.get('recall',0):.4f}  F1={m.get('f1',0):.4f}  (n={m.get('n_samples',0)})")
    print(f"\nSpan-level F1 : {sl['mean_span_f1']:.4f}")
    print(f"\nClassification Report:\n{report['classification_report']}")

    # ─── Comparison table ──────────────────────────────────────────────────
    print("="*70)
    print("COMPARISON TABLE (Track A, n=600)")
    print("="*70)

    # Official LLaMA-2-13B numbers
    llama = {
        "Method": "LLaMA-2-13B (official, supervised)",
        "F1": 0.7822, "P": 0.7380, "R": 0.8320,
        "QA": 0.7149, "Sum": 0.7341, "D2T": 0.8358,
    }
    gpt_row = None
    if GPT_PREDICTIONS_FILE.exists():
        gpt_preds = load_jsonl(GPT_PREDICTIONS_FILE)
        gr = full_evaluation_report(gpt_preds, "Self-Verification GPT-4o-mini")
        gpt_row = {
            "Method": "Self-Verification GPT-4o-mini",
            "F1": gr["case_level"]["f1"],
            "P": gr["case_level"]["precision"],
            "R": gr["case_level"]["recall"],
            "QA": gr["per_task"].get("QA", {}).get("f1", 0),
            "Sum": gr["per_task"].get("Summary", {}).get("f1", 0),
            "D2T": gr["per_task"].get("Data2txt", {}).get("f1", 0),
        }
    else:
        # Use CKPT2 reported numbers
        gpt_row = {
            "Method": "Self-Verification GPT-4o-mini (CKPT2)",
            "F1": 0.6791, "P": None, "R": None,
            "QA": 0.5349, "Sum": 0.4818, "D2T": 0.8308,
        }

    ma_row = {
        "Method": "Multi-Agent Pipeline (ours)",
        "F1": cl["f1"], "P": cl["precision"], "R": cl["recall"],
        "QA": pt.get("QA", {}).get("f1", 0),
        "Sum": pt.get("Summary", {}).get("f1", 0),
        "D2T": pt.get("Data2txt", {}).get("f1", 0),
    }

    hdr = f"{'Method':<42} {'F1':>6} {'P':>6} {'R':>6} {'QA':>6} {'Sum':>6} {'D2T':>6}"
    print(hdr)
    print("-" * len(hdr))
    for row in [llama, gpt_row, ma_row]:
        p_str = f"{row['P']:.4f}" if row["P"] else "  N/A "
        r_str = f"{row['R']:.4f}" if row["R"] else "  N/A "
        print(
            f"{row['Method']:<42} {row['F1']:>6.4f} {p_str:>6} {r_str:>6} "
            f"{row['QA']:>6.4f} {row['Sum']:>6.4f} {row['D2T']:>6.4f}"
        )

    # Improvement over GPT baseline
    delta = cl["f1"] - gpt_row["F1"]
    sign = "+" if delta >= 0 else ""
    print(f"\nDelta vs Self-Verification: {sign}{delta:.4f} F1 "
          f"({'improvement' if delta >= 0 else 'regression'})")

    # ─── Log to experiment logger ──────────────────────────────────────────
    logger = ExperimentLogger()
    logger.log_run(
        method="Multi-Agent Pipeline (GPT-4o-mini + GPT-4o-mini Verifier)",
        metrics={
            "n_samples": len(ma_preds),
            "overall_f1": cl["f1"],
            "overall_precision": cl["precision"],
            "overall_recall": cl["recall"],
            "per_task": {
                t: {"f1": pt.get(t, {}).get("f1", 0),
                    "precision": pt.get(t, {}).get("precision", 0),
                    "recall": pt.get(t, {}).get("recall", 0)}
                for t in ["QA", "Summary", "Data2txt"]
            },
            "bootstrap_ci": ci,
        },
        config={
            "model_extractor": "gpt-4o-mini",
            "model_verifier": "gpt-4o-mini",
            "baseless_threshold": 0.4,
            "workers": 8,
            "seed": 42,
        },
        predictions_file=str(PREDICTIONS_FILE),
        notes="GPT verifier (batched, 8 workers), bt=0.4, 600-sample balanced test",
    )

    print("\n[Logger] Run logged to logs/summary.csv")
    logger.print_summary()

    # ─── Save report JSON ──────────────────────────────────────────────────
    report_path = RESULTS_DIR / "multi_agent_eval_report.json"
    save_report = {k: v for k, v in report.items() if k != "confusion_matrix"}
    save_report["confusion_matrix"] = report["confusion_matrix"]
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(save_report, f, indent=2)
    print(f"\nFull report saved: {report_path}")
    print(f"Evaluation complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
