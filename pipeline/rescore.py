"""
Re-scoring script: re-applies aggregation rules to v3 verdicts with tuned threshold.
No API calls needed. Saves final predictions to multi_agent_predictions_final.jsonl.

Usage:
    python pipeline/rescore.py --baseless-thresh 0.3
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.decision_aggregator import _map_claims_to_spans

RESULTS_DIR = ROOT / "results"
INPUT_FILE  = RESULTS_DIR / "multi_agent_predictions_v3.jsonl"
OUTPUT_FILE = RESULTS_DIR / "multi_agent_predictions_final.jsonl"


def load_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def reaggregate(verdicts: list, response: str, baseless_thresh: float) -> dict:
    """Re-apply aggregation with tuned BASELESS threshold."""
    if not verdicts:
        return {"hallucination": False, "hallucination_list": [],
                "verdict_counts": {}, "reason": "no_claims"}

    counts = {"ENTAILMENT": 0, "NEUTRAL": 0, "CONTRADICTION": 0, "BASELESS": 0}
    for v in verdicts:
        counts[v.get("label", "ENTAILMENT")] = counts.get(v.get("label", "ENTAILMENT"), 0) + 1

    total = len(verdicts)
    hallucinated_claims = []
    reason = "all_entailed"

    if counts["CONTRADICTION"] >= 1:
        hallucinated_claims = [v["claim"] for v in verdicts if v.get("label") == "CONTRADICTION"]
        reason = "contradiction"

    unsupported = counts["NEUTRAL"] + counts["BASELESS"]
    if not hallucinated_claims and unsupported / total >= baseless_thresh:
        hallucinated_claims = [v["claim"] for v in verdicts
                               if v.get("label") in ("NEUTRAL", "BASELESS")]
        reason = "baseless_info"

    spans = _map_claims_to_spans(hallucinated_claims, response)
    return {
        "hallucination": len(hallucinated_claims) > 0,
        "hallucination_list": spans,
        "verdict_counts": counts,
        "reason": reason,
    }


def score_records(records, bt_qa, bt_sum, bt_d2t):
    """Re-aggregate all records with per-task thresholds, return (results, metrics_dict)."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    task_thresh = {"QA": bt_qa, "Summary": bt_sum, "Data2txt": bt_d2t}
    results = []
    for record in records:
        task = record.get("task_type", "QA")
        bt = task_thresh.get(task, 0.4)
        verdicts = record.get("verdicts", [])
        response = record.get("response", "")
        agg = reaggregate(verdicts, response, bt)
        r = dict(record)
        r["aggregation"] = agg
        r["pred"] = agg["hallucination_list"]
        results.append(r)

    metrics = {}
    y_true = [1 if len(r.get("labels", [])) > 0 else 0 for r in results]
    y_pred = [1 if len(r.get("pred", [])) > 0 else 0 for r in results]
    metrics["overall_f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["overall_p"]  = precision_score(y_true, y_pred, zero_division=0)
    metrics["overall_r"]  = recall_score(y_true, y_pred, zero_division=0)
    for task in ["QA", "Summary", "Data2txt"]:
        sub = [x for x in results if x.get("task_type") == task]
        yt = [1 if len(x.get("labels", [])) > 0 else 0 for x in sub]
        yp = [1 if len(x.get("pred", [])) > 0 else 0 for x in sub]
        metrics[f"{task}_f1"] = f1_score(yt, yp, zero_division=0)
    return results, metrics


def main(baseless_thresh: float = 0.4, tune: bool = False):
    records = load_jsonl(INPUT_FILE)

    if tune:
        print("=== Per-Task Threshold Grid Search ===")
        print(f"{'QA_bt':>6} {'Sum_bt':>7} {'D2T_bt':>7}  "
              f"{'F1':>7} {'QA_F1':>7} {'Sum_F1':>7} {'D2T_F1':>7}")
        print("-" * 60)

        qa_vals  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
        sum_vals = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
        d2t_vals = [0.30, 0.35, 0.40, 0.45, 0.50]

        best_f1 = 0.0
        best_combo = (0.4, 0.4, 0.4)
        best_metrics = {}

        for bt_qa in qa_vals:
            for bt_sum in sum_vals:
                for bt_d2t in d2t_vals:
                    _, m = score_records(records, bt_qa, bt_sum, bt_d2t)
                    if m["overall_f1"] > best_f1:
                        best_f1 = m["overall_f1"]
                        best_combo = (bt_qa, bt_sum, bt_d2t)
                        best_metrics = m

        # Print top result
        bt_qa, bt_sum, bt_d2t = best_combo
        m = best_metrics
        print(f"{bt_qa:>6.2f} {bt_sum:>7.2f} {bt_d2t:>7.2f}  "
              f"{m['overall_f1']:>7.4f} {m['QA_f1']:>7.4f} "
              f"{m['Summary_f1']:>7.4f} {m['Data2txt_f1']:>7.4f}  <- BEST")

        print(f"\nBest combo: QA={bt_qa}  Summary={bt_sum}  Data2txt={bt_d2t}")
        print(f"Overall F1: {best_f1:.4f}  "
              f"QA={m['QA_f1']:.4f}  Sum={m['Summary_f1']:.4f}  D2T={m['Data2txt_f1']:.4f}")

        # Compare vs global threshold baseline
        _, base = score_records(records, 0.4, 0.4, 0.4)
        delta = best_f1 - base["overall_f1"]
        print(f"\nvs global bt=0.4 baseline: F1={base['overall_f1']:.4f}  "
              f"delta={delta:+.4f}")

        # Save tuned predictions
        tuned_path = RESULTS_DIR / "multi_agent_predictions_tuned.jsonl"
        best_results, _ = score_records(records, bt_qa, bt_sum, bt_d2t)
        with open(tuned_path, "w", encoding="utf-8") as f:
            for r in best_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nSaved tuned predictions -> {tuned_path}")
        return

    # Single-threshold mode (original behaviour)
    results, metrics = score_records(records, baseless_thresh,
                                     baseless_thresh, baseless_thresh)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} records -> {OUTPUT_FILE}")
    print(f"\n=== Results (baseless_thresh={baseless_thresh}) ===")
    print(f"Overall  -> P:{metrics['overall_p']:.4f}  "
          f"R:{metrics['overall_r']:.4f}  F1:{metrics['overall_f1']:.4f}")
    for task in ["QA", "Summary", "Data2txt"]:
        sub = [x for x in results if x.get("task_type") == task]
        from sklearn.metrics import precision_score, recall_score, f1_score
        yt = [1 if len(x.get("labels", [])) > 0 else 0 for x in sub]
        yp = [1 if len(x.get("pred", [])) > 0 else 0 for x in sub]
        print(f"{task:<10} -> P:{precision_score(yt,yp,zero_division=0):.4f}"
              f"  R:{recall_score(yt,yp,zero_division=0):.4f}"
              f"  F1:{f1_score(yt,yp,zero_division=0):.4f}  (n={len(sub)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseless-thresh", type=float, default=0.4)
    parser.add_argument("--tune", action="store_true",
                        help="Grid search per-task thresholds independently")
    args = parser.parse_args()
    main(baseless_thresh=args.baseless_thresh, tune=args.tune)
