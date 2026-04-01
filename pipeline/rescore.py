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


def main(baseless_thresh: float = 0.3):
    records = load_jsonl(INPUT_FILE)
    results = []

    for record in records:
        verdicts = record.get("verdicts", [])
        response = record.get("response", "")
        agg = reaggregate(verdicts, response, baseless_thresh)
        r = dict(record)
        r["aggregation"] = agg
        r["pred"] = agg["hallucination_list"]
        results.append(r)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} records -> {OUTPUT_FILE}")

    from sklearn.metrics import precision_score, recall_score, f1_score
    y_true = [1 if len(r.get("labels", [])) > 0 else 0 for r in results]
    y_pred = [1 if len(r.get("pred", [])) > 0 else 0 for r in results]
    p = precision_score(y_true, y_pred, zero_division=0)
    r2 = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n=== Final Results (baseless_thresh={baseless_thresh}) ===")
    print(f"Overall  -> P:{p:.4f}  R:{r2:.4f}  F1:{f:.4f}")
    for task in ["QA", "Summary", "Data2txt"]:
        sub = [x for x in results if x.get("task_type") == task]
        if sub:
            yt = [1 if len(x.get("labels", [])) > 0 else 0 for x in sub]
            yp = [1 if len(x.get("pred", [])) > 0 else 0 for x in sub]
            print(f"{task:<10} -> P:{precision_score(yt,yp,zero_division=0):.4f}"
                  f"  R:{recall_score(yt,yp,zero_division=0):.4f}"
                  f"  F1:{f1_score(yt,yp,zero_division=0):.4f}  (n={len(sub)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseless-thresh", type=float, default=0.3)
    args = parser.parse_args()
    main(baseless_thresh=args.baseless_thresh)
