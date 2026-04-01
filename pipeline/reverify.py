"""
Re-Verification Script
Reuses extracted claims from multi_agent_predictions.jsonl (no GPT calls)
and re-runs NLI with the new sentence-level reference chunking approach.

Usage:
    python pipeline/reverify.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.claim_verifier import verify_claims
from agents.decision_aggregator import aggregate, format_output

RESULTS_DIR = ROOT / "results"
INPUT_FILE = RESULTS_DIR / "multi_agent_predictions.jsonl"
OUTPUT_FILE = RESULTS_DIR / "multi_agent_predictions_v2.jsonl"


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    if not INPUT_FILE.exists():
        print(f"[ERROR] {INPUT_FILE} not found. Run run_pipeline.py first.")
        return

    records = load_jsonl(INPUT_FILE)
    print("=" * 60)
    print(f"Re-Verification (sentence-level NLI) — n={len(records)}")
    print("=" * 60)
    print("Reusing saved claims. No GPT calls needed.")

    results = []
    errors = 0
    start = time.time()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for i, record in enumerate(records):
            try:
                claims = record.get("claims", [])
                reference = record.get("reference", "")
                if isinstance(reference, dict):
                    reference = reference.get("passage", str(reference))

                # Re-run NLI with sentence-level chunking
                verdicts = verify_claims(claims, str(reference))

                # Re-apply aggregation
                response = record.get("response", "")
                aggregation = aggregate(verdicts, response)
                output = format_output(aggregation)

                result = dict(record)
                result["verdicts"] = verdicts
                result["aggregation"] = aggregation
                result["pred"] = output["hallucination list"]

                results.append(result)
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"  [ERROR] {record.get('id')}: {e}")
                errors += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(records):
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta = (len(records) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(records)}] errors={errors}  rate={rate:.1f}/s  ETA={eta:.0f}s")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Saved: {OUTPUT_FILE}")

    # Quick eval
    from sklearn.metrics import precision_score, recall_score, f1_score
    y_true = [1 if len(r.get("labels", [])) > 0 else 0 for r in results]
    y_pred = [1 if len(r.get("pred", [])) > 0 else 0 for r in results]
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n=== Quick Evaluation (n={len(results)}) ===")
    print(f"Overall  -> P:{p:.4f}  R:{r:.4f}  F1:{f:.4f}")
    for task in ["QA", "Summary", "Data2txt"]:
        sub = [x for x in results if x.get("task_type") == task]
        yt = [1 if len(x.get("labels", [])) > 0 else 0 for x in sub]
        yp = [1 if len(x.get("pred", [])) > 0 else 0 for x in sub]
        tf = f1_score(yt, yp, zero_division=0)
        tp = precision_score(yt, yp, zero_division=0)
        tr = recall_score(yt, yp, zero_division=0)
        print(f"{task:<10} -> P:{tp:.4f}  R:{tr:.4f}  F1:{tf:.4f}  (n={len(sub)})")


if __name__ == "__main__":
    main()
