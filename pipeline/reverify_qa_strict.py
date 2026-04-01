"""
QA Strict Re-Verification Script
Re-runs ONLY QA records from multi_agent_predictions_v3.jsonl using the
strict verification prompt (no background knowledge loophole).

Merges results back into multi_agent_predictions_v3.jsonl.

Usage:
    python pipeline/reverify_qa_strict.py
    python pipeline/reverify_qa_strict.py --workers 8
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.gpt_verifier import verify_claims_gpt
from agents.decision_aggregator import aggregate, format_output

RESULTS_DIR = ROOT / "results"
V3_FILE     = RESULTS_DIR / "multi_agent_predictions_v3.jsonl"
QA_FILE     = RESULTS_DIR / "multi_agent_predictions_v3_qa_strict.jsonl"


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def process_record(record: dict) -> dict:
    """Re-verify one QA record with strict prompt. Thread-safe."""
    claims = record.get("claims", [])
    reference = record.get("reference", "")
    if isinstance(reference, dict):
        reference = reference.get("passage", str(reference))

    verdicts = verify_claims_gpt(claims, str(reference), strict=True)
    response = record.get("response", "")
    aggregation = aggregate(verdicts, response)
    output = format_output(aggregation)

    result = dict(record)
    result["verdicts"] = verdicts
    result["aggregation"] = aggregation
    result["pred"] = output["hallucination list"]
    return result


def main(workers: int = 8):
    if not V3_FILE.exists():
        print(f"[ERROR] {V3_FILE} not found. Run reverify_gpt.py first.")
        return

    all_records = load_jsonl(V3_FILE)
    qa_records = [r for r in all_records if r.get("task_type") == "QA"]
    non_qa = [r for r in all_records if r.get("task_type") != "QA"]

    # Resume
    done_ids = set()
    if QA_FILE.exists():
        existing = load_jsonl(QA_FILE)
        done_ids = {r["id"] for r in existing}

    pending = [r for r in qa_records if r["id"] not in done_ids]

    print("=" * 65)
    print(f"QA Strict Re-Verification — {len(pending)} pending QA records")
    print(f"Workers: {workers}  |  Strict prompt: no background knowledge")
    print("=" * 65)

    lock = threading.Lock()
    counters = {"done": 0, "errors": 0}
    start = time.time()

    with open(QA_FILE, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_record, r): r for r in pending}
            for future in as_completed(futures):
                record = futures[future]
                try:
                    result = future.result()
                    with lock:
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush()
                        counters["done"] += 1
                        done = counters["done"]
                        total = len(pending)
                        if done % 25 == 0 or done == total:
                            elapsed = time.time() - start
                            rate = done / elapsed if elapsed > 0 else 0
                            eta = (total - done) / rate if rate > 0 else 0
                            print(f"  [{done}/{total}] errors={counters['errors']}"
                                  f"  rate={rate:.2f}/s  ETA={eta:.0f}s")
                except Exception as e:
                    with lock:
                        counters["errors"] += 1
                        print(f"  [ERROR] {record.get('id')}: {e}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")

    # Merge QA strict results back with non-QA records
    qa_strict = load_jsonl(QA_FILE)
    merged = non_qa + qa_strict

    # Save merged as v4
    v4_file = RESULTS_DIR / "multi_agent_predictions_v4.jsonl"
    with open(v4_file, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Merged {len(merged)} records -> {v4_file}")

    # Evaluate
    _quick_eval(merged, "v4 (strict QA + standard Summary/Data2txt)")


def _quick_eval(results: list, label: str = ""):
    from sklearn.metrics import precision_score, recall_score, f1_score
    y_true = [1 if len(r.get("labels", [])) > 0 else 0 for r in results]
    y_pred = [1 if len(r.get("pred", [])) > 0 else 0 for r in results]
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n=== Quick Evaluation {label} (n={len(results)}) ===")
    print(f"Overall  -> P:{p:.4f}  R:{r:.4f}  F1:{f:.4f}")
    for task in ["QA", "Summary", "Data2txt"]:
        sub = [x for x in results if x.get("task_type") == task]
        if sub:
            yt = [1 if len(x.get("labels", [])) > 0 else 0 for x in sub]
            yp = [1 if len(x.get("pred", [])) > 0 else 0 for x in sub]
            print(f"{task:<10} -> P:{precision_score(yt, yp, zero_division=0):.4f}"
                  f"  R:{recall_score(yt, yp, zero_division=0):.4f}"
                  f"  F1:{f1_score(yt, yp, zero_division=0):.4f}  (n={len(sub)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    main(workers=args.workers)
