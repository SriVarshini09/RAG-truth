"""
GPT Re-Verification Script (concurrent)
Reuses extracted claims from multi_agent_predictions.jsonl (no re-extraction needed)
and re-verifies each claim using GPT-4o-mini with 8 concurrent workers.

Runtime: ~20 min for 600 records (vs ~2 h sequential).

Usage:
    python pipeline/reverify_gpt.py
    python pipeline/reverify_gpt.py --limit 50      # test on first 50
    python pipeline/reverify_gpt.py --workers 8     # default workers
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
INPUT_FILE  = RESULTS_DIR / "multi_agent_predictions.jsonl"
OUTPUT_FILE = RESULTS_DIR / "multi_agent_predictions_v3.jsonl"


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def process_record(record: dict) -> dict:
    """Process one record: GPT verify + aggregate. Thread-safe (no shared state)."""
    claims = record.get("claims", [])
    reference = record.get("reference", "")
    if isinstance(reference, dict):
        reference = reference.get("passage", str(reference))

    verdicts = verify_claims_gpt(claims, str(reference))
    response = record.get("response", "")
    aggregation = aggregate(verdicts, response)
    output = format_output(aggregation)

    result = dict(record)
    result["verdicts"] = verdicts
    result["aggregation"] = aggregation
    result["pred"] = output["hallucination list"]
    return result


def main(limit: int = None, workers: int = 8):
    if not INPUT_FILE.exists():
        print(f"[ERROR] {INPUT_FILE} not found. Run run_pipeline.py first.")
        return

    records = load_jsonl(INPUT_FILE)
    if limit:
        records = records[:limit]

    # Resume: skip already-done records
    done_ids = set()
    if OUTPUT_FILE.exists() and not limit:
        existing = load_jsonl(OUTPUT_FILE)
        done_ids = {r["id"] for r in existing}

    pending = [r for r in records if r["id"] not in done_ids]

    print("=" * 65)
    print(f"GPT Re-Verification — {len(pending)} pending  ({len(done_ids)} already done)")
    print(f"Workers: {workers}  |  Reusing saved claims (no GPT extraction)")
    print("=" * 65)

    if not pending:
        print("Nothing to do.")
        _quick_eval(load_jsonl(OUTPUT_FILE))
        return

    # Thread-safe counters and file handle
    lock = threading.Lock()
    counters = {"done": 0, "errors": 0}
    start = time.time()

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
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
    print(f"\nDone in {elapsed:.1f}s  ->  {OUTPUT_FILE}")

    all_results = load_jsonl(OUTPUT_FILE)
    _quick_eval(all_results)


def _quick_eval(results: list):
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
        if sub:
            yt = [1 if len(x.get("labels", [])) > 0 else 0 for x in sub]
            yp = [1 if len(x.get("pred", [])) > 0 else 0 for x in sub]
            print(f"{task:<10} -> P:{precision_score(yt, yp, zero_division=0):.4f}"
                  f"  R:{recall_score(yt, yp, zero_division=0):.4f}"
                  f"  F1:{f1_score(yt, yp, zero_division=0):.4f}  (n={len(sub)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    main(limit=args.limit, workers=args.workers)
