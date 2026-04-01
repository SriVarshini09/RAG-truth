"""
Multi-Agent Hallucination Detection Pipeline
Orchestrates: Claim Extractor → Claim Verifier → Decision Aggregator
Runs on test_balanced.jsonl (600 samples, 200/task).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.claim_extractor import extract_claims
from agents.gpt_verifier import verify_claims_gpt
from agents.decision_aggregator import aggregate, format_output

SPLITS_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results"
TEST_FILE = SPLITS_DIR / "test_balanced.jsonl"
OUTPUT_FILE = RESULTS_DIR / "multi_agent_predictions.jsonl"


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_on_record(record: dict) -> dict:
    """
    Full pipeline on a single record.
    Returns record enriched with: claims, verdicts, aggregation, pred fields.
    """
    response = record.get("response", "")
    reference = record.get("reference", "")
    if isinstance(reference, dict):
        reference = reference.get("passage", str(reference))

    # Agent 1: Extract atomic claims (with task context if available)
    source_info = record.get("source_info", {})
    context = source_info.get("question") if isinstance(source_info, dict) else None
    claims = extract_claims(response, context=context)

    # Agent 2: Verify each claim against source using GPT-4o-mini
    verdicts = verify_claims_gpt(claims, str(reference))

    # Agent 3: Aggregate verdicts → binary label + spans
    aggregation = aggregate(verdicts, response)
    output = format_output(aggregation)

    result = dict(record)
    result["claims"] = claims
    result["verdicts"] = verdicts
    result["aggregation"] = aggregation
    result["pred"] = output["hallucination list"]

    return result


def run_pipeline(
    test_path: Path = TEST_FILE,
    output_path: Path = OUTPUT_FILE,
    limit: int = None,
    resume: bool = True,
) -> list:
    """
    Run the full multi-agent pipeline on test_balanced.jsonl.

    Args:
        limit:  Process only first N records (for quick testing).
        resume: Skip records already saved in output_path.
    """
    print("=" * 60)
    print("Multi-Agent Hallucination Detection Pipeline")
    print("=" * 60)

    records = load_jsonl(test_path)
    if limit:
        records = records[:limit]
    print(f"Loaded {len(records)} records from {test_path.name}")

    # Resume support: skip already-processed records
    done_ids = set()
    if resume and output_path.exists():
        existing = load_jsonl(output_path)
        done_ids = {r["id"] for r in existing}
        print(f"Resuming: {len(done_ids)} already done, {len(records) - len(done_ids)} remaining")

    results = []
    errors = 0
    start_time = time.time()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = open(output_path, "a" if resume else "w", encoding="utf-8")

    try:
        for i, record in enumerate(records):
            if record["id"] in done_ids:
                continue

            try:
                result = run_on_record(record)
                results.append(result)
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_file.flush()
            except Exception as e:
                print(f"  [ERROR] Record {record['id']}: {e}")
                errors += 1
                continue

            # Progress log every 10 records
            if (i + 1) % 10 == 0 or (i + 1) == len(records):
                elapsed = time.time() - start_time
                done = len(results)
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(records) - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{i+1}/{len(records)}] done={done}  "
                    f"errors={errors}  rate={rate:.1f}/s  ETA={eta:.0f}s"
                )

            # Small delay to avoid rate limits on GPT calls
            time.sleep(0.3)
    finally:
        out_file.close()

    elapsed = time.time() - start_time
    print(f"\nPipeline complete: {len(results)} processed, {errors} errors, {elapsed:.1f}s total")
    print(f"Saved to: {output_path}")
    return results


def quick_eval(predictions_file: Path = OUTPUT_FILE) -> None:
    """Quick case-level metrics on saved predictions."""
    if not predictions_file.exists():
        print("No predictions file found.")
        return

    records = load_jsonl(predictions_file)
    y_true = [1 if len(r.get("labels", [])) > 0 else 0 for r in records]
    y_pred = [1 if len(r.get("pred", [])) > 0 else 0 for r in records]

    from sklearn.metrics import precision_score, recall_score, f1_score
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n=== Quick Evaluation (n={len(records)}) ===")
    print(f"Overall  -> P:{p:.4f}  R:{r:.4f}  F1:{f:.4f}")

    for task in ["QA", "Summary", "Data2txt"]:
        subset = [rec for rec in records if rec.get("task_type") == task]
        if subset:
            yt = [1 if len(r.get("labels", [])) > 0 else 0 for r in subset]
            yp = [1 if len(r.get("pred", [])) > 0 else 0 for r in subset]
            tp = precision_score(yt, yp, zero_division=0)
            tr = recall_score(yt, yp, zero_division=0)
            tf = f1_score(yt, yp, zero_division=0)
            print(f"{task:<10} -> P:{tp:.4f}  R:{tr:.4f}  F1:{tf:.4f}  (n={len(subset)})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Multi-Agent Hallucination Detection Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N records")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (overwrite output)")
    parser.add_argument("--eval-only", action="store_true", help="Only run quick_eval on existing predictions")
    args = parser.parse_args()

    if args.eval_only:
        quick_eval()
    else:
        run_pipeline(limit=args.limit, resume=not args.no_resume)
        quick_eval()
