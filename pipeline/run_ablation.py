"""
Ablation Study Runner
Tests 4 configurations to isolate each agent's contribution.

Config A: Claim Extractor only (any claim found → hallucination, no verification)
Config B: NLI on full response only (no claim decomposition) — uses DeBERTa local
Config C: Claims + GPT Verifier, majority vote (no custom aggregation rules)
Config D: Full pipeline (claims + GPT verifier + custom rules) — final system

Reuses saved claims/verdicts from v3 file to avoid GPT re-extraction.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.claim_verifier import verify_claim   # DeBERTa, for Config B only
from agents.decision_aggregator import aggregate, format_output

SPLITS_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results" / "ablation"
V3_FILE = ROOT / "results" / "multi_agent_predictions_v3.jsonl"
FINAL_FILE = ROOT / "results" / "multi_agent_predictions_final.jsonl"
TEST_FILE = SPLITS_DIR / "test_balanced.jsonl"


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_v3_by_id() -> dict:
    """Load v3 predictions indexed by record ID for fast lookup."""
    if not V3_FILE.exists():
        return {}
    with open(V3_FILE, "r", encoding="utf-8") as f:
        return {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}


_V3 = None


def _get_v3():
    global _V3
    if _V3 is None:
        _V3 = _load_v3_by_id()
    return _V3


# ─── Config A: Claim Extractor Only ──────────────────────────────────────────

def config_a_claim_only(record: dict) -> dict:
    """Use saved claims. Any non-empty claims → hallucination (no verification)."""
    v3 = _get_v3().get(record["id"], record)
    claims = v3.get("claims", [])
    pred = claims[:3] if claims else []
    result = dict(record)
    result["ablation_config"] = "A_claim_only"
    result["claims"] = claims
    result["pred"] = pred
    return result


# ─── Config B: NLI on Full Response (no decomposition) ───────────────────────

def config_b_nli_full(record: dict) -> dict:
    """DeBERTa NLI on full response truncated to 500 chars — no claim extraction."""
    response = record.get("response", "")
    reference = record.get("reference", "")
    if isinstance(reference, dict):
        reference = reference.get("passage", str(reference))

    verdict = verify_claim(response[:500], str(reference))
    is_hallucinated = verdict["label"] in ("CONTRADICTION", "NEUTRAL")
    pred = [response[:100]] if is_hallucinated else []

    result = dict(record)
    result["ablation_config"] = "B_nli_full"
    result["verdicts"] = [verdict]
    result["pred"] = pred
    return result


# ─── Config C: Claims + GPT Verifier, Majority Vote (no custom rules) ────────

def config_c_no_rules(record: dict) -> dict:
    """Reuse saved GPT verdicts (v3). Predict hallucination by simple majority vote."""
    v3 = _get_v3().get(record["id"], record)
    verdicts = v3.get("verdicts", [])
    response = record.get("response", "")

    if not verdicts:
        pred = []
    else:
        non_entailed = [v for v in verdicts if v.get("label") != "ENTAILMENT"]
        if len(non_entailed) / len(verdicts) > 0.5:
            pred = [v["claim"] for v in non_entailed[:3]]
        else:
            pred = []

    result = dict(record)
    result["ablation_config"] = "C_no_rules"
    result["claims"] = v3.get("claims", [])
    result["verdicts"] = verdicts
    result["pred"] = pred
    return result


# ─── Config D: Full Pipeline (final system) ───────────────────────────────────

def config_d_full(record: dict) -> dict:
    """Reuse final predictions (GPT verifier + bt=0.4 aggregation)."""
    if FINAL_FILE.exists():
        # Load lazily on first use
        if not hasattr(config_d_full, "_cache"):
            with open(FINAL_FILE, "r", encoding="utf-8") as f:
                config_d_full._cache = {json.loads(l)["id"]: json.loads(l)
                                        for l in f if l.strip()}
        final_rec = config_d_full._cache.get(record["id"], record)
        result = dict(record)
        result["ablation_config"] = "D_full_pipeline"
        result["claims"] = final_rec.get("claims", [])
        result["verdicts"] = final_rec.get("verdicts", [])
        result["aggregation"] = final_rec.get("aggregation", {})
        result["pred"] = final_rec.get("pred", [])
        return result

    # Fallback: run fresh (should not be needed if final file exists)
    v3 = _get_v3().get(record["id"], record)
    verdicts = v3.get("verdicts", [])
    response = record.get("response", "")
    aggregation = aggregate(verdicts, response, neutral_threshold=0.4)
    output = format_output(aggregation)
    result = dict(record)
    result["ablation_config"] = "D_full_pipeline"
    result["claims"] = v3.get("claims", [])
    result["verdicts"] = verdicts
    result["aggregation"] = aggregation
    result["pred"] = output["hallucination list"]
    return result


# ─── Runner ──────────────────────────────────────────────────────────────────

CONFIGS = {
    "A_claim_only": config_a_claim_only,
    "B_nli_full": config_b_nli_full,
    "C_no_rules": config_c_no_rules,
    "D_full_pipeline": config_d_full,
}


def run_ablation(config_name: str, limit: int = None):
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(CONFIGS)}")

    fn = CONFIGS[config_name]
    records = load_jsonl(TEST_FILE)
    if limit:
        records = records[:limit]

    output_path = RESULTS_DIR / f"ablation_{config_name}.jsonl"
    print(f"\n{'='*60}")
    print(f"Ablation Config: {config_name} (n={len(records)})")
    print(f"{'='*60}")

    results = []
    errors = 0
    start = time.time()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            try:
                result = fn(record)
                results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"  [ERROR] {record.get('id')}: {e}")
                errors += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(records):
                elapsed = time.time() - start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{len(records)}] errors={errors} rate={rate:.1f}/s")

            time.sleep(0.2)

    elapsed = time.time() - start
    quick_eval(results, config_name)
    print(f"\nSaved: {output_path}  ({elapsed:.1f}s)")
    return results


def quick_eval(records, config_name=""):
    from sklearn.metrics import precision_score, recall_score, f1_score
    y_true = [1 if len(r.get("labels", [])) > 0 else 0 for r in records]
    y_pred = [1 if len(r.get("pred", [])) > 0 else 0 for r in records]
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n[{config_name}] Overall -> P:{p:.4f}  R:{r:.4f}  F1:{f:.4f}")
    for task in ["QA", "Summary", "Data2txt"]:
        sub = [x for x in records if x.get("task_type") == task]
        if sub:
            yt = [1 if len(x.get("labels", [])) > 0 else 0 for x in sub]
            yp = [1 if len(x.get("pred", [])) > 0 else 0 for x in sub]
            tf = f1_score(yt, yp, zero_division=0)
            print(f"  {task:<10} F1:{tf:.4f}  (n={len(sub)})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS), default="D_full_pipeline")
    parser.add_argument("--all", action="store_true", help="Run all 4 configs sequentially")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.all:
        for name in CONFIGS:
            run_ablation(name, limit=args.limit)
    else:
        run_ablation(args.config, limit=args.limit)
