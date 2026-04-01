"""
RAGTruth Data Pipeline
Loads, validates, and exports clean train/dev/test splits.
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATASET_DIR = ROOT / "RAGTruth" / "dataset"
SPLITS_DIR = ROOT / "data" / "splits"

REQUIRED_FIELDS = ["id", "source_id", "response", "labels", "split", "model"]
VALID_SPLITS = {"train", "dev", "test"}
VALID_TASKS = {"QA", "Summary", "Data2txt"}
VALID_LABEL_TYPES = {
    "Evident Conflict",
    "Evident Baseless Info",
    "Subtle Conflict",
    "Subtle Baseless Info",
}


# ─── Loaders ──────────────────────────────────────────────────────────────────

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


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_record(record: dict) -> list:
    """Returns list of issues. Empty list = valid."""
    issues = []

    for field in REQUIRED_FIELDS:
        if field not in record:
            issues.append(f"Missing field: {field}")

    if record.get("split") not in VALID_SPLITS:
        issues.append(f"Invalid split: {record.get('split')}")

    if record.get("task_type") and record.get("task_type") not in VALID_TASKS:
        issues.append(f"Invalid task_type: {record.get('task_type')}")

    for label in record.get("labels", []):
        start = label.get("start", 0)
        end = label.get("end", 0)
        if end < start:
            issues.append(f"Label span end ({end}) < start ({start})")
        if label.get("label_type") not in VALID_LABEL_TYPES:
            issues.append(f"Unknown label_type: {label.get('label_type')}")

    return issues


# ─── Join ─────────────────────────────────────────────────────────────────────

def build_full_records() -> list:
    """Join response.jsonl + source_info.jsonl into full records."""
    print("Loading response.jsonl ...")
    responses = load_jsonl(DATASET_DIR / "response.jsonl")

    print("Loading source_info.jsonl ...")
    sources_raw = load_jsonl(DATASET_DIR / "source_info.jsonl")
    source_map = {s["source_id"]: s for s in sources_raw}

    print("Joining on source_id ...")
    full_records = []
    skipped = 0

    for r in responses:
        sid = str(r.get("source_id", ""))
        src = source_map.get(sid)
        if src is None:
            skipped += 1
            continue

        record = {
            "id": r["id"],
            "source_id": sid,
            "model": r.get("model", ""),
            "split": r.get("split", ""),
            "task_type": src.get("task_type", ""),
            "source": src.get("source", ""),
            "reference": src.get("source_info", ""),
            "response": r.get("response", ""),
            "labels": r.get("labels", []),
            "quality": r.get("quality", ""),
        }

        # For QA tasks, extract question from source_info
        if record["task_type"] == "QA" and isinstance(record["reference"], dict):
            record["question"] = record["reference"].get("question", "")
            record["reference"] = record["reference"].get("passage", "")
        elif record["task_type"] == "QA" and isinstance(record["reference"], str):
            record["question"] = ""

        full_records.append(record)

    print(f"  Joined: {len(full_records)} records | Skipped (no source): {skipped}")
    return full_records


# ─── Validate All ─────────────────────────────────────────────────────────────

def validate_all(records: list) -> list:
    """Validate all records, print summary, return clean records."""
    valid, invalid = [], []
    issue_counts = Counter()

    for r in records:
        issues = validate_record(r)
        if issues:
            invalid.append((r["id"], issues))
            for issue in issues:
                issue_counts[issue] += 1
        else:
            valid.append(r)

    print(f"\nValidation: {len(valid)} valid | {len(invalid)} invalid")
    if issue_counts:
        print("  Issues found:")
        for issue, count in issue_counts.most_common(10):
            print(f"    [{count}] {issue}")

    return valid


# ─── Split ────────────────────────────────────────────────────────────────────

def create_splits(records: list) -> dict:
    """Separate records by official split field."""
    splits = defaultdict(list)
    for r in records:
        splits[r["split"]].append(r)
    return dict(splits)


def create_balanced_test(test_records: list, n_per_task: int = 200, seed: int = 42) -> list:
    """Sample n_per_task records per task type from test set."""
    rng = random.Random(seed)
    balanced = []
    for task in VALID_TASKS:
        subset = [r for r in test_records if r["task_type"] == task]
        if len(subset) < n_per_task:
            print(f"  WARNING: only {len(subset)} {task} records, wanted {n_per_task}")
            balanced.extend(subset)
        else:
            sampled = rng.sample(subset, n_per_task)
            balanced.extend(sampled)

    rng.shuffle(balanced)
    return balanced


# ─── Data Card ────────────────────────────────────────────────────────────────

def print_data_card(splits: dict, balanced_test: list) -> None:
    all_records = [r for records in splits.values() for r in records]

    print("\n" + "=" * 60)
    print("RAGTRUTH DATA CARD")
    print("=" * 60)

    print("\nTotal records: " + str(len(all_records)))
    print("\nSplits:")
    for split_name in ["train", "dev", "test"]:
        records = splits.get(split_name, [])
        print("  " + split_name + ": " + str(len(records)) + " records")
    print("  balanced_test: " + str(len(balanced_test)) + " records (200/task, seed=42)")

    print("\nTask distribution (all data):")
    task_counts = Counter(r["task_type"] for r in all_records)
    for task, count in sorted(task_counts.items()):
        pct = count / len(all_records) * 100
        print("  " + task + ": " + str(count) + " (" + str(round(pct, 1)) + "%)")

    print("\nHallucination rates (test set):")
    test_records = splits.get("test", [])
    for task in sorted(VALID_TASKS):
        subset = [r for r in test_records if r["task_type"] == task]
        halluc = sum(1 for r in subset if len(r["labels"]) > 0)
        rate = halluc / len(subset) * 100 if subset else 0
        print("  " + task + ": " + str(round(rate, 1)) + "% (" + str(halluc) + "/" + str(len(subset)) + ")")

    print("\nLabel type distribution (test set):")
    label_counts = Counter()
    for r in test_records:
        for label in r["labels"]:
            label_counts[label.get("label_type", "Unknown")] += 1
    for ltype, count in label_counts.most_common():
        print("  " + ltype + ": " + str(count))

    print("\nModel distribution:")
    model_counts = Counter(r["model"] for r in all_records)
    for model, count in model_counts.most_common():
        print("  " + model + ": " + str(count))

    print("\n" + "=" * 60)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_pipeline() -> dict:
    """
    Full pipeline: load → validate → split → export.
    Returns dict with split name → records list.
    """
    print("=" * 60)
    print("RAGTruth Data Pipeline")
    print("=" * 60)

    records = build_full_records()
    records = validate_all(records)

    splits = create_splits(records)
    balanced_test = create_balanced_test(splits.get("test", []), n_per_task=200, seed=42)

    print_data_card(splits, balanced_test)

    print("\nExporting splits ...")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(splits.get("train", []), SPLITS_DIR / "train.jsonl")
    save_jsonl(splits.get("dev", []), SPLITS_DIR / "dev.jsonl")
    save_jsonl(splits.get("test", []), SPLITS_DIR / "test_full.jsonl")
    save_jsonl(balanced_test, SPLITS_DIR / "test_balanced.jsonl")

    print(f"  Saved to: {SPLITS_DIR}/")
    print(f"    train.jsonl          → {len(splits.get('train', []))} records")
    print(f"    dev.jsonl            → {len(splits.get('dev', []))} records")
    print(f"    test_full.jsonl      → {len(splits.get('test', []))} records")
    print(f"    test_balanced.jsonl  → {len(balanced_test)} records")

    return {
        "train": splits.get("train", []),
        "dev": splits.get("dev", []),
        "test_full": splits.get("test", []),
        "test_balanced": balanced_test,
    }


if __name__ == "__main__":
    run_pipeline()
