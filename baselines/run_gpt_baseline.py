import json
import os
import sys
import time
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TEMPLATES = {
    "QA": (
        "Below is a question:\n"
        "{question}\n\n"
        "Below are related passages:\n"
        "{reference}\n\n"
        "Below is an answer:\n"
        "{response}\n\n"
        "Your task is to determine whether the answer contains either or both of the following two types of hallucinations:\n"
        "1. conflict: instances where the answer presents direct contradiction or opposition to the passages;\n"
        "2. baseless info: instances where the answer includes information which is not substantiated by or inferred from the passages.\n"
        "Then, compile the labeled hallucinated spans into a JSON dict, with a key \"hallucination list\" and its value is a list of hallucinated spans. "
        'If there exist potential hallucinations, output: {{"hallucination list": [span1, span2, ...]}}. '
        'Otherwise: {{"hallucination list": []}}.\n'
        "Output:"
    ),
    "Summary": (
        "Below is the original news:\n"
        "{reference}\n\n"
        "Below is a summary of the news:\n"
        "{response}\n"
        "Your task is to determine whether the summary contains either or both of the following two types of hallucinations:\n"
        "1. conflict: instances where the summary presents direct contradiction or opposition to the original news;\n"
        "2. baseless info: instances where the summary includes information which is not substantiated by or inferred from the original news.\n"
        "Then, compile the labeled hallucinated spans into a JSON dict, with a key \"hallucination list\" and its value is a list of hallucinated spans. "
        'If there exist potential hallucinations, output: {{"hallucination list": [span1, span2, ...]}}. '
        'Otherwise: {{"hallucination list": []}}.\n'
        "Output:"
    ),
    "Data2txt": (
        "Below is a structured data in the JSON format:\n"
        "{reference}\n\n"
        "Below is an overview article written in accordance with the structured data:\n"
        "{response}\n\n"
        "Your task is to determine whether the overview contains either or both of the following two types of hallucinations:\n"
        "1. conflict: instances where the overview presents direct contradiction or opposition to the structured data;\n"
        "2. baseless info: instances where the overview includes information which is not substantiated by or inferred from the structured data.\n"
        "Then, compile the labeled hallucinated spans into a JSON dict, with a key \"hallucination list\" and its value is a list of hallucinated spans. "
        'If there exist potential hallucinations, output: {{"hallucination list": [span1, span2, ...]}}. '
        'Otherwise: {{"hallucination list": []}}.\n'
        "Output:"
    ),
}


def build_prompt(record):
    task = record.get("task_type", "Summary")
    response = record.get("response", "")
    reference = str(record.get("reference", ""))[:3000]
    question = record.get("question", "")
    if task == "QA":
        return TEMPLATES["QA"].format(question=question, reference=reference, response=response)
    return TEMPLATES.get(task, TEMPLATES["Summary"]).format(reference=reference, response=response)


def predict(record):
    prompt = build_prompt(record)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            content = resp.choices[0].message.content.strip()
            parsed = json.loads(content)
            return parsed.get("hallucination list", [])
        except json.JSONDecodeError:
            return []
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"  error: {e}")
                return []
    return []


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compute_metrics(records):
    y_true = [1 if len(r["labels"]) > 0 else 0 for r in records]
    y_pred = [1 if len(r.get("pred", [])) > 0 else 0 for r in records]
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    return round(p, 4), round(r, 4), round(f, 4)


def main(n=100):
    test_path = os.path.join(os.path.dirname(__file__), "..", "RAGTruth", "baseline", "test.jsonl")
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "gpt_baseline_predictions.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_records = load_jsonl(test_path)
    per_task = n // 3
    records = []
    for task in ["QA", "Summary", "Data2txt"]:
        subset = [r for r in all_records if r.get("task_type") == task][:per_task]
        records.extend(subset)
    print(f"Running GPT-4o-mini on {len(records)} test records (balanced across tasks)...")

    results = []
    for i, record in enumerate(records):
        pred = predict(record)
        result = dict(record)
        result["pred"] = pred
        results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(records)}")
        time.sleep(0.2)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved predictions to {output_path}")
    print("\n=== GPT-4o-mini Baseline Results (n={}) ===".format(len(results)))

    p, r, f = compute_metrics(results)
    print(f"Overall  ->  Precision: {p}  Recall: {r}  F1: {f}")

    for task in ["QA", "Summary", "Data2txt"]:
        subset = [x for x in results if x.get("task_type") == task]
        if subset:
            p, r, f = compute_metrics(subset)
            print(f"{task:<10} ->  Precision: {p}  Recall: {r}  F1: {f}  (n={len(subset)})")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(n)
