import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

def generate_answer(question, reference_text):
    prompt = f"""Answer the following question using ONLY the information from the provided passages. If the passages do not contain enough information to answer the question, say "I cannot answer based on the given passages."

Question: {question}

{reference_text}

Answer:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def main():
    test_path = "../RAGTruth/baseline/test.jsonl"
    output_path = "rag_baseline_answers.jsonl"
    n = 100
    
    all_records = load_jsonl(test_path)
    qa_records = [r for r in all_records if r.get("task_type") == "QA"][:n]
    
    print(f"Running RAG baseline on {len(qa_records)} QA test records...")
    
    results = []
    for i, record in enumerate(qa_records):
        question = record.get("question", "")
        reference_text = record.get("reference", "")
        gold_answer = record.get("response", "")
        
        generated_answer = generate_answer(question, reference_text)
        
        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "generated_answer": generated_answer,
            "model": record.get("model", ""),
            "source_id": record.get("source_id", "")
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(qa_records)}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
