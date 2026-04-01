"""
Agent 1: Claim Extractor
Decomposes an LLM response into atomic factual claims using GPT-4o-mini.
"""
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "You are a precise fact-checker. Your job is to decompose a response "
    "into a list of simple, atomic factual claims — one fact per claim. "
    "Each claim must be a complete sentence. "
    "Return ONLY a JSON object with key \"claims\" and a list of strings. "
    "Example: {\"claims\": [\"Einstein was born in 1879.\", \"He developed the theory of relativity.\"]}"
)

EXTRACTION_TEMPLATE = (
    "Decompose the following response into atomic factual claims.\n\n"
    "Response:\n{response}\n\n"
    "Return JSON: {{\"claims\": [\"claim1\", \"claim2\", ...]}}"
)


def extract_claims(response: str, max_retries: int = 3) -> list:
    """
    Extract atomic claims from a response string.
    Returns list of claim strings.
    """
    prompt = EXTRACTION_TEMPLATE.format(response=response[:3000])

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content.strip()
            parsed = json.loads(content)
            claims = parsed.get("claims", [])
            if isinstance(claims, list):
                return [str(c) for c in claims if c]
            return []
        except json.JSONDecodeError:
            return []
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"[ClaimExtractor] Error: {e}")
                return []
    return []


def extract_claims_for_record(record: dict) -> dict:
    """
    Run claim extraction on a single RAGTruth record.
    Returns record enriched with 'claims' field.
    """
    response = record.get("response", "")
    claims = extract_claims(response)
    return {**record, "claims": claims}


if __name__ == "__main__":
    test_response = (
        "Einstein was born in Germany in 1879. He developed the special theory "
        "of relativity in 1905 while working at the patent office in Bern, Switzerland. "
        "He received the Nobel Prize in Physics in 1922 for his discovery of the law "
        "of the photoelectric effect."
    )
    claims = extract_claims(test_response)
    print(f"Extracted {len(claims)} claims:")
    for i, claim in enumerate(claims, 1):
        print(f"  {i}. {claim}")
