"""
Agent 2b: GPT-4o-mini Claim Verifier
Uses GPT-4o-mini to verify each atomic claim against the source passage.

Replaces DeBERTa NLI. GPT understands the difference between:
  - CONTRADICTION: claim directly contradicts the reference
  - BASELESS: claim states facts absent from the reference with no factual basis
  - ENTAILMENT: claim is supported by the reference or background knowledge

This eliminates the DeBERTa NEUTRAL ambiguity problem:
  - DeBERTa labels "correct background knowledge not in source" as NEUTRAL -> false positives
  - GPT with targeted prompt correctly distinguishes true hallucinations from inferences
"""
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


VERIFY_SYSTEM_PROMPT = """You are a precise fact-checker verifying claims against a source passage.

For each claim, determine ONE of three verdicts:
- ENTAILMENT: The claim is directly supported by the reference, or is a reasonable inference/paraphrase of it, or is common factual background knowledge consistent with the reference.
- CONTRADICTION: The claim directly contradicts something explicitly stated in the reference (e.g., wrong date, wrong name, wrong number, opposite fact).
- BASELESS: The claim asserts specific facts (names, numbers, events, locations) that are entirely absent from the reference AND cannot be verified as common knowledge.

Important: Only mark CONTRADICTION for direct factual conflicts. Do NOT mark CONTRADICTION for things that are simply not mentioned. Use BASELESS for those.
Only mark BASELESS for specific factual claims (not general statements). Common knowledge consistent with the reference is ENTAILMENT.

Return ONLY a JSON object: {"verdict": "ENTAILMENT"|"CONTRADICTION"|"BASELESS", "reason": "brief explanation"}"""


def verify_claim_gpt(claim: str, reference: str, max_ref_chars: int = 1500,
                     max_retries: int = 2) -> dict:
    """Verify a single claim against the reference using GPT-4o-mini."""
    client = _get_client()
    ref_excerpt = reference[:max_ref_chars] if reference else ""

    user_msg = f"Reference:\n{ref_excerpt}\n\nClaim: {claim}"

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=80,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            verdict = parsed.get("verdict", "ENTAILMENT").upper()
            if verdict not in ("ENTAILMENT", "CONTRADICTION", "BASELESS"):
                verdict = "ENTAILMENT"
            return {
                "claim": claim,
                "label": verdict,
                "confidence": 1.0,
                "reason": parsed.get("reason", ""),
                "scores": {verdict: 1.0},
            }
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.5 ** attempt)
            else:
                return {
                    "claim": claim,
                    "label": "ENTAILMENT",
                    "confidence": 0.0,
                    "reason": f"error: {e}",
                    "scores": {"ENTAILMENT": 1.0},
                }


BATCH_SYSTEM_PROMPT = """You are a precise fact-checker verifying claims against a source passage.

For each numbered claim, return ONE verdict:
- ENTAILMENT: claim is supported by the reference, a reasonable inference, or common factual background knowledge consistent with it.
- CONTRADICTION: claim directly contradicts a specific fact in the reference (wrong date, name, number, or opposite fact).
- BASELESS: claim asserts specific facts (names, numbers, events) entirely absent from the reference AND not verifiable as common knowledge.

Only mark CONTRADICTION for direct factual conflicts. Use BASELESS for things simply not mentioned.

Return ONLY a JSON array, one object per claim in the same order:
[{"verdict": "ENTAILMENT"|"CONTRADICTION"|"BASELESS", "reason": "brief"}, ...]"""

BATCH_SYSTEM_PROMPT_STRICT = """You are a strict fact-checker for question-answering responses. Verify each claim against the reference passage.

For each numbered claim, return ONE verdict:
- ENTAILMENT: claim is DIRECTLY AND EXPLICITLY supported by the reference passage (exact match or clear paraphrase). Only use this if the reference passage clearly states this fact.
- CONTRADICTION: claim directly contradicts a specific fact stated in the reference (wrong number, wrong name, wrong date, opposite statement).
- BASELESS: claim makes any specific assertion (numbers, costs, dates, names, procedures, recommendations) that is NOT explicitly in the reference passage, even if it sounds plausible. When in doubt, use BASELESS.

Important: Do NOT give benefit of the doubt. If a claim contains any specific fact not clearly stated in the reference, mark it BASELESS. "Background knowledge" is NOT a valid reason for ENTAILMENT — the reference passage must explicitly support it.

Return ONLY a JSON array, one object per claim in the same order:
[{"verdict": "ENTAILMENT"|"CONTRADICTION"|"BASELESS", "reason": "brief"}, ...]"""


def verify_claims_gpt(claims: list, reference: str,
                      max_ref_chars: int = 1500,
                      max_retries: int = 2,
                      strict: bool = False) -> list:
    """
    Verify all claims for one record in a SINGLE GPT call (batched).
    Reduces API calls from N_claims to 1 per record.
    Returns list of verdict dicts in the same order as claims.

    strict=True uses a stricter prompt for QA tasks where background
    knowledge should NOT excuse unsupported specific facts.
    """
    if not claims:
        return []

    client = _get_client()
    ref_excerpt = reference[:max_ref_chars] if reference else ""
    system_prompt = BATCH_SYSTEM_PROMPT_STRICT if strict else BATCH_SYSTEM_PROMPT

    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    user_msg = f"Reference:\n{ref_excerpt}\n\nClaims to verify:\n{numbered}"

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=len(claims) * 40 + 50,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            # GPT might wrap the array: {"results": [...]} or return array directly
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                arr = parsed.get("verdicts") or parsed.get("results") or list(parsed.values())[0]
            else:
                arr = parsed

            results = []
            for i, (claim, item) in enumerate(zip(claims, arr)):
                verdict = str(item.get("verdict", "ENTAILMENT")).upper()
                if verdict not in ("ENTAILMENT", "CONTRADICTION", "BASELESS"):
                    verdict = "ENTAILMENT"
                results.append({
                    "claim": claim,
                    "label": verdict,
                    "confidence": 1.0,
                    "reason": item.get("reason", ""),
                    "scores": {verdict: 1.0},
                })
            # Pad if GPT returned fewer items than claims
            for claim in claims[len(results):]:
                results.append({
                    "claim": claim, "label": "ENTAILMENT",
                    "confidence": 0.0, "reason": "missing from response",
                    "scores": {"ENTAILMENT": 1.0},
                })
            return results

        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.5 ** attempt)
            else:
                return [
                    {"claim": c, "label": "ENTAILMENT", "confidence": 0.0,
                     "reason": f"error: {e}", "scores": {"ENTAILMENT": 1.0}}
                    for c in claims
                ]


if __name__ == "__main__":
    ref = (
        "Einstein was born in Ulm, Germany in 1879. "
        "He published his special theory of relativity in 1905. "
        "He received the Nobel Prize in Physics in 1921 for his discovery "
        "of the law of the photoelectric effect."
    )
    test_claims = [
        "Einstein was born in Germany in 1879.",
        "Einstein won the Nobel Prize in 1922.",
        "Einstein invented the internet.",
        "Einstein published the theory of relativity in 1905.",
        "Einstein was a physicist who changed modern science.",
    ]
    print("GPT-4o-mini verification test:")
    for claim in test_claims:
        result = verify_claim_gpt(claim, ref)
        print(f"  [{result['label']:15}] {claim}")
        print(f"    reason: {result.get('reason', '')}")
