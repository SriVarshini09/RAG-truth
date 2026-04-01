"""
Agent 3: Decision Aggregator
Applies rules over claim verdicts to produce a binary hallucination label
and maps hallucinated claims back to spans in the original response.
"""


def aggregate(claims_with_verdicts: list, response: str, neutral_threshold: float = 0.5) -> dict:
    """
    Decide if a response is hallucinated based on claim verdicts.

    Supports two verifier backends:
      - DeBERTa labels: ENTAILMENT | NEUTRAL | CONTRADICTION
      - GPT labels:     ENTAILMENT | BASELESS | CONTRADICTION

    Rules:
      - Any CONTRADICTION  → hallucination = True
      - >= neutral_threshold fraction NEUTRAL or BASELESS → hallucination = True
      - All ENTAILMENT     → hallucination = False

    Returns:
        {
            "hallucination": bool,
            "hallucination_list": [str, ...],   # text spans found in response
            "verdict_counts": {"ENTAILMENT": int, "NEUTRAL": int, "CONTRADICTION": int, "BASELESS": int},
            "reason": str
        }
    """
    if not claims_with_verdicts:
        return {
            "hallucination": False,
            "hallucination_list": [],
            "verdict_counts": {"ENTAILMENT": 0, "NEUTRAL": 0, "CONTRADICTION": 0, "BASELESS": 0},
            "reason": "no_claims",
        }

    counts = {"ENTAILMENT": 0, "NEUTRAL": 0, "CONTRADICTION": 0, "BASELESS": 0}
    for v in claims_with_verdicts:
        label = v.get("label", "NEUTRAL")
        counts[label] = counts.get(label, 0) + 1

    total = len(claims_with_verdicts)
    hallucinated_claims = []
    reason = "all_entailed"

    # Rule 1: any contradiction
    if counts["CONTRADICTION"] > 0:
        hallucinated_claims = [
            v["claim"] for v in claims_with_verdicts if v.get("label") == "CONTRADICTION"
        ]
        reason = "contradiction_found"

    # Rule 2: too many neutral/baseless (unsupported info)
    # Treat BASELESS (GPT) and NEUTRAL (DeBERTa) the same way
    unsupported = counts["NEUTRAL"] + counts["BASELESS"]
    unsupported_rate = unsupported / total if total > 0 else 0
    if unsupported_rate >= neutral_threshold and not hallucinated_claims:
        hallucinated_claims = [
            v["claim"] for v in claims_with_verdicts
            if v.get("label") in ("NEUTRAL", "BASELESS")
        ]
        reason = "baseless_info"

    hallucination_detected = len(hallucinated_claims) > 0

    # Map hallucinated claims back to spans in the original response
    hallucination_spans = _map_claims_to_spans(hallucinated_claims, response)

    return {
        "hallucination": hallucination_detected,
        "hallucination_list": hallucination_spans,
        "verdict_counts": counts,
        "reason": reason,
    }


def _map_claims_to_spans(claims: list, response: str) -> list:
    """
    Find substrings in the response that overlap with hallucinated claims.
    Uses simple keyword matching: extract key noun phrases from claims and
    find them in the response text.
    """
    spans = []
    response_lower = response.lower()

    for claim in claims:
        # Try direct substring match first
        claim_lower = claim.lower().rstrip(".")
        if claim_lower in response_lower:
            idx = response_lower.find(claim_lower)
            spans.append(response[idx: idx + len(claim_lower)])
            continue

        # Fall back: find longest matching subsequence of words
        claim_words = [w for w in claim_lower.split() if len(w) > 3]
        best_span = _find_best_span(claim_words, response)
        if best_span:
            spans.append(best_span)

    # Deduplicate while preserving order
    seen = set()
    unique_spans = []
    for span in spans:
        if span not in seen:
            seen.add(span)
            unique_spans.append(span)

    return unique_spans


def _find_best_span(keywords: list, response: str) -> str:
    """Find the longest contiguous phrase in response that matches keywords."""
    if not keywords:
        return ""

    response_lower = response.lower()
    words = response_lower.split()
    response_words = response.split()

    best = ""
    window = min(len(keywords) + 3, len(words))

    for start in range(len(words)):
        for end in range(start + 1, min(start + window + 1, len(words) + 1)):
            window_lower = " ".join(words[start:end])
            match_count = sum(1 for kw in keywords if kw in window_lower)
            if match_count >= max(1, len(keywords) // 2):
                candidate = " ".join(response_words[start:end])
                if len(candidate) > len(best):
                    best = candidate

    return best


def format_output(aggregation_result: dict) -> dict:
    """Format result to match RAGTruth expected output format."""
    return {"hallucination list": aggregation_result["hallucination_list"]}


if __name__ == "__main__":
    response = (
        "Einstein was born in Germany in 1879. He won the Nobel Prize in 1922 "
        "for the theory of relativity. He also invented the telephone."
    )
    verdicts = [
        {"claim": "Einstein was born in Germany in 1879.", "label": "ENTAILMENT", "confidence": 0.95},
        {"claim": "He won the Nobel Prize in 1922.", "label": "CONTRADICTION", "confidence": 0.88},
        {"claim": "He invented the telephone.", "label": "CONTRADICTION", "confidence": 0.91},
    ]
    result = aggregate(verdicts, response)
    print(f"Hallucination: {result['hallucination']}")
    print(f"Reason: {result['reason']}")
    print(f"Spans: {result['hallucination_list']}")
    print(f"Counts: {result['verdict_counts']}")
