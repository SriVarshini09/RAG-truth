"""
Agent 2: Claim Verifier
Uses DeBERTa-v3-large NLI to verify each atomic claim against source passages.

Key improvement: sentence-level reference chunking.
Instead of truncating the full reference (which causes 65%+ NEUTRAL labels),
we split the reference into sentences and pick the top-K most relevant sentences
per claim using word-overlap scoring. NLI runs on the focused context only.
This fits comfortably in 512 tokens and dramatically improves precision.
"""
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cross-encoder/nli-deberta-v3-large"
_tokenizer = None
_model = None
_device = None


def _load_model():
    global _tokenizer, _model, _device
    if _tokenizer is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ClaimVerifier] Loading {MODEL_NAME} on {_device} ...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model = _model.to(_device)
        _model.eval()
        print(f"[ClaimVerifier] Model loaded on {_device}.")


# ─── Sentence-level retrieval ─────────────────────────────────────────────────

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "to", "of", "in", "on", "at", "by",
    "for", "with", "about", "as", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "out", "off", "over",
    "under", "again", "then", "that", "this", "these", "those", "it", "its",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "no", "he", "she", "they", "we", "you", "i", "his", "her",
    "their", "our", "your", "my", "its", "also", "just", "than",
}


def _tokenize_words(text: str) -> set:
    words = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _split_sentences(text: str) -> list:
    """Split reference into sentences, keeping overlapping 2-sentence windows."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    raw = [s.strip() for s in raw if len(s.strip()) > 20]
    if not raw:
        return [text[:800]]

    # Build overlapping windows: single sentences + pairs
    windows = []
    for i, s in enumerate(raw):
        windows.append(s)
        if i + 1 < len(raw):
            windows.append(s + " " + raw[i + 1])
    return windows


def _get_best_context(claim: str, reference: str, top_k: int = 3, max_chars: int = 600) -> str:
    """
    Find top_k most relevant windows from reference for the given claim.
    Uses word-overlap (Jaccard-style) scoring — no extra models needed.
    """
    claim_words = _tokenize_words(claim)
    if not claim_words:
        return reference[:max_chars]

    windows = _split_sentences(reference)
    if not windows:
        return reference[:max_chars]

    scored = []
    for w in windows:
        w_words = _tokenize_words(w)
        if not w_words:
            continue
        overlap = len(claim_words & w_words)
        score = overlap / (len(claim_words | w_words) + 1e-9)
        scored.append((score, w))

    scored.sort(key=lambda x: -x[0])
    top = [w for _, w in scored[:top_k]]

    context = " ".join(top)
    return context[:max_chars]


# ─── NLI inference ───────────────────────────────────────────────────────────

def _run_nli(premises: list, hypotheses: list) -> list:
    """Run batched NLI and return list of score dicts."""
    inputs = _tokenizer(
        premises,
        hypotheses,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs_batch = torch.softmax(logits, dim=-1).cpu().tolist()
    label_names = _model.config.id2label

    results = []
    for probs in probs_batch:
        label_map = {v.upper(): probs[k] for k, v in label_names.items()}
        scores = {
            "ENTAILMENT": round(label_map.get("ENTAILMENT", 0.0), 4),
            "NEUTRAL": round(label_map.get("NEUTRAL", 0.0), 4),
            "CONTRADICTION": round(label_map.get("CONTRADICTION", 0.0), 4),
        }
        results.append(scores)
    return results


def verify_claim(claim: str, reference: str) -> dict:
    """
    Verify one claim against reference using sentence-level context retrieval + NLI.
    """
    _load_model()
    if not reference or not claim:
        return {
            "claim": claim, "label": "NEUTRAL", "confidence": 0.0,
            "scores": {"ENTAILMENT": 0.0, "NEUTRAL": 1.0, "CONTRADICTION": 0.0},
        }

    context = _get_best_context(claim, reference)
    scores_list = _run_nli([context], [claim])
    scores = scores_list[0]
    best_label = max(scores, key=scores.__getitem__)
    return {
        "claim": claim, "label": best_label,
        "confidence": scores[best_label],
        "scores": scores,
    }


def verify_claims(claims: list, reference: str, batch_size: int = 16) -> list:
    """
    Verify a list of claims against the same reference.
    Each claim gets its own focused context window from the reference.
    Batched GPU inference for speed.
    """
    if not claims:
        return []
    _load_model()

    if not reference:
        return [
            {"claim": c, "label": "NEUTRAL", "confidence": 0.0,
             "scores": {"ENTAILMENT": 0.0, "NEUTRAL": 1.0, "CONTRADICTION": 0.0}}
            for c in claims
        ]

    # Build per-claim focused contexts
    contexts = [_get_best_context(claim, reference) for claim in claims]
    results = []

    for i in range(0, len(claims), batch_size):
        batch_claims = claims[i: i + batch_size]
        batch_contexts = contexts[i: i + batch_size]
        scores_list = _run_nli(batch_contexts, batch_claims)
        for claim, scores in zip(batch_claims, scores_list):
            best_label = max(scores, key=scores.__getitem__)
            results.append({
                "claim": claim, "label": best_label,
                "confidence": scores[best_label],
                "scores": scores,
            })

    return results


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
    ]
    print("Sentence-level context NLI test:")
    for claim in test_claims:
        ctx = _get_best_context(claim, ref)
        result = verify_claim(claim, ref)
        print(f"  [{result['label']:15}] ({result['confidence']:.3f})  {claim}")
        print(f"    context: {ctx[:80]}...")
