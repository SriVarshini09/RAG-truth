"""
Agent 2: Claim Verifier
Uses DeBERTa-v3-large-MNLI to verify each atomic claim against source passages.
No API cost — runs locally via HuggingFace Transformers.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cross-encoder/nli-deberta-v3-large"
_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _tokenizer is None:
        print(f"[ClaimVerifier] Loading {MODEL_NAME} ...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
        print("[ClaimVerifier] Model loaded.")


def verify_claim(claim: str, reference: str, max_ref_chars: int = 2000) -> dict:
    """
    Verify one claim against reference text using NLI.

    Returns:
        {
            "claim": str,
            "label": "ENTAILMENT" | "NEUTRAL" | "CONTRADICTION",
            "confidence": float,
            "scores": {"ENTAILMENT": float, "NEUTRAL": float, "CONTRADICTION": float}
        }
    """
    _load_model()

    ref_truncated = reference[:max_ref_chars] if reference else ""
    if not ref_truncated or not claim:
        return {
            "claim": claim,
            "label": "NEUTRAL",
            "confidence": 0.0,
            "scores": {"ENTAILMENT": 0.0, "NEUTRAL": 1.0, "CONTRADICTION": 0.0},
        }

    inputs = _tokenizer(
        ref_truncated,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        logits = _model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    # DeBERTa NLI label order: contradiction, neutral, entailment
    label_names = _model.config.id2label
    label_map = {v.upper(): probs[k] for k, v in label_names.items()}

    # Normalise key names
    scores = {
        "ENTAILMENT": label_map.get("ENTAILMENT", 0.0),
        "NEUTRAL": label_map.get("NEUTRAL", 0.0),
        "CONTRADICTION": label_map.get("CONTRADICTION", 0.0),
    }

    best_label = max(scores, key=scores.__getitem__)

    return {
        "claim": claim,
        "label": best_label,
        "confidence": round(scores[best_label], 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
    }


def verify_claims(claims: list, reference: str) -> list:
    """
    Verify a list of claims against the same reference.
    Returns list of verdict dicts.
    """
    return [verify_claim(claim, reference) for claim in claims]


if __name__ == "__main__":
    ref = (
        "Einstein was born in Ulm, Germany in 1879. "
        "He published his special theory of relativity in 1905. "
        "He received the Nobel Prize in Physics in 1921."
    )
    test_claims = [
        "Einstein was born in Germany in 1879.",
        "Einstein won the Nobel Prize in 1922.",
        "Einstein invented the internet.",
    ]
    for claim in test_claims:
        result = verify_claim(claim, ref)
        print(f"  [{result['label']}] ({result['confidence']:.3f}) {claim}")
