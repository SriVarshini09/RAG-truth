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
    inputs = {k: v.to(_device) for k, v in inputs.items()}

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


def verify_claims(claims: list, reference: str, batch_size: int = 16) -> list:
    """
    Verify a list of claims against the same reference.
    Uses batched inference on GPU for significant speedup.
    Returns list of verdict dicts.
    """
    if not claims:
        return []
    _load_model()

    ref_truncated = reference[:2000] if reference else ""
    results = []

    for i in range(0, len(claims), batch_size):
        batch_claims = claims[i: i + batch_size]
        if not ref_truncated:
            for claim in batch_claims:
                results.append({
                    "claim": claim, "label": "NEUTRAL",
                    "confidence": 0.0,
                    "scores": {"ENTAILMENT": 0.0, "NEUTRAL": 1.0, "CONTRADICTION": 0.0},
                })
            continue

        inputs = _tokenizer(
            [ref_truncated] * len(batch_claims),
            batch_claims,
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

        for claim, probs in zip(batch_claims, probs_batch):
            label_map = {v.upper(): probs[k] for k, v in label_names.items()}
            scores = {
                "ENTAILMENT": round(label_map.get("ENTAILMENT", 0.0), 4),
                "NEUTRAL": round(label_map.get("NEUTRAL", 0.0), 4),
                "CONTRADICTION": round(label_map.get("CONTRADICTION", 0.0), 4),
            }
            best_label = max(scores, key=scores.__getitem__)
            results.append({
                "claim": claim,
                "label": best_label,
                "confidence": scores[best_label],
                "scores": scores,
            })

    return results


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
