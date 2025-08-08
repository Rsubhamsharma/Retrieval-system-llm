from typing import List, Dict, Any


POSITIVE_TERMS = {"cover", "covered", "covers", "eligible", "included", "payable"}
NEGATIVE_TERMS = {"not covered", "excluded", "exclusion", "denied", "no coverage", "not payable"}
CONDITION_TERMS = {"provided that", "subject to", "only if", "after", "waiting period", "conditions"}


def evaluate_decision(question: str, answer: str, retrieved: List[Dict[str, Any]], decision_from_llm: str | None) -> str:
    if decision_from_llm in {"ALLOW", "DENY", "NEEDS_MORE_INFO"}:
        return decision_from_llm  # trust model if explicit

    text = (answer or "") + "\n" + "\n".join(r.get("text", "") for r in retrieved)
    text_lower = text.lower()

    positive = any(term in text_lower for term in POSITIVE_TERMS)
    negative = any(term in text_lower for term in NEGATIVE_TERMS)
    conditional = any(term in text_lower for term in CONDITION_TERMS)

    if positive and not negative and not conditional:
        return "ALLOW"
    if negative and not positive:
        return "DENY"
    return "NEEDS_MORE_INFO"