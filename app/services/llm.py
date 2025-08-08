import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are an expert policy analysis assistant for insurance/legal/compliance. "
    "Given a user question and retrieved clauses with citations, answer precisely from the text. "
    "If a clear yes/no decision is possible, state it as 'ALLOW', 'DENY', or 'NEEDS_MORE_INFO'. "
    "Provide a brief rationale referencing the clauses. Keep answers concise and factual."
)


def _format_context(retrieved: List[Dict[str, Any]]) -> str:
    lines = []
    for i, r in enumerate(retrieved, 1):
        page = r.get("page")
        text = r.get("text", "").replace("\n", " ")
        lines.append(f"[Clause {i}{' p.'+str(page) if page else ''}] {text}")
    return "\n".join(lines)


async def generate_rationale_and_answer(question: str, retrieved: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    context = _format_context(retrieved)

    if not api_key:
        # Fallback heuristic: pick the top chunk and echo
        top = retrieved[0]["text"][:300] if retrieved else ""
        rationale = "Based on the top-matching clause(s)."
        answer = top if top else "No information found in the document for this query."
        decision = "NEEDS_MORE_INFO"
        return rationale, answer, decision

    client = OpenAI(api_key=api_key)
    prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Respond with JSON having keys: answer, rationale, decision (ALLOW|DENY|NEEDS_MORE_INFO)."
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    content = completion.choices[0].message.content.strip()

    # Try to parse JSON, fallback to plain
    import json

    try:
        data = json.loads(content)
        answer = data.get("answer", "")
        rationale = data.get("rationale", "")
        decision = data.get("decision", "NEEDS_MORE_INFO")
    except Exception:
        answer = content
        rationale = "Model returned non-JSON content; treated as answer."
        decision = "NEEDS_MORE_INFO"

    return rationale, answer, decision