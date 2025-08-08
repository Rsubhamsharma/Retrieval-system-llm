from typing import List, Dict, Any
import asyncio
from app.services.ingest import fetch_and_parse_document
from app.services.embeddings import EmbeddingSearcher
from app.services.logic import evaluate_decision
from app.services.llm import generate_rationale_and_answer


async def run_pipeline(document_url: str, questions: List[str]) -> List[Dict[str, Any]]:
    text, pages = await fetch_and_parse_document(document_url)

    searcher = EmbeddingSearcher()
    searcher.build_index_from_text(text, pages)

    results: List[Dict[str, Any]] = []

    async def process_question(question: str) -> Dict[str, Any]:
        retrieved = searcher.search(question, top_k=6)
        rationale, answer, decision = await generate_rationale_and_answer(question, retrieved)
        evaluated_decision = evaluate_decision(question, answer, retrieved, decision)
        citations = [
            {
                "page": r.get("page"),
                "score": float(r.get("score", 0.0)),
                "text": r.get("text", "")[:500],
            }
            for r in retrieved
        ]
        return {
            "question": question,
            "answer": answer,
            "rationale": rationale,
            "citations": citations,
            "decision": evaluated_decision,
        }

    tasks = [process_question(q) for q in questions]
    for completed in await asyncio.gather(*tasks):
        results.append(completed)

    return results