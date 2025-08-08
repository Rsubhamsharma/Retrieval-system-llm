from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

try:
    from pinecone import Pinecone
except Exception:  # pragma: no cover
    Pinecone = None  # type: ignore


class EmbeddingSearcher:
    def __init__(self, use_pinecone: bool = False, pinecone_index: Optional[str] = None):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.use_pinecone = use_pinecone and Pinecone is not None and pinecone_index is not None
        self.pinecone_index_name = pinecone_index
        self.index = None
        self.embeddings = None
        self.chunks: List[str] = []
        self.pages: List[Optional[int]] = []
        if self.use_pinecone:
            pc = Pinecone()
            self.pinecone = pc.Index(pinecone_index)  # type: ignore[attr-defined]
        else:
            self.pinecone = None

    def build_index_from_text(self, chunks: List[str], pages: List[Optional[int]]):
        self.chunks = chunks
        self.pages = pages
        vectors = self.model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings = vectors.astype("float32")
        if self.use_pinecone:
            # Upsert to Pinecone
            items = [
                {
                    "id": f"chunk-{i}",
                    "values": vec.tolist(),
                    "metadata": {"text": chunks[i], "page": pages[i]},
                }
                for i, vec in enumerate(vectors)
            ]
            # Chunk upserts
            for i in range(0, len(items), 100):
                self.pinecone.upsert(vectors=items[i : i + 100])  # type: ignore
        else:
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        if self.use_pinecone:
            results = self.pinecone.query(vector=q_vec[0].tolist(), top_k=top_k, include_metadata=True)  # type: ignore
            out: List[Dict[str, Any]] = []
            for match in results["matches"]:
                out.append(
                    {
                        "text": match["metadata"]["text"],
                        "page": match["metadata"].get("page"),
                        "score": match.get("score", 0.0),
                    }
                )
            return out
        else:
            assert self.index is not None and self.embeddings is not None
            sims, idxs = self.index.search(q_vec, top_k)
            out = []
            for score, idx in zip(sims[0], idxs[0]):
                if idx == -1:
                    continue
                out.append(
                    {
                        "text": self.chunks[idx],
                        "page": self.pages[idx],
                        "score": float(score),
                    }
                )
            return out