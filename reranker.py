"""Optional cross-encoder reranker for ablation experiments.

Wraps a cross-encoder model from sentence-transformers to rerank the
top-k chunks retrieved by the FAISS retriever.

Usage in evaluate.py ablation:
    from reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    hits = rag.retrieve(question, k=8)        # retrieve more than usual
    hits = reranker.rerank(question, hits, k=4)  # rerank to top 4

The default model is 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
It is small (~67 MB) and runs on CPU.

NOTE: First use downloads the model. Requires:
    pip install sentence-transformers
(already in requirements.txt)
"""
from __future__ import annotations

try:
    from sentence_transformers import CrossEncoder
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Rerank retrieved chunks using a cross-encoder relevance model."""

    def __init__(self, model_name: str = RERANKER_MODEL):
        if not _AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for the reranker. "
                "Run: pip install sentence-transformers"
            )
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[dict], k: int | None = None) -> list[dict]:
        """
        Score (query, chunk) pairs with the cross-encoder and return them
        sorted by cross-encoder score (descending).

        Parameters
        ----------
        query : user question
        hits  : list of chunk dicts (must have 'text' key)
        k     : optional limit on number returned (None → return all, sorted)
        """
        if not hits:
            return hits
        pairs = [(query, h["text"]) for h in hits]
        ce_scores = self._model.predict(pairs)
        ranked = sorted(
            [dict(h, ce_score=float(s)) for h, s in zip(hits, ce_scores)],
            key=lambda x: x["ce_score"],
            reverse=True,
        )
        return ranked[:k] if k else ranked


# ---------------------------------------------------------------------------
# Convenience: patch a CampusRAG instance to use the reranker
# ---------------------------------------------------------------------------

def patch_rag_with_reranker(rag_instance, reranker: CrossEncoderReranker,
                            retrieve_k: int = 8, final_k: int = 4):
    """
    Monkey-patch a CampusRAG so that every call to .retrieve() automatically
    fetches retrieve_k chunks and reranks them down to final_k.

    Example:
        rag = CampusRAG()
        reranker = CrossEncoderReranker()
        patch_rag_with_reranker(rag, reranker)
        # rag.answer() now uses reranking transparently
    """
    original_retrieve = rag_instance.retrieve

    def new_retrieve(query, k=final_k):
        big_hits = original_retrieve(query, k=retrieve_k)
        return reranker.rerank(query, big_hits, k=k)

    rag_instance.retrieve = new_retrieve
