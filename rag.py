"""Core RAG pipeline: retrieve, ground the prompt, answer with citations, refuse when unsure.

New in post-mid version:
- Assignment-aware prompting (detects assignment queries and adds guidance preamble).
- answer() accepts an optional extra_system kwarg for caller-supplied prompt additions.
"""
import json
import faiss
import ollama
from sentence_transformers import SentenceTransformer
import config
from assignment_support import detect_assignment_query, ASSIGNMENT_SYSTEM_ADDENDUM

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
REFUSAL = ("I don't know based on the available campus documents. "
           "Please contact the relevant office for confirmation.")

SYSTEM = (
    "You are the Intelligent Campus Assistant for SVNIT Surat. "
    "Answer ONLY using the numbered context passages provided. "
    "Do not use outside knowledge. Cite the passages you used like [1] or [2]. "
    "If passages give different values for different years or document versions, "
    "say which document or year each value comes from. "
    "If the context does not contain the answer, reply exactly: "
    "\"I don't know based on the available campus documents.\" Be concise."
)


def label(h):
    bits = [h.get("title") or h["source"]]
    if h.get("url"):
        bits.append(h["url"])
    if h.get("page"):
        bits.append(f"page {h['page']}")
    return ", ".join(bits)


def looks_like_refusal(text):
    t = text.lower()
    return any(p in t for p in ["i don't know", "i do not know", "not sure",
                                "cannot find", "no information", "i'm not able"])


class CampusRAG:
    def __init__(self):
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        self.index = faiss.read_index(str(config.INDEX_PATH))
        self.chunks = json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))

    def retrieve(self, query, k=config.TOP_K):
        q = self.embedder.encode([QUERY_PREFIX + query],
                                 normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, k)
        return [dict(self.chunks[i], score=float(s))
                for s, i in zip(scores[0], ids[0]) if i != -1]

    def llm(self, messages):
        r = ollama.chat(model=config.LLM_MODEL, messages=messages,
                        options={"temperature": 0, "num_ctx": config.NUM_CTX})
        return r["message"]["content"].strip()

    def answer(self, query, k=config.TOP_K, min_score=config.MIN_SCORE,
               extra_system: str = ""):
        """
        Retrieve, ground, and answer. If the query looks like an assignment
        question, the system prompt is augmented with guidance instructions.

        Parameters
        ----------
        query        : user question
        k            : number of chunks to retrieve
        min_score    : refusal threshold
        extra_system : additional text appended to the system prompt
        """
        hits = self.retrieve(query, k)
        if not hits or hits[0]["score"] < min_score:
            return {"answer": REFUSAL, "sources": hits, "refused": True, "context": ""}

        context = "\n\n".join(
            f"[{n}] ({label(h)})\n{h['text']}"
            for n, h in enumerate(hits, 1))

        # Build system prompt, augmenting for assignment queries
        system = SYSTEM
        if detect_assignment_query(query):
            system = system + ASSIGNMENT_SYSTEM_ADDENDUM
        if extra_system:
            system = system + "\n" + extra_system

        ans = self.llm([
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ])
        return {"answer": ans, "sources": hits,
                "refused": looks_like_refusal(ans), "context": context}

    def answer_baseline(self, query):
        """LLM alone, no retrieval. Used as the comparison baseline."""
        ans = self.llm([
            {"role": "system", "content": "You are a helpful university campus assistant."},
            {"role": "user", "content": query},
        ])
        return {"answer": ans, "sources": [], "refused": looks_like_refusal(ans), "context": ""}
