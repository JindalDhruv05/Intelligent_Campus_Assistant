"""Conversation memory for the campus assistant.

Stores the last N turns and rewrites ambiguous follow-up questions into
standalone queries using the LLM, so the RAG retriever gets a fully
self-contained question every time.

Usage:
    from memory import ConversationMemory
    mem = ConversationMemory(llm_fn=rag.llm, max_turns=5)

    # Rewrite a follow-up into a standalone query before retrieval
    standalone = mem.rewrite(user_question)
    # ... call rag.answer(standalone) ...
    mem.add_turn(user_question, bot_answer)
"""
from __future__ import annotations

REWRITE_SYSTEM = """\
You are a query rewriter for a university campus assistant.
Given the conversation history and a new follow-up question from the user,
rewrite the follow-up into a single, self-contained question that can be
answered without reading the history.

Rules:
- Resolve all pronouns (it, they, that course, etc.) using the history.
- Keep the rewritten question concise.
- If the follow-up is already standalone, return it unchanged.
- Output only the rewritten question, no explanation.
"""


class ConversationMemory:
    """Ring buffer of (user, assistant) turn pairs."""

    def __init__(self, llm_fn=None, max_turns: int = 5):
        self._llm = llm_fn
        self._max = max_turns
        self._turns: list[dict] = []   # each: {"user": ..., "assistant": ...}

    # ------------------------------------------------------------------
    def add_turn(self, user: str, assistant: str) -> None:
        self._turns.append({"user": user, "assistant": assistant})
        if len(self._turns) > self._max:
            self._turns.pop(0)

    def clear(self) -> None:
        self._turns = []

    def is_empty(self) -> bool:
        return len(self._turns) == 0

    # ------------------------------------------------------------------
    def _history_text(self) -> str:
        lines = []
        for i, t in enumerate(self._turns, 1):
            lines.append(f"Turn {i}")
            lines.append(f"  User: {t['user']}")
            lines.append(f"  Assistant: {t['assistant'][:300]}")
        return "\n".join(lines)

    def _looks_like_followup(self, question: str) -> bool:
        """Heuristic: does this question reference something from prior context?"""
        q = question.lower().strip()
        # Short questions are almost always follow-ups
        if len(q.split()) <= 4:
            return True
        # Common referencing words
        FOLLOWUP_WORDS = [
            "it", "they", "that", "this", "those", "these",
            "the course", "the subject", "the exam", "the grade",
            "and for", "what about", "how about", "same for",
            "also", "additionally", "furthermore",
            "what else", "tell me more", "explain more",
        ]
        return any(w in q for w in FOLLOWUP_WORDS)

    # ------------------------------------------------------------------
    def rewrite(self, question: str) -> str:
        """
        Rewrite a follow-up question into a standalone query.
        Returns the question unchanged if memory is empty or question is standalone.
        """
        if self.is_empty() or not self._looks_like_followup(question):
            return question
        if self._llm is None:
            return question   # no LLM available; return as-is

        history = self._history_text()
        result = self._llm([
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": (
                f"Conversation history:\n{history}\n\n"
                f"Follow-up question: {question}\n\n"
                "Rewritten standalone question:"
            )},
        ])
        return result.strip().strip('"\'') or question

    # ------------------------------------------------------------------
    def as_messages(self) -> list[dict]:
        """Return conversation history as OpenAI-style message list."""
        msgs = []
        for t in self._turns:
            msgs.append({"role": "user", "content": t["user"]})
            msgs.append({"role": "assistant", "content": t["assistant"]})
        return msgs

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return f"ConversationMemory(turns={len(self._turns)}, max={self._max})"
