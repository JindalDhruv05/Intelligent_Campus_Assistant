"""Assignment support module.

Helps students understand what a course covers for a given topic,
and frames assignment-level guidance using the RAG pipeline.

This is integrated into the RAG pipeline — when the router classifies
a query as "document" and the question mentions "assignment" or "project",
the rag.py pipeline naturally retrieves the relevant syllabus chunks.

This module provides extra utilities:
  - detect_assignment_query: heuristic to flag assignment-support queries
  - assignment_preamble: adds an assignment-aware system-level note
"""

import re

# ---------------------------------------------------------------------------
# Detection heuristic
# ---------------------------------------------------------------------------

ASSIGNMENT_PATTERNS = [
    r"\bassignment\b",
    r"\bproject\b",
    r"\bhomework\b",
    r"\bpractical\b.*\breport\b",
    r"\blab report\b",
    r"\bhelp (me )?(with|on|for)\b",
    r"\bhow (do i|to) (implement|code|build|solve|write)\b",
    r"\bexplain .{0,30}(for my|in my|for the)\b",
    r"\bwhat (should|do) i (include|write|cover)\b",
]


def detect_assignment_query(question: str) -> bool:
    """Return True if this looks like an assignment-support query."""
    q = question.lower()
    return any(re.search(p, q) for p in ASSIGNMENT_PATTERNS)


# ---------------------------------------------------------------------------
# Enhanced system prompt for assignment context
# ---------------------------------------------------------------------------

ASSIGNMENT_SYSTEM_ADDENDUM = (
    "\nThe student is asking for assignment-related help. "
    "In addition to answering from the context, briefly outline:\n"
    "  1. What the course syllabus covers related to this topic (from the context).\n"
    "  2. Key concepts the student should address.\n"
    "  3. Relevant tools or libraries mentioned in the syllabus.\n"
    "Do NOT write the assignment for the student. Guide them conceptually."
)
