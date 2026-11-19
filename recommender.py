"""Course recommendation engine for B.Tech AI curriculum.

Provides interest-based and prerequisite-based course recommendations
using the structured curriculum data embedded below.

Usage (as a library):
    from recommender import CourseRecommender
    rec = CourseRecommender()
    result = rec.answer("Which electives suit someone interested in NLP?")

The recommender uses an LLM to reason over the curriculum catalogue.
"""
import json

# ---------------------------------------------------------------------------
# Curriculum catalogue (derived from B.Tech AI 2023-24 scheme)
# This is a structured summary; the full syllabi are in the FAISS index.
# ---------------------------------------------------------------------------

CURRICULUM = {
    "core": [
        {"code": "AI101", "name": "Programming for AI", "sem": 1, "credits": 5,
         "tags": ["python", "programming", "basics"], "prereqs": []},
        {"code": "AI102", "name": "Data Structures", "sem": 2, "credits": 5,
         "tags": ["algorithms", "data structures", "programming"], "prereqs": ["AI101"]},
        {"code": "AI201", "name": "Design and Analysis of Algorithms", "sem": 3, "credits": 4,
         "tags": ["algorithms", "complexity", "optimization"], "prereqs": ["AI102"]},
        {"code": "AI203", "name": "Database Management Systems", "sem": 3, "credits": 4,
         "tags": ["databases", "sql", "data"], "prereqs": []},
        {"code": "AI204", "name": "Probability and Statistics", "sem": 3, "credits": 4,
         "tags": ["probability", "statistics", "maths"], "prereqs": []},
        {"code": "AI301", "name": "Machine Learning", "sem": 5, "credits": 4,
         "tags": ["machine learning", "ml", "regression", "classification", "svm",
                  "neural networks", "data science"], "prereqs": ["AI204"]},
        {"code": "AI303", "name": "Data Science", "sem": 5, "credits": 4,
         "tags": ["data science", "analytics", "visualization", "pandas"], "prereqs": ["AI204"]},
        {"code": "AI302", "name": "Deep Learning", "sem": 6, "credits": 4,
         "tags": ["deep learning", "cnn", "rnn", "transformers", "neural networks"],
         "prereqs": ["AI301"]},
        {"code": "AI304", "name": "Cloud Computing", "sem": 6, "credits": 4,
         "tags": ["cloud", "aws", "azure", "deployment", "distributed"], "prereqs": []},
        {"code": "AI401", "name": "Intelligent Multiagent and Expert Systems", "sem": 7, "credits": 4,
         "tags": ["agents", "expert systems", "multiagent", "ai"], "prereqs": ["AI301"]},
        {"code": "AIP08", "name": "Industrial Internship / Professional Experience", "sem": 8, "credits": 20,
         "tags": ["internship", "industry", "professional"], "prereqs": []},
    ],
    "electives": [
        {"code": "AI357", "name": "Natural Language Processing", "sem": 5, "credits": 4,
         "tags": ["nlp", "text", "language", "tokenization", "bert", "word embeddings",
                  "sentiment", "named entity", "machine translation"],
         "prereqs": ["AI301"]},
        {"code": "AI354", "name": "Responsible AI", "sem": 6, "credits": 4,
         "tags": ["ethics", "fairness", "bias", "explainability", "responsible ai", "safety"],
         "prereqs": ["AI301"]},
        {"code": "AI331", "name": "Information Security and Cryptography", "sem": 5, "credits": 4,
         "tags": ["security", "cryptography", "encryption", "cyber security"],
         "prereqs": []},
        {"code": "AI332", "name": "Reinforcement Learning", "sem": 6, "credits": 4,
         "tags": ["reinforcement learning", "rl", "q-learning", "policy", "reward",
                  "game playing", "optimization"], "prereqs": ["AI301"]},
        {"code": "AI463", "name": "Introduction to Large Language Models", "sem": 7, "credits": 4,
         "tags": ["llm", "large language models", "gpt", "bert", "transformers",
                  "rag", "fine-tuning", "prompt engineering", "nlp", "generative ai"],
         "prereqs": ["AI302", "AI357"]},
        {"code": "AI465", "name": "Agentic AI", "sem": 7, "credits": 4,
         "tags": ["agents", "agentic ai", "autonomous agents", "tool use", "mcp",
                  "llm agents", "planning", "reasoning"],
         "prereqs": ["AI463"]},
    ],
    "honors": [
        {"code": "AI212", "name": "Advanced Algorithms", "sem": 4, "credits": 4,
         "tags": ["algorithms", "complexity", "np-completeness", "approximation",
                  "competitive programming"], "prereqs": ["AI201"]},
        {"code": "AI311", "name": "Nature Inspired Computing and Optimization", "sem": 5, "credits": 4,
         "tags": ["genetic algorithms", "swarm intelligence", "pso", "optimization",
                  "evolutionary computing", "bio-inspired"], "prereqs": ["AI204"]},
        {"code": "AI312", "name": "Generative AI", "sem": 6, "credits": 4,
         "tags": ["generative ai", "gans", "vae", "diffusion models", "stable diffusion",
                  "image generation", "dcgan"], "prereqs": ["AI302"]},
        {"code": "AI411", "name": "Advanced Topics in Deep Learning", "sem": 7, "credits": 4,
         "tags": ["deep learning", "attention", "graph neural networks", "gnn",
                  "self-supervised learning", "few-shot learning", "vision transformers"],
         "prereqs": ["AI302"]},
    ],
}

ALL_COURSES = (
    CURRICULUM["core"] + CURRICULUM["electives"] + CURRICULUM["honors"]
)

CATALOGUE_JSON = json.dumps(ALL_COURSES, indent=2)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

RECOMMEND_SYSTEM = """\
You are the course advisor for the B.Tech AI programme at SVNIT Surat.
Your job is to recommend courses from the official curriculum catalogue provided.

Rules:
1. Only recommend courses that exist in the catalogue below.
2. For interest-based queries, pick courses whose tags match the stated interest.
3. For prerequisite queries, trace the prereqs chain and list what must be done first.
4. Always mention: course name, code, semester, credit count, and why it fits.
5. Mention the semester so the student knows when they can take it.
6. If no course fits, say so honestly.
7. Do not invent courses that are not in the catalogue.

Catalogue (JSON):
{catalogue}
"""

RECOMMEND_USER = "Student query: {question}"

# ---------------------------------------------------------------------------
# Recommender class
# ---------------------------------------------------------------------------


class CourseRecommender:
    def __init__(self, llm_fn=None):
        """
        llm_fn: callable that accepts a list of messages and returns a string.
        If None, a simple keyword fallback is used (no LLM required for basic use).
        """
        self._llm = llm_fn

    def _keyword_fallback(self, question: str) -> str:
        """Simple tag-matching fallback when no LLM is available."""
        q = question.lower()
        matched = []
        for course in ALL_COURSES:
            if any(tag in q for tag in course["tags"]):
                matched.append(course)

        if not matched:
            return ("I could not find a close match in the curriculum catalogue. "
                    "Please describe your interest more specifically (e.g. 'NLP', "
                    "'deep learning', 'generative AI', 'optimization').")

        lines = ["Here are the courses that match your interest:\n"]
        for c in matched:
            kind = ("Core" if c in CURRICULUM["core"]
                    else "Honors" if c in CURRICULUM["honors"] else "Elective")
            prereq_str = (", ".join(c["prereqs"]) if c["prereqs"] else "None")
            lines.append(
                f"**{c['name']} ({c['code']})** — Sem {c['sem']}, {c['credits']} credits [{kind}]\n"
                f"  Prerequisites: {prereq_str}\n"
                f"  Tags: {', '.join(c['tags'][:6])}\n"
            )
        return "\n".join(lines)

    def answer(self, question: str) -> dict:
        """Answer a recommendation question. Returns standard response dict."""
        if self._llm is None:
            text = self._keyword_fallback(question)
        else:
            system = RECOMMEND_SYSTEM.format(catalogue=CATALOGUE_JSON)
            text = self._llm([
                {"role": "system", "content": system},
                {"role": "user", "content": RECOMMEND_USER.format(question=question)},
            ])
        return {
            "answer": text,
            "sources": [{"source": "B_Tech-AI-2023-24-Curriculum-First_to_Fouth_Year-Syllabus.pdf",
                         "text": "Curriculum catalogue (structured)",
                         "score": 1.0, "page": None}],
            "refused": False,
            "context": CATALOGUE_JSON,
        }

    def prerequisites_for(self, code: str) -> list[str]:
        """Return the full prerequisite chain for a course code."""
        course = next((c for c in ALL_COURSES if c["code"] == code), None)
        if not course:
            return []
        chain = list(course["prereqs"])
        for prereq in course["prereqs"]:
            chain.extend(self.prerequisites_for(prereq))
        return list(dict.fromkeys(chain))  # deduplicated, insertion order

    def courses_in_sem(self, sem: int) -> list[dict]:
        """Return all courses (core + elective + honors) in a given semester."""
        return [c for c in ALL_COURSES if c["sem"] == sem]
