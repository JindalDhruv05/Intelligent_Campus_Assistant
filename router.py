"""Question router: two classifiers, one regex-based and one LLM-based.

The app uses classify_llm (more accurate). classify (regex) is kept as a
lightweight alternative and for demos/evaluation comparisons.

Usage:
    from router import classify, classify_llm

    # Fast, no model call (for demos / showing the approach)
    route = classify(question)

    # Accurate, uses local LLM (used by app.py)
    route = classify_llm(question, llm_fn=rag.llm)
"""
import re

# ---------------------------------------------------------------------------
# Pattern-based classifier (lightweight, no model needed)
# ---------------------------------------------------------------------------

_TIMETABLE_PATTERNS = [
    r"\b(clash|clashes|conflict|schedule|timetable|time[-\s]?table)\b",
    r"\bwhen (is|are|does)\b.*(class|lecture|lab|practical|exam)",
    r"\b[A-Z]{2,4}\d{3}\b",          # bare course codes like AI463, CS301
    r"\bslot\b",
    r"\broom\b.*\b(for|of)\b",
    r"\bdo .+ clash\b",
    r"\boverlap\b",
]

_RECOMMEND_PATTERNS = [
    r"\b(recommend|suggest)\b",
    r"\bwhich (elective|course|subject)s?\b.*(suit|interest|take|choose|pick)\b",
    r"\bwhich (elective|course|subject)s?\b",
    r"\bsuit(s|ed|able)?\b.*\b(interest|me|my)\b",
    r"\binterest(ed)? in\b",
    r"\bprerequisite[s]?\b",
    r"\bbefore taking\b",
    r"\bshould i (take|choose|pick|do)\b",
    r"\bgood for (someone|me|a student)\b",
    r"\bwhat to (take|study) (before|after|next)\b",
    r"\brelated course[s]?\b",
    r"\bfor (nlp|ml|deep learning|cv|computer vision|data science|security|cloud|rl|gen ai|generative)\b",
    r"\bsuit(s|ed)? (nlp|ml|ai|cv|data|security|cloud)\b",
    r"\binterested in (nlp|ml|deep learning|gen|generative|rl|security|cloud|data)\b",
]

_NAVIGATE_PATTERNS = [
    r"\b(where is|where are|how (do i|to) (get|reach|find|go))\b",
    r"\b(location|located|direction[s]?|navigate|navigation)\b",
    r"\b(library|hostel|canteen|mess|admin|administration|office|lab|block|gate|auditorium|gym|ground|clinic|hospital|infirmary|atm|bank|workshop)\b",
    r"\bbuilding\b",
    r"\bwhich (block|floor|room)\b",
]


def _matches(patterns, text):
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def classify(question: str) -> str:
    """
    Regex-based classifier. Fast, no model call.
    Returns one of: 'timetable', 'recommend', 'navigate', 'document'.

    Limitation: may misclassify edge cases (e.g. 'library timings' → navigate
    instead of document). Use classify_llm for production.
    """
    if _matches(_TIMETABLE_PATTERNS, question):
        return "timetable"
    if _matches(_RECOMMEND_PATTERNS, question):
        return "recommend"
    if _matches(_NAVIGATE_PATTERNS, question):
        return "navigate"
    return "document"


# ---------------------------------------------------------------------------

ROUTER_SYSTEM = """\
You are a query classifier for a university campus assistant at SVNIT Surat.
Classify the user query into exactly ONE of these four categories:

  timetable  – about class timings, room numbers, schedule, or whether two courses clash
  recommend  – asking which course or elective to pick, based on interest or prerequisites
  navigate   – asking where a physical location, building, office or facility is on campus
  document   – anything else: academic rules, grades, syllabus content, exam policy, fees, library rules, scholarship, etc.

Important rules:
- "library timings" is about a RULE or POLICY, not a physical location → document
- "where is the library" is about LOCATION → navigate
- "which courses should I take for NLP" → recommend
- "when is the Machine Learning class" or "do AI463 and AI465 clash" → timetable
- "what is the minimum attendance" → document
- When in doubt, choose document.

Reply with the single category word only. No explanation.
"""


def classify_llm(question: str, llm_fn) -> str:
    """
    Classify a question using the LLM.

    Parameters
    ----------
    question : str   – the user's question
    llm_fn   : callable – accepts a list of message dicts, returns str
                          (this is CampusRAG.llm from rag.py)

    Returns
    -------
    One of: 'timetable', 'recommend', 'navigate', 'document'
    """
    result = llm_fn([
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": question},
    ]).strip().lower()

    for cat in ("timetable", "recommend", "navigate", "document"):
        if cat in result:
            return cat

    return "document"   # safe fallback
