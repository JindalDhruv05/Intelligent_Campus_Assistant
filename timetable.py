"""Timetable module: clash checker + LLM-assisted natural-language query handler.

Standalone CLI:
    python timetable.py                    # show full timetable
    python timetable.py AI463 AI465        # schedule + clash check for those codes

As a library (called by app.py via the router):
    from timetable import TimetableHandler
    handler = TimetableHandler(llm_fn=rag.llm)
    result = handler.answer("When is the LLMs class?")
    result = handler.answer("Do AI463 and AI465 clash?")
"""
import csv
import re
from itertools import combinations
from pathlib import Path

PATH = Path(__file__).parent / "data" / "timetable.csv"
DAYS_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


# ---------------------------------------------------------------------------
# Core data helpers
# ---------------------------------------------------------------------------

def _to_min(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def _load_all() -> list[dict]:
    if not PATH.exists():
        return []
    with open(PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _filter(rows: list[dict], codes: set[str]) -> list[dict]:
    if not codes:
        return rows
    return [r for r in rows if r["course_code"] in codes]


def _find_clashes(rows: list[dict]) -> list[tuple[dict, dict]]:
    return [
        (a, b)
        for a, b in combinations(rows, 2)
        if a["day"] == b["day"]
        and _to_min(a["start"]) < _to_min(b["end"])
        and _to_min(b["start"]) < _to_min(a["end"])
    ]


# ---------------------------------------------------------------------------
# LLM-based intent + entity extraction
# ---------------------------------------------------------------------------

def _build_extraction_prompt(question: str, catalogue: str) -> str:
    return f"""\
You help parse timetable queries for a university campus assistant.

The timetable contains these courses:
{catalogue}

Given the user query, extract:
1. intent: one of "clash_check", "schedule_lookup", "show_all"
   - clash_check: user wants to know if two or more courses overlap in time
   - schedule_lookup: user wants to know the timing/room for one or more courses
   - show_all: user wants to see the full timetable
2. codes: a JSON list of course codes mentioned (e.g. ["AI463", "AI465"]).
   Match by code OR by course name (partial match is fine).
   If the user says "LLMs class" or "Introduction to Large Language Models", that is AI463.
   If no specific course is mentioned, return [].

Reply in this exact format (nothing else):
intent: <intent>
codes: <json list>
"""


# ---------------------------------------------------------------------------
# Natural-language timetable handler
# ---------------------------------------------------------------------------

class TimetableHandler:
    def __init__(self, llm_fn=None):
        """
        llm_fn: callable that accepts a list of message dicts and returns str.
        When provided, used to parse NL queries intelligently.
        Falls back to regex extraction if llm_fn is None.
        """
        self.rows = _load_all()
        self._llm = llm_fn
        # Build a quick name->code map for fuzzy name matching
        self._name_to_code: dict[str, str] = {
            r["course_name"].lower(): r["course_code"] for r in self.rows
        }

    # ------------------------------------------------------------------
    # Catalogue text (for the LLM extraction prompt)
    # ------------------------------------------------------------------
    def _catalogue_text(self) -> str:
        seen = {}
        for r in self.rows:
            seen[r["course_code"]] = r["course_name"]
        return "\n".join(f"  {code}: {name}" for code, name in sorted(seen.items()))

    # ------------------------------------------------------------------
    # Code extraction helpers
    # ------------------------------------------------------------------
    def _extract_codes_regex(self, text: str) -> set[str]:
        """Pull uppercase course codes like AI463 from text."""
        return set(re.findall(r"\b[A-Z]{2,4}\d{3}\b", text))

    def _extract_codes_llm(self, question: str) -> tuple[str, set[str]]:
        """
        Use the LLM to extract intent and course codes from a NL question.
        Returns (intent, codes_set).
        Falls back to regex on parse error.
        """
        catalogue = self._catalogue_text()
        prompt = _build_extraction_prompt(question, catalogue)
        try:
            raw = self._llm([{"role": "user", "content": prompt}]).strip()
            intent, codes = "schedule_lookup", set()
            for line in raw.splitlines():
                line = line.strip()
                if line.startswith("intent:"):
                    val = line.split(":", 1)[1].strip().lower()
                    if val in ("clash_check", "schedule_lookup", "show_all"):
                        intent = val
                elif line.startswith("codes:"):
                    import json
                    raw_list = line.split(":", 1)[1].strip()
                    parsed = json.loads(raw_list)
                    codes = set(parsed)
            return intent, codes
        except Exception:
            # Fallback: use regex and keyword detection
            codes = self._extract_codes_regex(question)
            q = question.lower()
            intent = ("clash_check" if any(w in q for w in ["clash", "conflict", "overlap", "same time"])
                      else "show_all" if (not codes and any(w in q for w in ["all", "full", "show", "list", "complete"]))
                      else "schedule_lookup")
            return intent, codes

    # ------------------------------------------------------------------
    # Output formatters
    # ------------------------------------------------------------------
    def all_courses_text(self) -> str:
        if not self.rows:
            return "No timetable data loaded."
        # Deduplicate: group slots by course
        course_slots: dict[str, list[dict]] = {}
        for r in self.rows:
            course_slots.setdefault(r["course_code"], []).append(r)

        lines = ["**Full Timetable:**\n",
                 "| Code | Course | Day | Start | End | Room |",
                 "|------|--------|-----|-------|-----|------|"]
        for r in sorted(self.rows,
                        key=lambda x: (DAYS_ORDER.index(x["day"]) if x["day"] in DAYS_ORDER else 99,
                                       x["start"])):
            lines.append(f"| {r['course_code']} | {r['course_name']} | "
                         f"{r['day']} | {r['start']} | {r['end']} | {r['room']} |")
        return "\n".join(lines)

    def schedule_for(self, codes: set[str]) -> str:
        rows = _filter(self.rows, codes)
        if not rows:
            known = ", ".join(sorted({r["course_code"] for r in self.rows}))
            return (f"No schedule found for: {', '.join(sorted(codes))}.\n"
                    f"Available codes: {known}\n"
                    "Check the course codes or update `data/timetable.csv`.")
        lines = [f"**Schedule for {', '.join(sorted(codes))}:**\n",
                 "| Code | Course | Day | Start | End | Room |",
                 "|------|--------|-----|-------|-----|------|"]
        for r in sorted(rows,
                        key=lambda x: (DAYS_ORDER.index(x["day"]) if x["day"] in DAYS_ORDER else 99,
                                       x["start"])):
            lines.append(f"| {r['course_code']} | {r['course_name']} | "
                         f"{r['day']} | {r['start']} | {r['end']} | {r['room']} |")
        return "\n".join(lines)

    def clash_report(self, codes: set[str] | None = None) -> str:
        rows = _filter(self.rows, codes) if codes else self.rows
        if not rows:
            return "No timetable data to check."
        clashes = _find_clashes(rows)
        if not clashes:
            checked = f"Checked: {', '.join(sorted(codes))}." if codes else "Checked all courses."
            return f"✅ No clashes found. {checked}"
        lines = [f"⚠️ **{len(clashes)} clash(es) found:**\n"]
        for a, b in clashes:
            lines.append(
                f"- **{a['day']}**: **{a['course_code']}** ({a['course_name']}) "
                f"{a['start']}–{a['end']} in {a['room']} clashes with "
                f"**{b['course_code']}** ({b['course_name']}) "
                f"{b['start']}–{b['end']} in {b['room']}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def answer(self, question: str) -> dict:
        """
        Handle a natural-language timetable question.
        Uses the LLM to parse intent and extract course codes/names.
        Returns standard response dict.
        """
        if not self.rows:
            text = ("The timetable file (`data/timetable.csv`) could not be loaded. "
                    "Make sure it exists with columns: course_code, course_name, day, start, end, room.")
            return {"answer": text, "sources": [], "refused": False, "context": ""}

        # Use LLM if available, else regex fallback
        if self._llm:
            intent, codes = self._extract_codes_llm(question)
        else:
            q = question.lower()
            codes = self._extract_codes_regex(question)
            intent = ("clash_check" if any(w in q for w in ["clash", "conflict", "overlap", "same time"])
                      else "show_all" if (not codes and any(w in q for w in ["all", "full", "show", "list"]))
                      else "schedule_lookup")

        # Dispatch
        if intent == "show_all":
            text = self.all_courses_text()
        elif intent == "clash_check":
            text = self.clash_report(codes if codes else None)
            if codes:
                # Also show their schedules above the clash report
                sched = self.schedule_for(codes)
                text = sched + "\n\n" + text
        else:
            # schedule_lookup
            if codes:
                sched = self.schedule_for(codes)
                clash = self.clash_report(codes)
                text = sched + "\n\n" + clash
            else:
                # No codes extracted — show everything
                text = self.all_courses_text()

        return {"answer": text, "sources": [], "refused": False, "context": ""}


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    handler = TimetableHandler()   # no LLM in CLI mode; regex fallback used
    codes = set(sys.argv[1:])
    if not codes:
        print(handler.all_courses_text())
    else:
        print(handler.schedule_for(codes))
        print()
        print(handler.clash_report(codes))
