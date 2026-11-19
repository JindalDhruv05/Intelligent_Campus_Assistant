"""Campus navigation module for SVNIT Surat.

Provides location information and directions for buildings and facilities.
Data is a static knowledge base; expand as needed.

Usage:
    from navigation import CampusNav
    nav = CampusNav()
    result = nav.answer("Where is the AI department?")
"""
import re
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# SVNIT Surat campus location database
# Expand this list with more locations as needed.
# ---------------------------------------------------------------------------

LOCATIONS = [
    {
        "name": "Department of Computer Science and Engineering (CSE)",
        "aliases": ["cse", "cs department", "computer science", "computer science department"],
        "block": "C Block",
        "floor": "Ground & First Floor",
        "landmark": "Near the main administrative building",
        "description": (
            "The CSE department is located in C Block on the ground and first floors. "
            "It houses faculty offices, seminar halls, and the computing labs."
        ),
    },
    {
        "name": "Department of Artificial Intelligence (AI)",
        "aliases": ["ai department", "ai dept", "artificial intelligence", "ai block",
                    "ai lab", "artificial intelligence department"],
        "block": "C Block / New AI Block",
        "floor": "Second Floor",
        "landmark": "Adjacent to the CSE department",
        "description": (
            "The AI department is located on the second floor of C Block / New AI Block. "
            "It includes faculty cabins, the AI research lab, and classrooms (LH-1, LH-2, LH-3)."
        ),
    },
    {
        "name": "Main Library (Central Library)",
        "aliases": ["library", "central library", "lib", "reading room"],
        "block": "Library Building",
        "floor": "Ground, First & Second Floor",
        "landmark": "Near the main gate, opposite the administrative block",
        "description": (
            "The Central Library is in a dedicated building near the main gate. "
            "Ground floor: periodicals and digital resources. "
            "First floor: lending section. "
            "Second floor: reading and reference rooms. "
            "Library timings are typically 8 AM to 10 PM on weekdays (verify with library notice)."
        ),
    },
    {
        "name": "Administrative Block (Dean's Office / Registrar)",
        "aliases": ["admin", "administration", "admin block", "registrar", "dean",
                    "dean academics", "deans office"],
        "block": "Main Administrative Block",
        "floor": "Ground & First Floor",
        "landmark": "Central campus, near the main entrance road",
        "description": (
            "The Administrative Block is centrally located and houses the Registrar's office, "
            "Dean (Academics), Dean (Research), and finance offices. "
            "Office hours: Monday–Friday, 9 AM to 5 PM."
        ),
    },
    {
        "name": "Lecture Hall Complex (LHC)",
        "aliases": ["lhc", "lecture hall", "lh", "lecture complex", "classroom", "classrooms"],
        "block": "Lecture Hall Complex",
        "floor": "Ground, First & Second Floor",
        "landmark": "Near the department blocks",
        "description": (
            "The Lecture Hall Complex (LHC) contains the main classrooms (LH-1 through LH-6). "
            "It is centrally located between the department blocks."
        ),
    },
    {
        "name": "Canteen / Cafeteria",
        "aliases": ["canteen", "cafeteria", "food", "eat", "mess", "dining"],
        "block": "Near Boys Hostel (Canteen Block)",
        "floor": "Ground Floor",
        "landmark": "Between the boys' hostel and the academic blocks",
        "description": (
            "The main canteen is located near the boys' hostel, between the hostel area and "
            "the academic buildings. It serves meals and snacks. "
            "Hostel mess is inside each hostel block."
        ),
    },
    {
        "name": "Boys Hostel",
        "aliases": ["boys hostel", "male hostel", "gents hostel", "hostel", "bh"],
        "block": "Hostel Zone (North Campus)",
        "floor": "Ground to Third Floor",
        "landmark": "Northern side of the campus",
        "description": (
            "The boys' hostels (BH-1 through BH-6) are in the northern part of the campus. "
            "Each block has rooms, common rooms, and a mess."
        ),
    },
    {
        "name": "Girls Hostel",
        "aliases": ["girls hostel", "female hostel", "ladies hostel", "women hostel", "gh"],
        "block": "Hostel Zone (East Campus)",
        "floor": "Ground to Third Floor",
        "landmark": "Eastern side of the campus",
        "description": (
            "The girls' hostels (GH-1, GH-2) are in the eastern section of the campus. "
            "They include a dedicated mess and recreation room."
        ),
    },
    {
        "name": "Health Centre / Medical Centre",
        "aliases": ["health centre", "medical", "clinic", "doctor", "hospital",
                    "infirmary", "healthcare", "dispensary"],
        "block": "Health Centre Building",
        "floor": "Ground Floor",
        "landmark": "Near the main gate / guest house area",
        "description": (
            "The campus health centre provides basic medical services. "
            "A resident doctor is available during office hours. "
            "For emergencies outside hours, contact the security desk."
        ),
    },
    {
        "name": "Guest House",
        "aliases": ["guest house", "guesthouse", "visitor accommodation", "svpb"],
        "block": "SVPB Guest House",
        "floor": "Ground & First Floor",
        "landmark": "Near the main gate",
        "description": (
            "The SVNIT Paryatak Bhavan (SVPB) Guest House is near the main gate. "
            "It accommodates official visitors and parents. "
            "Bookings are managed through the administrative section."
        ),
    },
    {
        "name": "Sports Complex / Physical Education Centre",
        "aliases": ["sports", "gym", "ground", "playground", "football", "cricket",
                    "basketball", "badminton", "pec", "physical education"],
        "block": "Sports Complex",
        "floor": "Ground",
        "landmark": "Southern/Western campus area",
        "description": (
            "The sports complex includes outdoor grounds (cricket, football, volleyball), "
            "indoor courts (badminton, table tennis), and a gymnasium. "
            "The Physical Education Centre office is adjacent to the courts."
        ),
    },
    {
        "name": "Central Computing Centre (CCC)",
        "aliases": ["ccc", "computer centre", "computing centre", "internet lab"],
        "block": "CCC Building",
        "floor": "Ground & First Floor",
        "landmark": "Near the library",
        "description": (
            "The Central Computing Centre provides computing and internet facilities. "
            "It manages the campus network, email services, and open-access labs."
        ),
    },
    {
        "name": "Main Gate",
        "aliases": ["main gate", "entrance", "gate", "entry", "exit", "front gate"],
        "block": "Main Gate",
        "floor": "Ground",
        "landmark": "Ichhanath, Surat – the primary campus entrance",
        "description": (
            "The main gate is on the Dumas Road side (Ichhanath, Surat). "
            "Security is present 24x7. Visitors must register at the security office. "
            "The nearest bus stop is Ichhanath."
        ),
    },
    {
        "name": "ATM",
        "aliases": ["atm", "cash machine", "bank atm", "cash"],
        "block": "Near Administrative Block",
        "floor": "Ground",
        "landmark": "Near the administrative block or main gate area",
        "description": (
            "An ATM is typically available near the administrative block. "
            "Check the latest notices on the SVNIT website for exact location and timings."
        ),
    },
    {
        "name": "Examination Hall",
        "aliases": ["exam hall", "examination hall", "exam centre", "examination centre",
                    "exam room"],
        "block": "LHC / Dedicated Exam Halls",
        "floor": "Ground & First Floor",
        "landmark": "Lecture Hall Complex",
        "description": (
            "End-semester examinations are usually held in the Lecture Hall Complex (LHC) "
            "and in the department-specific labs. Seating arrangements are posted on the "
            "SVNIT notice board and Moodle before exams."
        ),
    },
]

# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _find_location(query: str) -> list[dict]:
    """Return the best matching location(s) for a query string."""
    q = query.lower()
    scored = []
    for loc in LOCATIONS:
        # Check exact alias match first
        for alias in loc["aliases"]:
            if alias in q:
                scored.append((1.0, loc))
                break
        else:
            # Fuzzy match against name and aliases
            best = max(
                _similarity(q, loc["name"]),
                *[_similarity(q, a) for a in loc["aliases"]],
            )
            if best > 0.4:
                scored.append((best, loc))

    scored.sort(key=lambda x: -x[0])
    # Return top 2 unique results
    seen = set()
    results = []
    for score, loc in scored:
        if loc["name"] not in seen:
            seen.add(loc["name"])
            results.append(loc)
            if len(results) == 2:
                break
    return results


# ---------------------------------------------------------------------------
# Campus navigator class
# ---------------------------------------------------------------------------


class CampusNav:
    def answer(self, question: str) -> dict:
        """Answer a campus navigation question."""
        locations = _find_location(question)
        if not locations:
            text = (
                "I couldn't find that location in my campus database. "
                "Please try a more specific name (e.g. 'AI department', 'library', "
                "'boys hostel', 'health centre', 'CCC'). "
                "For detailed directions, check the campus map on the SVNIT website: "
                "https://www.svnit.ac.in/web/campus.php"
            )
        else:
            parts = []
            for loc in locations:
                parts.append(
                    f"### {loc['name']}\n"
                    f"**Block / Building:** {loc['block']}  \n"
                    f"**Floor:** {loc['floor']}  \n"
                    f"**Landmark:** {loc['landmark']}  \n\n"
                    f"{loc['description']}"
                )
            text = "\n\n---\n\n".join(parts)
            if len(locations) > 1:
                text = "I found multiple relevant locations:\n\n" + text

        return {
            "answer": text,
            "sources": [{"source": "Campus Navigation Database",
                         "text": "Static SVNIT campus location database",
                         "score": 1.0, "page": None}],
            "refused": False,
            "context": "",
        }
