"""Run with:  streamlit run app.py

Full campus assistant with:
  - RAG chat (grounded document answers with citations)
  - Question router (timetable / recommendation / navigation / document)
  - Conversation memory with follow-up rewriting
  - Timetable chat integration
  - Course recommender
  - Campus navigation
  - RAG on/off toggle for hallucination demo
"""
import streamlit as st
import config
from rag import CampusRAG
from router import classify_llm
from timetable import TimetableHandler
from recommender import CourseRecommender
from navigation import CampusNav
from memory import ConversationMemory

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Campus Assistant – SVNIT",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_rag():
    return CampusRAG()


@st.cache_resource
def load_timetable():
    # Pass rag.llm so the handler can parse course names with the LLM
    return TimetableHandler(llm_fn=rag.llm)


@st.cache_resource
def load_nav():
    return CampusNav()


rag = load_rag()
timetable_handler = load_timetable()
nav_handler = load_nav()

# CourseRecommender gets the LLM once rag is loaded
recommender = CourseRecommender(llm_fn=rag.llm)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(llm_fn=rag.llm, max_turns=5)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    use_rag = st.toggle("🔍 RAG (grounded answers)", value=True,
                        help="Turn off to show the raw LLM without documents (hallucination demo).")
    use_memory = st.toggle("🧠 Conversation memory", value=True,
                           help="Rewrite follow-up questions using conversation history.")
    use_router = st.toggle("🧭 Question router", value=True,
                           help="Automatically route timetable, navigation and recommendation queries.")
    k = st.slider("Top-k chunks", 1, 8, config.TOP_K)
    min_score = st.slider("Refusal threshold", 0.0, 1.0, config.MIN_SCORE, 0.01,
                          help="Lower = more answers; higher = more refusals.")

    st.divider()
    st.caption("**Route legend**")
    st.markdown(
        "📄 Document → RAG  \n"
        "📅 Timetable → Schedule module  \n"
        "🎯 Recommend → Curriculum adviser  \n"
        "🗺️ Navigate → Campus map"
    )

    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.session_state.memory.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎓 Intelligent Campus Assistant")
st.caption(
    "SVNIT Surat · Answers grounded in official documents · "
    "Timetable, course recommendations and campus navigation included."
)

# ---------------------------------------------------------------------------
# Source card renderer
# ---------------------------------------------------------------------------

def render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander("📚 Sources used", expanded=False):
        for n, s in enumerate(sources, 1):
            where = s.get("title") or s["source"]
            if s.get("page"):
                where += f", page {s['page']}"
            score_str = f" (score {s['score']:.2f})" if s.get("score") and s["score"] < 1.0 else ""
            st.markdown(f"**[{n}] {where}**{score_str}")
            if s.get("url"):
                st.markdown(s["url"])
            if s.get("text"):
                st.caption(s["text"][:300] + ("…" if len(s.get("text", "")) > 300 else ""))


# ---------------------------------------------------------------------------
# Message renderer
# ---------------------------------------------------------------------------

def render_msg(msg: dict):
    with st.chat_message(msg["role"]):
        # Route badge (for non-document assistant messages)
        if msg.get("route") and msg["route"] != "document":
            badge = {
                "timetable": "📅 Timetable",
                "recommend": "🎯 Course Recommendation",
                "navigate": "🗺️ Campus Navigation",
            }.get(msg["route"], "")
            st.caption(badge)
        st.markdown(msg["text"])
        # Only show sources panel for RAG document answers
        if msg.get("route") == "document" or msg.get("role") == "user":
            render_sources(msg.get("sources", []))


# ---------------------------------------------------------------------------
# Display history
# ---------------------------------------------------------------------------
for m in st.session_state.history:
    render_msg(m)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
q = st.chat_input("Ask about attendance rules, courses, timetable, campus locations…")

if q:
    # --- Show user message ---
    user_msg = {"role": "user", "text": q}
    st.session_state.history.append(user_msg)
    render_msg(user_msg)

    with st.spinner("Thinking…"):
        # --- Memory: rewrite follow-ups ---
        standalone_q = q
        if use_memory and not st.session_state.memory.is_empty():
            standalone_q = st.session_state.memory.rewrite(q)

        # --- Router ---
        route = classify_llm(standalone_q, llm_fn=rag.llm) if use_router else "document"

        # --- Dispatch ---
        if not use_rag and route == "document":
            # Baseline mode: raw LLM, no retrieval
            res = rag.answer_baseline(standalone_q)
            route = "document"

        elif route == "timetable":
            res = timetable_handler.answer(standalone_q)

        elif route == "recommend":
            res = recommender.answer(standalone_q)

        elif route == "navigate":
            res = nav_handler.answer(standalone_q)

        else:
            # document: full RAG pipeline
            res = rag.answer(standalone_q, k, min_score)

    # --- Rewrite notice (if the query was rewritten) ---
    answer_text = res["answer"]
    if use_memory and standalone_q != q:
        note = f'> *(Interpreted as: "{standalone_q}")*\n\n'
        answer_text = note + answer_text

    # --- Store and render ---
    bot_msg = {
        "role": "assistant",
        "text": answer_text,
        "sources": res.get("sources", []),
        "route": route,
    }
    st.session_state.history.append(bot_msg)
    render_msg(bot_msg)

    # --- Update memory ---
    if use_memory:
        st.session_state.memory.add_turn(q, res["answer"])

# ---------------------------------------------------------------------------
# Footer info (shown when no messages yet)
# ---------------------------------------------------------------------------
if not st.session_state.history:
    st.info(
        "**Try asking:**\n\n"
        "- 📄 *What is the minimum attendance required?*\n"
        "- 📄 *Which practical in Introduction to Large Language Models covers RAG?*\n"
        "- 📅 *When is AI463? Do AI463 and AI465 clash?*\n"
        "- 🎯 *Which electives suit someone interested in NLP?*\n"
        "- 🎯 *What are the prerequisites for Introduction to Large Language Models?*\n"
        "- 🗺️ *Where is the AI department?*\n"
        "- 🗺️ *How do I get to the library?*\n"
        "- ❌ *What was the placement package last year?* (should be refused)"
    )
