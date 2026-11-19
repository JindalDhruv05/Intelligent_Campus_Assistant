# Intelligent Campus Assistant (RAG) — SVNIT Surat

> **Course:** Introduction to LLMs · B.Tech AI Sem 7 · SVNIT Surat · 2026

LLM-powered campus assistant for B.Tech AI students. Answers grounded in
official SVNIT documents (Academic Regulations, Curriculum, Honors Scheme).
Includes timetable clash detection, course recommendations, campus navigation,
conversation memory, and a full evaluation pipeline.

> [!NOTE]
> The FAISS index (`store/`) and source PDFs (`data/docs/*.pdf`) are not
> committed to this repo (too large). Run `python ingest.py` after adding
> the documents to regenerate the index locally.

---

## 1. Setup (once)

```bash
# Install Ollama: https://ollama.com
ollama pull qwen2.5:3b

python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## 2. Documents

Place in `data/docs/`:
- `Academic_Regulation-2023-24.pdf`
- `B_Tech-AI-2023-24-Curriculum-First_to_Fouth_Year-Syllabus.pdf`
- `B_Tech_Honors_Scheme_Syllabus.md` (already included; it is a text transcription)

Do NOT put the scanned `B_Tech_Honors_Scheme_Syllabus.pdf` there.

## 3. Build the index and run

```bash
python ingest.py
python check_evidence.py        # validate question set evidence phrases
streamlit run app.py            # full multi-feature UI
```

## 4. What the app can do

| Feature | Module | How to use |
|---------|--------|------------|
| RAG document Q&A (attendance, grades, syllabus…) | `rag.py` | Just ask in chat |
| Assignment support (syllabus-grounded guidance) | `rag.py` + `assignment_support.py` | Ask "help me with my X assignment" |
| Timetable clash checker (natural language) | `timetable.py` | "Do AI463 and AI465 clash?" |
| Course recommendation (interest / prerequisites) | `recommender.py` | "Which electives suit NLP?" |
| Campus navigation (buildings, facilities) | `navigation.py` | "Where is the AI department?" |
| Conversation memory + follow-up rewriting | `memory.py` | Automatic (toggle in sidebar) |
| Question router | `router.py` | Automatic |
| RAG off baseline (hallucination demo) | `rag.py` | Toggle in sidebar |

## 5. Evaluation

```bash
python evaluate.py              # full run, saves eval/results.csv
python evaluate.py --limit 25   # quick test
python evaluate.py --rerank     # ablation: add cross-encoder reranker

# Chunk size ablations (rebuild index first):
python ingest.py --chunk-words 100 --overlap 20 && python evaluate.py --out eval/results_c100.csv
python ingest.py --chunk-words 300 --overlap 60 && python evaluate.py --out eval/results_c300.csv

# Retrieval k ablations:
python evaluate.py --k 2 --out eval/results_k2.csv
python evaluate.py --k 6 --out eval/results_k6.csv

# Refusal threshold ablations:
python evaluate.py --min-score 0.35 --out eval/results_ms35.csv
python evaluate.py --min-score 0.55 --out eval/results_ms55.csv
```

## 6. Timetable CLI

```bash
python timetable.py AI463 AI465     # clash check for two courses
python timetable.py                  # show full timetable
```

## 7. Web crawler (optional)

```bash
python crawler.py --max-pages 30    # small test run
python crawler.py                    # full crawl (settings in config.py)
python check_evidence.py && python ingest.py
python probe.py --out-of-scope      # check what crawler added
```

## 8. Tips for a 4 GB GPU

- After `ollama pull qwen2.5:3b`, confirm: run a query then `ollama ps` → 100% GPU.
- If out of memory: lower `NUM_CTX` in `config.py` to 2048.
- Close other GPU programs before the demo.

## 9. File overview

```text
app.py               Multi-tab Streamlit UI (router, memory, all modules integrated)
rag.py               RAG pipeline (retrieve + ground + cite + refuse + assignment support)
router.py            Question classifier (timetable / recommend / navigate / document)
timetable.py         Clash checker + natural-language timetable handler
recommender.py       Interest/prerequisite-based course recommender
navigation.py        SVNIT campus location database + fuzzy search
memory.py            Conversation memory + follow-up query rewriter
assignment_support.py Assignment-aware prompt augmentation
reranker.py          Optional cross-encoder reranker (ablation)
ingest.py            Document → chunks → FAISS index
evaluate.py          Evaluation pipeline (baseline vs RAG, metrics, ablations)
check_evidence.py    Validate evidence phrases before evaluation
probe.py             Inspect retrieval for a set of queries
config.py            All settings in one place
crawler.py           SVNIT website crawler
data/
  docs/              PDF and markdown documents to index
  timetable.csv      B.Tech AI 7th-sem sample timetable
eval/
  questions.csv      55-question evaluation set (45 in-scope, 10 out-of-scope)
  results.csv        Output of evaluate.py
store/               FAISS index + chunks.json (created by ingest.py)
```
