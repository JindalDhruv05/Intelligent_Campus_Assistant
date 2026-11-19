# Intelligent Campus Assistant: Team Reference

Course: Introduction to LLMs, B.Tech AI, Semester 7, SVNIT Surat
Team: [Member 1], [Member 2], [Member 3]
Guide: [Name]

This file is the single source of truth for the team. Read sections 1 to 4 to understand the project. Use sections 10 to 14 to prepare for evaluation. If something here disagrees with the code, trust the code and fix this file.

---

## 1. What the project is

An LLM-powered assistant that answers SVNIT campus and academic questions using Retrieval-Augmented Generation (RAG). It retrieves relevant passages from official SVNIT documents, gives them to a local LLM, and returns an answer with the source file and page. If the documents do not contain the answer, it refuses instead of guessing.

The course emphasis is critical evaluation, so the project is built around measurement:

- Accuracy: is the answer correct against a ground-truth question set?
- Explainability: every answer shows the chunks, file names, pages and similarity scores behind it.
- Reliability: the system refuses out-of-scope questions, and we compare it against the same LLM used alone (baseline).

Core claim to demonstrate: grounding a small local LLM in institutional documents reduces hallucination, and we can measure by how much.

---

## 2. Honest status (use this for slides and the viva)

| Feature | Status | Notes |
|---|---|---|
| Document ingestion, chunking, embeddings, FAISS index | ✅ Done | `ingest.py` |
| Header/footer stripping for repeated PDF boilerplate | ✅ Done | Improved retrieval on the curriculum PDF |
| Retrieval, grounded prompt, citations | ✅ Done | `rag.py` |
| Refusal when retrieval confidence is low | ✅ Done | Threshold `MIN_SCORE` is an untuned starting value |
| Streamlit chat UI with RAG on/off toggle and sources panel | ✅ Done | `app.py` (fully rewritten post-mid) |
| Evaluation pipeline (LLM-only vs RAG) | ✅ Done | `evaluate.py`; run and record real numbers |
| Policy queries | ✅ Working | Academic Regulations 2023-24 |
| Course information Q&A | ✅ Working | B.Tech AI curriculum and Honors scheme |
| Timetable clash checker (standalone CLI) | ✅ Done | `timetable.py` |
| **Timetable questions in chat** | ✅ Done | Integrated via router; `timetable.py` handles NL queries |
| **Course recommendation (interest / prerequisite)** | ✅ Done | `recommender.py` with LLM + keyword fallback |
| **Assignment support** | ✅ Done | `assignment_support.py` augments RAG prompt for assignment queries |
| **Campus navigation** | ✅ Done | `navigation.py` with SVNIT location database |
| **Conversation memory** | ✅ Done | `memory.py`; follow-up rewriting via LLM |
| **Question router** | ✅ Done | `router.py`; pattern-based + LLM fallback |
| **Reranking (cross-encoder)** | ✅ Done | `reranker.py`; ablation via `--rerank` flag in evaluate.py |
| Web crawler | ✅ Done | `crawler.py` (post-mid, not part of core eval) |

Rules for presenting:
- Never claim "course recommendation" or "timetable planning" as working features. Say "course information Q&A" and "timetable clash checker (standalone, integration planned)".
- Never show a number you did not compute from `evaluate.py`.

---

## 3. Architecture

```
SVNIT documents (PDF, MD)
        |
   [ingest.py]
   read pages -> strip repeated header/footer lines -> clean whitespace
   -> split into 200-word chunks with 40-word overlap
   -> embed with BAAI/bge-small-en-v1.5 (normalized vectors)
   -> FAISS IndexFlatIP index + chunks.json (source, page, text)
        |
   [app.py] user question
        |
   [memory.py] rewrite follow-up questions into standalone queries
        |
   [router.py] classify question
   timetable ─────────────> [timetable.py]  clash checker + schedule lookup
   recommend ─────────────> [recommender.py] interest/prereq-based suggestions
   navigate  ─────────────> [navigation.py]  SVNIT building/facility lookup
   document  ─────────────> [rag.py]
                            embed question (BGE query prefix)
                            -> top-k chunks by cosine similarity (FAISS)
                            -> optional [reranker.py] cross-encoder rerank
                            if best score < MIN_SCORE -> refuse (no LLM call)
                            if assignment query -> augment system prompt
                            else build grounded prompt + numbered context
                            -> Ollama qwen2.5:3b (temp 0) -> answer [1][2]
        |
   [app.py] Streamlit chat: route badge, answer, sources panel (file, page, score)
   [evaluate.py] baseline vs RAG evaluation with 5 metrics + ablations
```

Why each choice:
- Local model via Ollama: free, offline, reproducible demo, no API keys. A 4 GB RTX 3050 laptop GPU fits a 3B model at 4-bit quantization (about 2 GB). A 7B model would spill to the CPU and run slowly.
- `bge-small-en-v1.5`: small, strong for retrieval, runs on CPU. Vectors are normalized, so inner product equals cosine similarity.
- FAISS `IndexFlatIP`: exact search. With a few hundred chunks there is no need for approximate indexes.
- Temperature 0: deterministic answers, which makes evaluation repeatable.
- Refuse-before-generate: if retrieval confidence is low, we skip the LLM entirely. This is the main anti-hallucination mechanism.

---

## 4. Tech stack

Python, pypdf, sentence-transformers, FAISS (cpu), Ollama (python client), Streamlit, pandas, numpy.
LLM: `qwen2.5:3b` (alternatives: `llama3.2:3b`, `phi3:mini`).
Embeddings: `BAAI/bge-small-en-v1.5`.

---

## 5. Project folder and what every file does

```
campus_assistant/
  config.py          all settings in one place
  ingest.py          build the index from data/docs
  rag.py             retrieval + grounded answering + baseline
  app.py             Streamlit chat UI
  evaluate.py        baseline vs RAG evaluation
  check_evidence.py  validate the question set against the documents
  timetable.py       standalone clash checker
  download_docs.py   optional downloader (not needed for our current files)
  requirements.txt
  README.md
  data/
    docs/            documents to index (see section 6)
    timetable.csv    FAKE sample timetable for the clash demo
  eval/
    questions.csv    55-question evaluation set
    results.csv      created by evaluate.py
  store/             created by ingest.py (index.faiss, chunks.json)
```

### config.py
`EMBED_MODEL`, `LLM_MODEL = "qwen2.5:3b"`, `NUM_CTX = 4096`, `CHUNK_WORDS = 200`, `OVERLAP_WORDS = 40`, `TOP_K = 4`, `MIN_SCORE = 0.45`.

### ingest.py
- Reads `.pdf` page by page (with pypdf), and `.md`/`.txt` as one page.
- Skips PDFs with no extractable text and prints a warning (scanned PDFs).
- `strip_repeated_lines`: removes any line that repeats on at least half of a document's pages, ignoring digits. This removes page headers and footers.
- `chunk_words`: word-based sliding window, size and overlap from config or `--chunk-words` and `--overlap`.
- Saves `store/index.faiss` and `store/chunks.json`.

### rag.py
- `CampusRAG.retrieve(query, k)`: embeds the query with the BGE instruction prefix, returns chunks with scores.
- `CampusRAG.answer(query, k, min_score)`: refuses if best score is below threshold, otherwise builds the grounded prompt and calls the LLM. Returns answer, sources, a refused flag and the context.
- `CampusRAG.answer_baseline(query)`: the same LLM with no documents. This is our comparison baseline.
- `looks_like_refusal`: phrase check ("i don't know", "not sure", etc.) used to detect refusals in both modes.

### app.py
Streamlit chat. Sidebar: RAG on/off toggle, top-k slider (1 to 8), refusal threshold slider. Each answer has a "Sources used" expander showing file, page, score and the chunk text. There is no memory between questions.

### evaluate.py
Runs every question in `eval/questions.csv` through baseline and RAG and writes `eval/results.csv`. See section 8 for exact metric definitions. `--limit N` evaluates a random subset. `--k` and `--min-score` override retrieval settings.

### check_evidence.py
Builds chunks (no models needed) and checks that each in-scope question's `evidence_phrase` appears in a chunk of its `source_doc`. A phrase that is not found would always count as a retrieval miss, so fix it first.

### timetable.py
Loads `data/timetable.csv` and reports overlapping slots on the same day. The CSV is fake sample data, not the real SVNIT timetable. Not connected to the chat.

---

## 6. Data sources

Documents currently in `data/docs/`:

1. `Academic_Regulation-2023-24.pdf` (62 pages). SVNIT academic regulations, 58th Senate, 31 May 2023. Covers attendance, grading, examinations, registration, credits, Minor/Honor, Ph.D. rules. Applies to the 2023 admission batch.
2. `B_Tech-AI-2023-24-Curriculum-First_to_Fouth_Year-Syllabus.pdf` (112 pages). B.Tech AI scheme for all 8 semesters, the elective list, and full syllabi, including Introduction to Large Language Models (AI463) and Agentic AI (AI465).
3. `B_Tech_Honors_Scheme_Syllabus.md`. A text transcription of the Honors in AI scheme and syllabi.

Things the team must know about the data:
- The Honors PDF is a scanned image with no text layer. `ingest.py` would read nothing from it. We transcribed it into the `.md` file by reading the page images. The transcription may contain small errors, so proofread it against the scan. Keep the scanned PDF out of `data/docs/`.
- Every page of the curriculum PDF repeats a long header block. Before header stripping, this diluted chunks and hurt retrieval. Header stripping fixed first-semester course listings that had been answered wrongly.
- The curriculum document is internally inconsistent in places. For example, the elective list gives Quantum Computing the code `AI352` but its syllabus page says `AI353`, and Cloud Computing appears twice (`AI304` core and `AI353` elective). Our question set avoids these items. They make a good limitations point.
- Curricula for other years differ from the 2023 batch (for example course codes and schemes). Do not index several years together, because answers would conflict.
- No hostel, library, mess or placement documents are indexed. Questions about them are expected to be refused.

---

## 7. How the pipeline behaves (details worth knowing)

Chunking: 200 words with 40-word overlap, per page. A chunk never crosses a page boundary, so a table that spans pages can be split. The overlap reduces the chance that an answer is cut across two chunks.

Retrieval: cosine similarity between the query vector and chunk vectors, top-k = 4 by default. BGE models expect a retrieval instruction prefix on queries, which `rag.py` adds.

Refusal: if the highest similarity is below `MIN_SCORE`, the bot answers "I don't know based on the available campus documents..." without calling the LLM. This value (0.45) is a guess and must be tuned:
- Too high: the bot refuses valid questions (false refusals).
- Too low: the bot answers out-of-scope questions (hallucination risk).
To tune it, run `python evaluate.py --min-score 0.35`, `0.45`, `0.55` and compare correctness against correct refusals.

Grounded prompt: the system prompt tells the model to use only the numbered context, cite passages as [1], [2], and reply with an exact "I don't know" sentence when the context lacks the answer. Even so, a 3B model can occasionally ignore this. That is why we measure faithfulness.

Known weakness: a question can pass the similarity threshold because it is related to a document, yet the answer is not there (for example library opening hours, when the regulations only mention library dues). Those questions test the second line of defense, the prompt. They are in the question set on purpose.

---

## 8. Evaluation design

Question set: `eval/questions.csv`, 55 questions.
- 45 in-scope: 21 on the Academic Regulations, 18 on the B.Tech AI curriculum, 6 on Honors.
- 10 out-of-scope, which the system should refuse. Examples: placement package, director's name, Compiler Design syllabus, Ph.D. stipend amount (a hard case, because the regulations mention stipends but give no amount).

Columns: `id`, `question`, `ground_truth`, `type` (in_scope or out_of_scope), `source_doc`, `evidence_phrase` (a short exact phrase from the document that contains the answer).

Ground truths were written from the actual document text. Spot-check a few against the PDFs before presenting.

Metrics printed by `evaluate.py`:

| Metric | Definition |
|---|---|
| Answer correctness (in-scope) | Share of in-scope questions where the local LLM judge says the answer agrees with the ground truth. A refusal to an in-scope question counts as incorrect |
| Faithfulness to context (RAG only) | Share of non-refused in-scope RAG answers where the judge says every claim is supported by the retrieved context |
| Retrieval hit rate (RAG only) | Share of in-scope questions where the `evidence_phrase` appears in the retrieved chunks |
| Hallucination rate | Out-of-scope: the system answered instead of refusing. In-scope: it answered confidently but incorrectly. Averaged over all questions |
| Correct refusals (out-of-scope) | Share of out-of-scope questions that were refused |

Extra numbers to compute yourselves from `results.csv`: false refusal rate (in-scope questions that were refused), and correctness among questions where retrieval hit was True versus False. The second one separates retrieval errors from generation errors, which is a strong analysis point.

Limitations of this evaluation (state them openly):
- The judge for correctness and faithfulness is the same small 3B model, so its verdicts are imperfect. Hand-check 10 to 15 rows and report the agreement.
- Refusal detection uses phrase matching.
- 55 questions is a small sample. Report numbers as indicative.
- Questions were written by the team from the same documents, so they may be easier than real student questions.

Ablations to run and report (each is a table row):
- Header stripping off versus on (set `frac=2.0` in `strip_repeated_lines` to disable it). This is a real fix we found, so it is a strong result.
- Chunk size: `python ingest.py --chunk-words 100 --overlap 20` and `--chunk-words 300 --overlap 60`, then `python evaluate.py`.
- Top-k: `python evaluate.py --k 2` and `--k 6`.
- Refusal threshold: `--min-score 0.35` and `0.55`.

Save `results.csv` under a new name after every run (for example `results_k2.csv`) so runs are not overwritten.

---

## 9. Setup and run

One time:
1. Install Ollama from https://ollama.com and run `ollama pull qwen2.5:3b`.
2. Create and activate a virtual environment, then `pip install -r requirements.txt`.
3. Put the three documents from section 6 in `data/docs/`.

Every time:
```
python check_evidence.py       # validate the question set
python ingest.py               # build the index (rerun after any document or chunk change)
streamlit run app.py           # chat UI
python evaluate.py --limit 25  # quick test run
python evaluate.py             # full run, then copy eval/results.csv somewhere safe
python timetable.py CS401 AI402 ML403   # standalone clash demo (sample data)
```

---

## 10. Known issues and limitations

- The chat cannot access `timetable.csv`. The chat only searches indexed documents. Timetable logic needs a router.
- No course recommendation logic. Questions like "which electives suit NLP?" get only whatever the retrieved text happens to say.
- No conversation memory. A follow-up like "and for sixth semester?" fails.
- Tables in PDFs are flattened into text, so row and column structure is lost. Listing questions can still go wrong when a table is split across chunks.
- Small 3B model: can ignore instructions, miscount, or merge items. Reported in faithfulness.
- Refusal threshold is untuned.
- The judge model is weak (section 8).
- The Honors document is a transcription from a scan and needs a proofread.
- Source documents contain inconsistencies (section 6).
- Documents date from the 2023 curriculum. Rules may change in later regulations. Always cite the document version in answers or slides.
- Privacy: no student personal data is used. All processing is local and offline.

---

## 11. Remaining work (suggested priority and owners)

### Completed post-mid:
5. ✅ **Question router** (`router.py`): pattern + LLM-fallback classifier dispatches to timetable, recommender, navigation or RAG.
6. ✅ **Course recommendation** (`recommender.py`): LLM over structured curriculum catalogue; keyword fallback for demos without Ollama.
7. ✅ **Conversation memory** (`memory.py`): ring buffer of 5 turns; LLM rewrites follow-up queries into standalone queries.
8. ✅ **Reranking** (`reranker.py`): cross-encoder (ms-marco-MiniLM-L-6-v2); plug in via `--rerank` flag in evaluate.py.
9. ✅ **Assignment support** (`assignment_support.py`): detected assignment queries get an augmented system prompt.
10. ✅ **Campus navigation** (`navigation.py`): fuzzy-matched SVNIT location database with directions.
11. ✅ **Full multi-module UI** (`app.py`): route badge, memory toggle, router toggle, all modules wired in.
12. ✅ **Timetable chat integration**: `timetable.py` now handles natural-language timetable queries from chat.

### Still to do before final evaluation:
- **Run `evaluate.py`** and fill in real numbers on the slides (do `--limit 25` for a quick test first).
- **Run the 4 ablations** (chunk size, top-k, threshold, reranker) and add a comparison table.
- **Hand-check 10–15 rows** of `eval/results.csv` to validate judge accuracy.
- **Update timetable CSV** with the real SVNIT 7th-sem timetable (current file has realistic sample data).
- **Final report** and demo screenshots.

---

## 12. Slide deck

The deck is a 7-slide Artifact: https://claude.ai/artifact/DTbMivuXzrTVkdUQz7pTvK
Slides: cover, problem and objectives, architecture, scope and data sources, progress, evaluation, remaining plan.

Text to fix before presenting:
- Slide 4: data sources should name the SVNIT Academic Regulations, B.Tech AI curriculum and Honors scheme. Change "Course recommendation: Prototype" to "Course information Q&A: prototype; recommendation: planned". Change "Timetable planning: In progress" to "Standalone clash checker; chat integration planned".
- Slide 5: fill the placeholders (3 documents, chunk settings, 55 test questions, header-stripping fix).
- Slide 6: fill every `[__]` with numbers from `evaluate.py`. If you did not run something, leave it out.
- Slide 7: set real owners and dates.

---

## 13. Viva preparation (short answers)

1. Why RAG instead of fine-tuning? The facts live in documents that change. RAG updates by re-indexing, gives citations, needs no training data and does not risk the model memorizing wrong rules. Fine-tuning mostly teaches style, not reliable facts.
2. What is chunking and why overlap? Documents are split into pieces small enough to embed and retrieve precisely. Overlap keeps an answer from being cut between two chunks.
3. Why normalized embeddings and inner product? After normalization, inner product equals cosine similarity, so FAISS inner-product search ranks by cosine.
4. Why the BGE query prefix? BGE retrieval models were trained with an instruction on queries. Using it improves matching.
5. What does the refusal threshold do and what is the trade-off? It refuses when the best retrieval score is low. High values cause false refusals, low values allow hallucination. We tune it with the evaluation set.
6. How do you measure hallucination? Out-of-scope answers that are not refusals, and in-scope answers that are confident but incorrect. We also measure faithfulness to the retrieved context.
7. What are the weaknesses of your evaluation? The judge is a small model, the set has 55 questions written by us, and refusal detection is phrase-based. We hand-check a sample.
8. Difference between retrieval hit rate and correctness? Hit rate checks whether the right text was retrieved. Correctness checks the final answer. If hit is true but the answer is wrong, the LLM failed. If hit is false, retrieval failed.
9. Why temperature 0? Deterministic output makes evaluation repeatable.
10. Why a 3B model? A 4 GB laptop GPU. A 3B model at 4-bit fits entirely on the GPU. Larger models would be slow.
11. What did header stripping fix? Every curriculum page repeated a long header, which diluted chunk embeddings and polluted context. Removing lines that repeat across pages improved retrieval, and first-semester listings became correct.
12. How does the system explain its answers? It shows the retrieved chunks with file, page and score, and the answer cites passage numbers.
13. What happens to a question outside the documents? Usually low similarity, so it refuses without calling the LLM. If it passes the threshold, the prompt tells the model to say it does not know.
14. How did you add timetable support in chat? A pattern-based router detects course codes or words like "clash" and routes to the timetable module. RAG is the wrong tool for structured scheduling; we keep them separate.
15. What are the privacy considerations? Only institutional documents, no student data, and all processing is local.
16. Why FAISS flat index? Exact search is simple and fast for a small corpus. Approximate indexes matter at much larger scale.
17. How does the course recommender work? It uses a structured curriculum catalogue (course name, code, semester, credits, tags, prerequisites). For LLM mode it passes the whole catalogue as context. For the keyword fallback it matches user interest words against course tags.
18. How does conversation memory prevent the RAG pipeline from breaking? The memory module rewrites follow-up questions into standalone queries using the LLM before the query reaches the retriever, so the retriever always gets a self-contained question.
19. What is the question router and how does it classify? It uses compiled regex patterns. No model overhead for most queries. A second LLM-based path handles ambiguous cases. Patterns check for course codes, the word "clash", interests, location words, etc.
20. What is a cross-encoder reranker and how does it differ from bi-encoder retrieval? A bi-encoder (BGE) encodes query and document independently and compares them with dot product — fast but approximate. A cross-encoder takes (query, document) together and scores them jointly — more accurate but 10–100x slower. We use it only as an ablation, not by default.

---

## 14. Five-minute demo script

1. State the problem in one sentence and show the architecture slide.
2. In the app, ask: **"What is the minimum attendance required in a course?"** Show the answer and open "Sources used".
3. Ask: **"What happens if my attendance is below 75% but above 60%?"** Show it cites the right page.
4. Ask: **"Which practical in Introduction to Large Language Models covers RAG?"** (Our own course — lands well.)
5. Ask a timetable question: **"Do AI463 and AI465 clash?"** The route badge shows "📅 Timetable".
6. Ask a recommendation: **"Which electives are good for someone interested in NLP?"** Route badge: "🎯 Course Recommendation".
7. Ask navigation: **"Where is the AI department?"** Route badge: "🗺️ Campus Navigation".
8. Ask an out-of-scope question: **"What was the average placement package last year?"** Show the refusal.
9. Turn RAG off and ask the attendance question again. Show the baseline LLM answering confidently without sources.
10. Show the evaluation table with real numbers and one ablation (header stripping).
11. Close with honest limitations and what was left out.

Backup plan: have screenshots of steps 2–9 in the slides in case the laptop or model misbehaves. Start Ollama and run one question 10 minutes before the demo so the model is loaded.

---

## 15. Troubleshooting

- `check_evidence.py` reports a phrase not found: open the PDF, copy a short exact phrase from the answer text into `questions.csv`. PDF extraction can change spacing (for example "160 -170").
- Scanned PDF warning: the file has no text. Provide a text or markdown version, or add OCR.
- Out of GPU memory: lower `NUM_CTX` to 2048 or use `llama3.2:3b`. Close other GPU programs.
- Model seems slow: run `ollama ps` and confirm it says 100% GPU.
- The bot refuses too often or answers too much: adjust the threshold slider in the sidebar, then set a good value in `config.py`.
- Wrong answers after changing documents: rerun `python ingest.py`. The index does not update on its own.
- `evaluate.py` takes long: use `--limit 25` while testing.

---

## 16. Decision log (for the report)

- Local `qwen2.5:3b` chosen for the 4 GB RTX 3050 laptop: fits fully on GPU, free, offline.
- Refuse-before-generate chosen as the primary hallucination control.
- Header/footer stripping added after first-semester course listings returned wrong answers. Cause: a repeated header diluted the chunk embeddings.
- Honors scheme transcribed to markdown because the source PDF is an image scan.
- Only the 2023 curriculum indexed, to avoid conflicting course schemes from other years.
- Timetable integrated into chat via the router. Timetable queries use the CSV directly (not RAG), because RAG is unreliable for structured scheduling data.
- Course recommender uses a hand-crafted curriculum catalogue (not the FAISS index) so that prerequisite chains are always correct.
- Conversation memory uses LLM-based follow-up rewriting so the retriever always receives a standalone question.
- Campus navigation uses a static location database with fuzzy matching, not RAG, because precise location facts need to be reliable and cannot hallucinate.
- Cross-encoder reranker added as an ablation experiment; not enabled by default due to latency.
