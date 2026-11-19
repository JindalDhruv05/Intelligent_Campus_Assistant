"""Show what the retriever returns for questions: use it to inspect retrieval and tune MIN_SCORE.
    python probe.py "What are the library opening hours?"
    python probe.py --out-of-scope        # probe every out-of-scope question in eval/questions.csv
Needs the index (python ingest.py) and the embedding model; does NOT call the LLM."""
import sys
import pandas as pd
import config
from rag import CampusRAG, label

args = sys.argv[1:]
if args and args[0] == "--out-of-scope":
    df = pd.read_csv(config.BASE / "eval" / "questions.csv").fillna("")
    qs = list(df[df.type == "out_of_scope"].question)
else:
    qs = args
if not qs:
    raise SystemExit(__doc__)

rag = CampusRAG()
for q in qs:
    hits = rag.retrieve(q, 3)
    best = hits[0]["score"] if hits else 0
    verdict = "WOULD REFUSE" if best < config.MIN_SCORE else "would answer"
    print(f"\nQ: {q}\n   best score {best:.2f} -> {verdict} (threshold {config.MIN_SCORE})")
    for h in hits:
        print(f"   {h['score']:.2f}  {label(h)[:110]}\n         {h['text'][:140]}...")
