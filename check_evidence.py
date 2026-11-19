"""Verify every evidence_phrase in eval/questions.csv appears in the document chunks.
Works BEFORE building the index (no models needed). Run:  python check_evidence.py
Uses the chunk settings from config.py. Phrases reported here would always count as
retrieval misses in evaluate.py, so fix them (copy a short exact phrase from the document)."""
import re
import pandas as pd
import config
from ingest import build_chunks


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


corpus = {}
for c in build_chunks():
    corpus.setdefault(c["source"], []).append(norm(c["text"]))

df = pd.read_csv(config.BASE / "eval" / "questions.csv").fillna("")
bad = 0
ins = df[df.type == "in_scope"]
for _, r in ins.iterrows():
    phrase = norm(r["evidence_phrase"])
    texts = corpus.get(r["source_doc"], [])
    if not texts:
        print(f"Q{r['id']}: source_doc '{r['source_doc']}' not found in data/docs"); bad += 1
    elif not any(phrase in t for t in texts):
        print(f"Q{r['id']}: phrase not found in any chunk -> {r['evidence_phrase']!r}"); bad += 1
print(f"\nChecked {len(ins)} in-scope questions across {len(corpus)} documents: {bad} problems.")
