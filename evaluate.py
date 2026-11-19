"""Compare baseline LLM vs RAG on eval/questions.csv.

Usage:
    python evaluate.py [--k 4] [--min-score 0.45] [--limit N]
    python evaluate.py --rerank        # ablation: add cross-encoder reranker
    python evaluate.py --no-header-strip  # ablation: rebuild index without stripping

Outputs eval/results.csv (per question) and prints the summary table.

NOTE: correctness and faithfulness are judged by the same small local model.
Spot-check 10–15 rows by hand and report that you did.
"""
import argparse
import pandas as pd
import config
from rag import CampusRAG


def judge_correct(rag, q, truth, ans):
    p = (f"Question: {q}\nReference answer: {truth}\nStudent answer: {ans}\n"
         "Does the student answer agree with the reference answer on the key facts? "
         "Reply with one word: CORRECT or INCORRECT.")
    return rag.llm([{"role": "user", "content": p}]).upper().startswith("CORRECT")


def judge_faithful(rag, ctx, ans):
    p = (f"Context:\n{ctx}\n\nAnswer: {ans}\n"
         "Is every factual claim in the answer supported by the context? Reply YES or NO.")
    return rag.llm([{"role": "user", "content": p}]).upper().startswith("YES")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--min-score", type=float, default=config.MIN_SCORE)
    ap.add_argument("--limit", type=int, default=0,
                    help="Evaluate a random subset of N questions")
    ap.add_argument("--rerank", action="store_true",
                    help="Ablation: add cross-encoder reranker")
    ap.add_argument("--out", type=str, default="eval/results.csv",
                    help="Output CSV path (default: eval/results.csv)")
    args = ap.parse_args()

    rag = CampusRAG()

    # Optional: apply reranker
    if args.rerank:
        try:
            from reranker import CrossEncoderReranker, patch_rag_with_reranker
            reranker = CrossEncoderReranker()
            patch_rag_with_reranker(rag, reranker, retrieve_k=args.k * 2, final_k=args.k)
            print(f"Reranker enabled (retrieve {args.k * 2}, rerank to {args.k})")
        except ImportError as e:
            print(f"Reranker not available: {e}")

    df = pd.read_csv(config.BASE / "eval" / "questions.csv").fillna("")
    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=0)

    rows = []
    for _, r in df.iterrows():
        in_scope = r["type"] == "in_scope"
        for mode in ("baseline", "rag"):
            res = (rag.answer_baseline(r["question"]) if mode == "baseline"
                   else rag.answer(r["question"], args.k, args.min_score))
            row = {
                "id": r["id"], "mode": mode, "type": r["type"],
                "question": r["question"], "answer": res["answer"],
                "refused": res["refused"], "correct": None,
                "faithful": None, "hit": None,
            }
            if in_scope and not res["refused"]:
                row["correct"] = judge_correct(rag, r["question"], r["ground_truth"], res["answer"])
            elif in_scope:
                row["correct"] = False
            if mode == "rag" and in_scope:
                phrase = str(r["evidence_phrase"]).lower()
                texts = " ".join(s["text"].lower() for s in res["sources"])
                names = {s["source"] for s in res["sources"]}
                row["hit"] = (phrase in texts) if phrase else (r["source_doc"] in names)
                if not res["refused"]:
                    row["faithful"] = judge_faithful(rag, res["context"], res["answer"])
            # hallucination: answered an out-of-scope question, or gave a wrong in-scope answer
            row["hallucinated"] = (
                (not res["refused"]) if not in_scope
                else (not res["refused"] and row["correct"] is False)
            )
            rows.append(row)
        print("done", r["id"])

    out = pd.DataFrame(rows)
    out_path = config.BASE / args.out
    out_path.parent.mkdir(exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    def pct(s):
        s = s.dropna()
        return f"{100 * s.astype(float).mean():.0f}%" if len(s) else "n/a"

    suffix = " (+rerank)" if args.rerank else ""
    print(f"\nSummary (k={args.k}, min_score={args.min_score}, n={len(df)}{suffix})")
    print(f"{'Metric':42}{'LLM only':>10}{'RAG':>10}")
    for lbl, fn in [
        ("Answer correctness (in-scope)", lambda d: d[d.type == "in_scope"]["correct"]),
        ("Faithfulness to context",       lambda d: d["faithful"]),
        ("Retrieval hit rate",             lambda d: d["hit"]),
        ("Hallucination rate (all)",       lambda d: d["hallucinated"]),
        ("Correct refusals (out-of-scope)", lambda d: d[d.type == "out_of_scope"]["refused"]),
    ]:
        b = pct(fn(out[out["mode"] == "baseline"]))
        g = pct(fn(out[out["mode"] == "rag"]))
        print(f"{lbl:42}{b:>10}{g:>10}")

    # --- Extra analysis ---
    rag_df = out[out["mode"] == "rag"]
    hit_true = rag_df[(rag_df["hit"] == True) & (rag_df["type"] == "in_scope")]
    hit_false = rag_df[(rag_df["hit"] == False) & (rag_df["type"] == "in_scope")]
    false_refusals = rag_df[(rag_df["type"] == "in_scope") & (rag_df["refused"] == True)]

    print("\n--- Extra analysis (RAG only) ---")
    print(f"{'False refusal rate (in-scope)':42}{pct(false_refusals['refused']):>10}")
    if len(hit_true):
        print(f"{'Correctness when retrieval hit=True':42}{pct(hit_true['correct']):>10}")
    if len(hit_false):
        print(f"{'Correctness when retrieval hit=False':42}{pct(hit_false['correct']):>10}")


if __name__ == "__main__":
    main()
