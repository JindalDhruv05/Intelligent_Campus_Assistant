"""
run_ablations.py  –  Run all post-mid evaluation ablations in one go.

Usage:
    python run_ablations.py              # run all ablations (skips default eval, assumes it finished)
    python run_ablations.py --with-default  # also re-run the default eval first
    python run_ablations.py --limit 15   # use --limit N for a quick smoke-test of every ablation

What it runs (in order, no re-ingest needed unless chunk size changes):
  1.  k=2  ablation
  2.  k=6  ablation
  3.  min-score=0.35 ablation
  4.  min-score=0.55 ablation
  5.  chunk-words=100  (re-ingest → evaluate → restore default index)
  6.  chunk-words=300  (re-ingest → evaluate → restore default index)
  7.  reranker ablation  (requires cross-encoder model download on first run)

All results land in eval/ with descriptive names.
A summary table is printed at the end.
"""

import subprocess
import sys
import shutil
import argparse
import time
from pathlib import Path

BASE   = Path(__file__).parent
PYTHON = str(Path(sys.executable))   # same venv python that's running this script
EVAL   = BASE / "eval"
STORE  = BASE / "store"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], label: str) -> tuple[bool, str]:
    """Run a command, stream output, return (success, stdout)."""
    print(f"\n{'='*60}")
    print(f"▶  {label}")
    print(f"   {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0
    ok = proc.returncode == 0
    status = "✅ done" if ok else "❌ FAILED"
    print(f"\n{status}  ({elapsed:.0f}s)")
    return ok, ""


def save_index():
    """Copy current store/ to store_backup/ so we can restore it later."""
    backup = BASE / "store_backup"
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(STORE, backup)


def restore_index():
    """Restore store/ from store_backup/ and delete the backup."""
    backup = BASE / "store_backup"
    if not backup.exists():
        print("  ⚠ No backup found — skipping restore.")
        return
    if STORE.exists():
        shutil.rmtree(STORE)
    shutil.copytree(backup, STORE)
    shutil.rmtree(backup)
    print("  ✅ Default index restored.")


def ingest(chunk_words: int, overlap: int):
    run([PYTHON, "ingest.py", "--chunk-words", str(chunk_words),
         "--overlap", str(overlap)],
        label=f"ingest  chunk-words={chunk_words}  overlap={overlap}")


def evaluate(out: str, extra_flags: list[str] = [], limit: int = 0, label: str = ""):
    flags = ["--out", f"eval/{out}"] + extra_flags
    if limit:
        flags += ["--limit", str(limit)]
    run([PYTHON, "evaluate.py"] + flags, label=label or f"evaluate → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-default", action="store_true",
                    help="Also re-run the default evaluate.py before ablations.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Pass --limit N to every evaluate call (quick smoke test).")
    ap.add_argument("--skip-chunk", action="store_true",
                    help="Skip the chunk-size ablations (they take longest).")
    ap.add_argument("--skip-rerank", action="store_true",
                    help="Skip the reranker ablation (needs model download).")
    args = ap.parse_args()

    lim = args.limit
    results_made = []

    # ------------------------------------------------------------------
    # 0. Optional: re-run the default evaluation
    # ------------------------------------------------------------------
    if args.with_default:
        evaluate("results.csv", label="default  k=4  min-score=0.45", limit=lim)
        results_made.append("results.csv  (default)")

    # ------------------------------------------------------------------
    # 1–2. Top-k ablations (no re-ingest needed)
    # ------------------------------------------------------------------
    evaluate("results_k2.csv",  ["--k", "2"],  lim, "ablation  k=2")
    results_made.append("results_k2.csv")

    evaluate("results_k6.csv",  ["--k", "6"],  lim, "ablation  k=6")
    results_made.append("results_k6.csv")

    # ------------------------------------------------------------------
    # 3–4. Refusal threshold ablations
    # ------------------------------------------------------------------
    evaluate("results_ms35.csv", ["--min-score", "0.35"], lim, "ablation  min-score=0.35")
    results_made.append("results_ms35.csv")

    evaluate("results_ms55.csv", ["--min-score", "0.55"], lim, "ablation  min-score=0.55")
    results_made.append("results_ms55.csv")

    # ------------------------------------------------------------------
    # 5–6. Chunk-size ablations (requires re-ingest each time)
    # ------------------------------------------------------------------
    if not args.skip_chunk:
        print("\n🔁 Saving current index before chunk-size ablations…")
        save_index()

        ingest(chunk_words=100, overlap=20)
        evaluate("results_c100.csv", limit=lim, label="ablation  chunk-words=100  overlap=20")
        results_made.append("results_c100.csv")

        ingest(chunk_words=300, overlap=60)
        evaluate("results_c300.csv", limit=lim, label="ablation  chunk-words=300  overlap=60")
        results_made.append("results_c300.csv")

        print("\n🔁 Restoring default index (chunk-words=200)…")
        restore_index()
    else:
        print("\n⏭  Skipping chunk-size ablations (--skip-chunk).")

    # ------------------------------------------------------------------
    # 7. Reranker ablation
    # ------------------------------------------------------------------
    if not args.skip_rerank:
        evaluate("results_rerank.csv", ["--rerank"], lim, "ablation  reranker")
        results_made.append("results_rerank.csv")
    else:
        print("\n⏭  Skipping reranker ablation (--skip-rerank).")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ALL ABLATIONS COMPLETE")
    print(f"{'='*60}")
    print("Results saved in eval/:")
    for r in results_made:
        path = EVAL / r.split()[0]
        size = f"{path.stat().st_size // 1024} KB" if path.exists() else "missing!"
        print(f"  {r:<30}  {size}")
    print("\nNext steps:")
    print("  1. Open eval/results.csv and each ablation CSV.")
    print("  2. Hand-check 10–15 rows of results.csv (judge vs your own reading).")
    print("  3. Fill in the numbers table in your slides.")


if __name__ == "__main__":
    main()
