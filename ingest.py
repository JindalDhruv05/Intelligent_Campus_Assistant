"""Read documents, chunk them, embed them, and build a FAISS index."""
import argparse, hashlib, json, re
from collections import Counter
import config


def read_pages(path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        for i, page in enumerate(PdfReader(str(path)).pages, start=1):
            yield i, page.extract_text() or ""
    elif ext in (".txt", ".md"):
        yield 1, path.read_text(encoding="utf-8", errors="ignore")


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def strip_repeated_lines(pages, frac=0.5, min_pages=6):
    """Drop header/footer lines that repeat on at least `frac` of a document's pages.
    Digits are ignored when comparing, so 'Page 5 of 62' and 'Page 6 of 62' match."""
    if len(pages) < min_pages:
        return pages
    key = lambda s: re.sub(r"\d+", "#", s.strip())
    counts = Counter()
    for _, txt in pages:
        counts.update({key(l) for l in txt.splitlines() if len(l.strip()) > 10})
    limit = max(3, int(frac * len(pages)))
    boiler = {k for k, c in counts.items() if c >= limit}
    return [(n, "\n".join(l for l in txt.splitlines() if key(l) not in boiler))
            for n, txt in pages]


def chunk_words(text, size, overlap):
    words = text.split()
    step = max(size - overlap, 1)
    for start in range(0, max(len(words), 1), step):
        piece = words[start:start + size]
        if len(piece) >= 10:
            yield " ".join(piece)
        if start + size >= len(words):
            break


def load_manifest():
    mp = config.DOCS_DIR / "web" / "manifest.json"
    return json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}


def build_chunks(size=config.CHUNK_WORDS, overlap=config.OVERLAP_WORDS):
    manifest = load_manifest()
    chunks, seen = [], set()
    for path in sorted(config.DOCS_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".pdf", ".txt", ".md"):
            continue
        parts = path.relative_to(config.DOCS_DIR).parts
        rel = "/".join(parts[1:]) if parts[0] == "web" else path.name
        meta = manifest.get(rel, {})
        title, url, is_page = meta.get("title", ""), meta.get("url", ""), meta.get("kind") == "page"
        pages = list(read_pages(path))
        if sum(len(t.strip()) for _, t in pages) < 100:
            print(f"WARNING: {path.name} has no extractable text (scanned images?). "
                  f"Skipped. Provide a text/markdown version instead.")
            continue
        for page_no, raw in strip_repeated_lines(pages):
            for piece in chunk_words(clean(raw), size, overlap):
                key = hashlib.md5(piece.lower().encode()).hexdigest()
                if key in seen:          # identical text already indexed (e.g. same PDF linked twice)
                    continue
                seen.add(key)
                chunks.append({"id": len(chunks), "source": path.name,
                               "page": None if is_page else page_no,
                               "url": url, "title": title,
                               "text": f"[{title}] {piece}" if title else piece})
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-words", type=int, default=config.CHUNK_WORDS)
    ap.add_argument("--overlap", type=int, default=config.OVERLAP_WORDS)
    args = ap.parse_args()

    chunks = build_chunks(args.chunk_words, args.overlap)
    if not chunks:
        raise SystemExit(f"No usable documents found in {config.DOCS_DIR}")
    print(f"{len(chunks)} chunks from {len({c['source'] for c in chunks})} files")

    import faiss
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBED_MODEL)
    vecs = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                        show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    config.STORE_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(config.INDEX_PATH))
    config.CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print("Index saved to", config.STORE_DIR)


if __name__ == "__main__":
    main()
