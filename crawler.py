"""Polite, scoped crawler for the SVNIT website.

Saves readable text of HTML pages and downloads linked PDFs into data/docs/web/, plus
manifest.json (file -> source URL and title) and crawl_log.csv (what was saved or skipped, and why).
Then run:  python ingest.py

Usage:
    python crawler.py --max-pages 30          # small first run to sanity-check
    python crawler.py                          # full run with settings from config.py
    python crawler.py --refresh                # re-download PDFs that already exist
"""
import argparse, csv, hashlib, json, re, sys, time
from collections import Counter, deque
from datetime import date
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, NavigableString

import config

OUT = config.DOCS_DIR / "web"
MANIFEST = OUT / "manifest.json"
LOG = OUT / "crawl_log.csv"
UA = f"SVNIT-CampusAssistant-StudentProject (course project; contact {config.CRAWL_CONTACT})"
MAX_BOILER_LINE = 150   # menus/footers are short lines; long repeated paragraphs are real content
NAV_RE = re.compile(r"(^|[-_ ])(nav|navbar|menu|footer|sidebar|breadcrumb|carousel|slider|social)([-_ ]|$)", re.I)


# ----------------------------------------------------------------------------- URLs
def norm_url(url, base=None):
    """Canonical form so www/non-www/trailing-dot/space variants are the same page."""
    try:
        url = urljoin(base, url.strip()) if base else url.strip()
        p = urlsplit(url)
    except ValueError:
        return None
    if p.scheme not in ("http", "https") or not p.netloc:
        return None
    host = p.netloc.lower().rstrip(".")
    bare = host[4:] if host.startswith("www.") else host
    scheme, netloc = p.scheme, host
    if bare == "svnit.ac.in":
        scheme, netloc = "https", "www.svnit.ac.in"
    path = quote(unquote(p.path), safe="/()'!*-._~:@&=+$,;")
    return urlunsplit((scheme, netloc, path, p.query, ""))


def host_key(url):
    h = urlsplit(url).netloc.lower().rstrip(".")
    return h[4:] if h.startswith("www.") else h


DENY_RE = [re.compile(p, re.I) for p in config.CRAWL_DENY_PATTERNS]


def skip_reason(url, anchor=""):
    if host_key(url) not in config.CRAWL_HOSTS:
        return "outside allowed hosts"
    ext = Path(unquote(urlsplit(url).path)).suffix.lower()
    if ext in config.CRAWL_DENY_EXT:
        return f"extension {ext}"
    hay = unquote(url) + " " + (anchor or "")
    for rx in DENY_RE:
        m = rx.search(hay)
        if m:
            return f"deny pattern '{m.group(0)}'"
    return None


def is_pdf_url(url):
    return urlsplit(url).path.lower().endswith(".pdf")


def slug(url, ext):
    p = urlsplit(url)
    name = Path(unquote(p.path)).stem or "index"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:70] or "page"
    h = hashlib.sha1(url.encode()).hexdigest()[:8]
    return f"{name}_{h}{ext}"


# ----------------------------------------------------------------------------- fetching
class Fetcher:
    def __init__(self, delay, site):
        self.site = site
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.delay, self.last = delay, 0.0
        self.robots = RobotFileParser()
        self._load_robots()

    def _load_robots(self):
        url = self.site + "/robots.txt"
        try:
            r = self.s.get(url, timeout=20)
            if r.status_code == 200:
                self.robots.parse(r.text.splitlines())
                print("robots.txt loaded")
            else:
                self.robots.parse([])
                print(f"robots.txt returned {r.status_code}; treating everything as allowed")
        except requests.RequestException as e:
            self.robots.parse([])
            print(f"could not read robots.txt ({e}); continuing carefully")

    def allowed(self, url):
        return self.robots.can_fetch(UA, url)

    def get(self, url, stream=False):
        wait = self.delay - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        return self.s.get(url, timeout=40, stream=stream, allow_redirects=True)


# ----------------------------------------------------------------------------- HTML -> text
def extract_html(html, anchor=""):
    soup = BeautifulSoup(html, "html.parser")
    links = [(a["href"], a.get_text(" ", strip=True)) for a in soup.find_all("a", href=True)]
    heading = ""
    for tag in ("h1", "h2"):
        el = soup.find(tag)
        if el and el.get_text(strip=True):
            heading = el.get_text(" ", strip=True)
            break
    title = heading or anchor or (soup.title.get_text(strip=True) if soup.title else "") or "SVNIT page"

    if soup.head:
        soup.head.decompose()
    for t in soup(["script", "style", "noscript", "nav", "header", "footer", "form", "iframe", "svg"]):
        t.decompose()
    doomed = []
    for t in soup.find_all(True):
        cls = " ".join(t.get("class") or []) if isinstance(t.get("class"), list) else str(t.get("class") or "")
        if NAV_RE.search(cls + " " + str(t.get("id") or "")):
            doomed.append(t)
    for t in doomed:
        try:
            t.decompose()
        except Exception:
            pass
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if any(cells):
                rows.append(" | ".join(cells))
        try:
            table.replace_with(NavigableString("\n" + "\n".join(rows) + "\n"))
        except Exception:
            pass
    lines = [re.sub(r"\s+", " ", l).strip() for l in soup.get_text("\n").splitlines()]
    return title, "\n".join(l for l in lines if l), links


# ----------------------------------------------------------------------------- site-wide boilerplate removal
def clean_site_boilerplate(min_pages=8, frac=0.3):
    """Remove lines (menus, footers) that repeat on >= frac of all saved pages."""
    pages_dir = OUT / "pages"
    files = sorted(pages_dir.glob("*.md")) if pages_dir.exists() else []
    if len(files) < min_pages:
        return 0
    key = lambda s: re.sub(r"\d+", "#", s.strip())
    texts = {f: f.read_text(encoding="utf-8").splitlines() for f in files}
    counts = Counter()
    for lines in texts.values():
        counts.update({key(l) for l in lines if 3 < len(l.strip()) <= MAX_BOILER_LINE})
    limit = max(3, int(frac * len(files)))
    boiler = {k for k, c in counts.items() if c >= limit}
    removed = 0
    for f, lines in texts.items():
        kept = [l for l in lines if not (len(l.strip()) <= MAX_BOILER_LINE and key(l) in boiler)]
        removed += len(lines) - len(kept)
        f.write_text("\n".join(kept), encoding="utf-8")
    print(f"site-wide boilerplate: {len(boiler)} repeating lines removed ({removed} occurrences)")
    return len(boiler)


# ----------------------------------------------------------------------------- main crawl
def crawl(args):
    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    first = urlsplit(norm_url(config.CRAWL_SEEDS[0]))
    fetcher = Fetcher(args.delay, f"{first.scheme}://{first.netloc}")

    queue, seen = deque(), set()
    for u in config.CRAWL_SEEDS:
        n = norm_url(u)
        if n and n not in seen:
            seen.add(n)
            queue.append((n, 0, ""))

    log, pages, files, text_hashes = [], 0, 0, set()
    def record(url, depth, action, detail=""):
        log.append({"url": url, "depth": depth, "action": action, "detail": detail})

    while queue:
        url, depth, anchor = queue.popleft()
        reason = skip_reason(url, anchor)
        if reason:
            record(url, depth, "skipped", reason)
            continue
        if not fetcher.allowed(url):
            record(url, depth, "skipped", "disallowed by robots.txt")
            continue

        try:
            if is_pdf_url(url):
                if files >= args.max_files:
                    record(url, depth, "skipped", "file cap reached")
                    continue
                name = slug(url, ".pdf")
                path = OUT / name
                if path.exists() and not args.refresh:
                    files += 1
                    manifest.setdefault(name, {"url": url, "title": anchor or Path(unquote(urlsplit(url).path)).stem,
                                               "kind": "pdf", "crawled": str(date.today())})
                    record(url, depth, "cached", name)
                    continue
                r = fetcher.get(url, stream=True)
                if r.status_code != 200:
                    record(url, depth, "error", f"HTTP {r.status_code}")
                    continue
                limit = args.max_file_mb * 1024 * 1024
                data, size = [], 0
                for chunk in r.iter_content(65536):
                    size += len(chunk)
                    if size > limit:
                        break
                    data.append(chunk)
                if size > limit:
                    record(url, depth, "skipped", f"larger than {args.max_file_mb} MB")
                    continue
                blob = b"".join(data)
                if not blob.startswith(b"%PDF"):
                    record(url, depth, "error", "not a PDF")
                    continue
                path.write_bytes(blob)
                try:
                    from pypdf import PdfReader
                    sample = "".join((p.extract_text() or "") for p in PdfReader(str(path)).pages[:3])
                except Exception as e:
                    path.unlink(missing_ok=True)
                    record(url, depth, "error", f"unreadable PDF ({e})")
                    continue
                if len(sample.strip()) < 100:
                    path.unlink(missing_ok=True)
                    record(url, depth, "skipped", "no text layer (scanned PDF)")
                    continue
                files += 1
                manifest[name] = {"url": url, "title": anchor or Path(unquote(urlsplit(url).path)).stem,
                                  "kind": "pdf", "crawled": str(date.today())}
                record(url, depth, "saved", name)
                print(f"[pdf ] {name}")
                continue

            if pages >= args.max_pages:
                record(url, depth, "skipped", "page cap reached")
                continue
            r = fetcher.get(url)
            ctype = r.headers.get("Content-Type", "")
            if r.status_code != 200:
                record(url, depth, "error", f"HTTP {r.status_code}")
                continue
            if "text/html" not in ctype:
                record(url, depth, "skipped", f"content-type {ctype.split(';')[0]}")
                continue
            r.encoding = r.apparent_encoding if r.encoding in (None, "ISO-8859-1") else r.encoding
            title, text, links = extract_html(r.text, anchor)
            h = hashlib.md5(re.sub(r"\W+", "", text.lower()).encode()).hexdigest()
            if len(text) < 200:
                record(url, depth, "skipped", "too little text")
            elif h in text_hashes:
                record(url, depth, "skipped", "duplicate content")
            else:
                text_hashes.add(h)
                name = slug(url, ".md")
                (OUT / "pages" / name).write_text(f"# {title}\n{text}", encoding="utf-8")
                manifest[f"pages/{name}"] = {"url": url, "title": title, "kind": "page", "crawled": str(date.today())}
                pages += 1
                record(url, depth, "saved", name)
                print(f"[page] {title[:60]}  ({url})")
            for href, a_text in links:
                n = norm_url(href, base=url)
                if not n or n in seen:
                    continue
                if depth < args.depth or is_pdf_url(n):
                    seen.add(n)
                    queue.append((n, depth + 1, a_text))
        except requests.RequestException as e:
            record(url, depth, "error", str(e)[:120])

    removed = clean_site_boilerplate()
    # drop pages that became empty after boilerplate removal
    for key in [k for k, v in manifest.items() if v["kind"] == "page"]:
        f = OUT / key
        if not f.exists() or len(f.read_text(encoding="utf-8").strip()) < 100:
            print(f"[drop] {manifest[key]['url']} (almost empty after boilerplate removal)")
            f.unlink(missing_ok=True)
            manifest.pop(key, None)

    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(LOG, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["url", "depth", "action", "detail"])
        w.writeheader()
        w.writerows(log)
    c = Counter(r["action"] for r in log)
    print(f"\nDone. pages saved: {sum(1 for v in manifest.values() if v['kind']=='page')}, "
          f"PDFs: {sum(1 for v in manifest.values() if v['kind']=='pdf')}; "
          f"log: {dict(c)}\nSee {LOG} for every skipped URL and the reason.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=config.CRAWL_MAX_DEPTH)
    ap.add_argument("--max-pages", type=int, default=config.CRAWL_MAX_PAGES)
    ap.add_argument("--max-files", type=int, default=config.CRAWL_MAX_FILES)
    ap.add_argument("--max-file-mb", type=int, default=config.CRAWL_MAX_FILE_MB)
    ap.add_argument("--delay", type=float, default=config.CRAWL_DELAY)
    ap.add_argument("--refresh", action="store_true", help="re-download PDFs that already exist")
    args = ap.parse_args()
    if config.CRAWL_CONTACT.startswith("team-email@"):
        print("Tip: set CRAWL_CONTACT in config.py to a real email before the full crawl.")
    crawl(args)


if __name__ == "__main__":
    main()
