from pathlib import Path

BASE = Path(__file__).parent
DOCS_DIR = BASE / "data" / "docs"
STORE_DIR = BASE / "store"
INDEX_PATH = STORE_DIR / "index.faiss"
CHUNKS_PATH = STORE_DIR / "chunks.json"

# Embeddings (small, runs fine on CPU or GPU)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# LLM served by Ollama. Fits a 4 GB RTX 3050. Alternatives: "llama3.2:3b", "phi3:mini"
LLM_MODEL = "qwen2.5:3b"
NUM_CTX = 4096

# Chunking (in words). Change these and re-run ingest.py to test variations.
CHUNK_WORDS = 200
OVERLAP_WORDS = 40

# Retrieval
TOP_K = 4
# If the best chunk scores below this (cosine similarity), refuse without calling the LLM.
# Tune this on your own documents using eval/questions.csv.
MIN_SCORE = 0.45

# ---------------------------------------------------------------- Web crawler (crawler.py)
# Identify yourselves politely; put a real contact email of one team member.
CRAWL_CONTACT = "dhruvjindal05@gmail.com"
CRAWL_HOSTS = {"svnit.ac.in"}          # www. and trailing dots are ignored; subdomains (mis., help.) are NOT crawled
CRAWL_DELAY = 1.0                      # seconds between requests
CRAWL_MAX_DEPTH = 2                    # 0 = only the seeds; PDFs found at the last level are still downloaded
CRAWL_MAX_PAGES = 300
CRAWL_MAX_FILES = 150
CRAWL_MAX_FILE_MB = 25

# Start points (all taken from the SVNIT site menu).
CRAWL_SEEDS = [
    "https://www.svnit.ac.in/web/rules_regulations.php",
    "https://www.svnit.ac.in/web/academic_section.php",
    "https://www.svnit.ac.in/web/fee-structure-inst.php",
    "https://www.svnit.ac.in/web/scholarship.php",
    "https://www.svnit.ac.in/web/forms.php",
    "https://www.svnit.ac.in/web/admission.php",
    "https://www.svnit.ac.in/web/about-us.php",
    "https://www.svnit.ac.in/web/brief-about-director.php",
    "https://www.svnit.ac.in/web/registrar.php",
    "https://www.svnit.ac.in/web/deans.php",
    "https://www.svnit.ac.in/web/hod.php",
    "https://www.svnit.ac.in/web/department/ai/",
    "https://www.svnit.ac.in/web/student_life/",
    "https://www.svnit.ac.in/web/cdc/about.php",
    "https://www.svnit.ac.in/web/library/library.php",
    "https://www.svnit.ac.in/web/canteen.php",
    "https://www.svnit.ac.in/web/healthcare.php",
    "https://www.svnit.ac.in/web/dcc.php",
    "https://www.svnit.ac.in/web/guesthouse.php",
    "https://www.svnit.ac.in/web/classroom-complex.php",
    "https://www.svnit.ac.in/web/physical_education_centre/index.php",
    "https://www.svnit.ac.in/web/ccc.php",
    "https://www.svnit.ac.in/web/nptel.php",
    "https://www.svnit.ac.in/web/campus.php",
    "https://www.svnit.ac.in/Data/Notice/2026/August/Revised Academic Cal_2026-27.pdf",
    "https://www.svnit.ac.in/Data/Notice/2026/September/holidays.pdf",
    "https://svnit.ac.in/Data/Notice/2026/August/Hostel Fees Structure of A.Y. 2026-27- Director Sign.pdf",
    "https://www.svnit.ac.in/Data/Notice/2018/policy_prevention_plagirism.pdf",
    "https://www.svnit.ac.in/qlinks/OFFICE_ORDER_TIMING2019.pdf",
    "https://www.svnit.ac.in/Data/Notice/2023/June/Regulation-2023-24-SVNIT-Annexure-62.1.pdf",
]

# Anything whose URL or link text matches is skipped. Mostly candidate lists (personal data),
# exam seating (roll numbers), tenders and recruitment. Edit freely.
CRAWL_DENY_PATTERNS = [
    r"eligible", r"selected[ _-]?candidate", r"selected[ _-]?phd", r"shortlist", r"wait[ _-]?list",
    r"scrutiny", r"seating", r"merit[ _-]?list", r"list[ _-]of[ _-](selected|provisionally|candidates|students)",
    r"result", r"tender", r"quotation", r"recruitment", r"walk[ _-]?in", r"corrigendum",
    r"login", r"logout", r"moodle", r"upload-\d{8,}", r"DOC\d{6}",
    r"/admission/\d{4}/(phd|mtech|mba)/",     # M.Tech/Ph.D/MBA admission notices (not B.Tech focused)
    r"academicregulations2013",               # old 2013-14 regulations conflict with 2023-24
]
CRAWL_DENY_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".xls", ".xlsx", ".xlsm", ".csv",
                  ".zip", ".rar", ".7z", ".mp4", ".mp3", ".avi", ".doc", ".docx", ".ppt", ".pptx"}
