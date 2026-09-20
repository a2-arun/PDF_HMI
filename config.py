"""Central configuration, loaded from environment variables / .env.

Nothing here hits disk or the network — it's safe to import from anywhere.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# --- Knowledge base sources -------------------------------------------------
GRADE_PDFS_DIR = Path(os.getenv("GRADE_PDFS_DIR", BASE_DIR / "grade_pdfs"))

# Maps an internal collection name (used as the ChromaDB collection name)
# to the textbook PDF it is built from.
GRADE_PDFS = {
    "Grade_6": GRADE_PDFS_DIR / "Grade6.pdf",
    "Grade_7": GRADE_PDFS_DIR / "Grade7.pdf",
    "Grade_8": GRADE_PDFS_DIR / "Grade8.pdf",
    "Grade_9": GRADE_PDFS_DIR / "Grade9.pdf",
}

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))

# --- Text processing ---------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "40"))

# Repeated headers/footers/watermarks (e.g. "Reprint 2024-25") are dropped
# once a line shows up on at least this fraction of pages.
BOILERPLATE_LINE_RATIO = float(os.getenv("BOILERPLATE_LINE_RATIO", "0.35"))
BOILERPLATE_MIN_PAGES = int(os.getenv("BOILERPLATE_MIN_PAGES", "10"))

# --- Retrieval ---------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")
TOP_K = int(os.getenv("TOP_K", "4"))

# Cosine distance (0 = identical, 2 = opposite). Chunks farther than this from
# the question are treated as "not covered in this textbook" rather than
# handed to the LLM, which cuts down on off-syllabus / hallucinated answers.
RELEVANCE_DISTANCE_THRESHOLD = float(os.getenv("RELEVANCE_DISTANCE_THRESHOLD", "0.9"))

# --- Language model (served locally via Ollama: https://ollama.com) --------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
