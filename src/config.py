import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
CHROMA_DIR = str(BASE_DIR / "chroma_db")
EVAL_DIR = BASE_DIR / "evaluation"

COLLECTION_NAME = "rbi_policy"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"

