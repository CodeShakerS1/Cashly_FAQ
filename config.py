from pathlib import Path

BASE_DIR = Path(__file__).parent

FAQ_PATH = BASE_DIR / "data" / "cashly_faq.txt"
VECTOR_DB_PATH = BASE_DIR / "data" / "vector_db"
COLLECTION_NAME = "cashly_collection"

# How many FAQ blocks the retriever brings back for each question
TOP_K = 3
