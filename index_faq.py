"""Reads data/cashly_faq.txt and stores each block in the Chroma vector database.

Run it once, and again every time the FAQ file changes:
    python index_faq.py
"""
import chromadb

from config import COLLECTION_NAME, FAQ_PATH, VECTOR_DB_PATH


def load_blocks(path):
    """Each block is a question + answer, separated from the next by a blank line."""
    text = path.read_text(encoding="utf-8")
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def main():
    blocks = load_blocks(FAQ_PATH)

    chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))

    # Recreate the collection so blocks removed from the FAQ don't stay in the database
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    collection.add(
        ids=[f"faq_{i}" for i in range(len(blocks))],
        documents=blocks,
    )
    print(f"{len(blocks)} FAQ blocks indexed in '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
