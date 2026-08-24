import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loader import load_all_docs, split_docs
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager


def main():
    docs = load_all_docs()
    if not docs:
        print("No documents found in data/raw. Add your RBI files first.")
        return

    chunks = split_docs(docs)

    embedder = EmbeddingManager()
    texts = [c.page_content for c in chunks]
    embeddings = embedder.generate_embeddings(texts)
    print(f"embeddings shape -> {embeddings.shape}")

    store = VectorStoreManager()
    store.reset()
    store.add_documents(chunks, embeddings)

    sources = {c.metadata.get("source_file") for c in chunks}
    print("\n=== INDEX SUMMARY (note these for your README) ===")
    print(f"source documents : {len(sources)}")
    print(f"pages loaded     : {len(docs)}")
    print(f"chunks indexed   : {len(chunks)}")


if __name__ == "__main__":
    main()
    