import chromadb

from src.config import CHROMA_DIR, COLLECTION_NAME


class VectorStoreManager:
    def __init__(self, persist_dir=CHROMA_DIR, collection_name=COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks, embeddings, batch_size=500):
        texts = [c.page_content for c in chunks]
        metadatas = [
            {
                "source_file": c.metadata.get("source_file", "unknown"),
                "page": int(c.metadata.get("page", 0)),
            }
            for c in chunks
        ]
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i + batch_size],
                documents=texts[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size].tolist(),
                metadatas=metadatas[i:i + batch_size],
            )
        print(f"docs in collection -> {self.collection.count()}")

    def search(self, query_embedding, top_k=3):
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

    def count(self):
        return self.collection.count()