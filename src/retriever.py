from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.config import TOP_K


class RAGRetriever:
    def __init__(self, embedder=None, store=None):
        self.embedder = embedder or EmbeddingManager()
        self.store = store or VectorStoreManager()

    def retrieve(self, query, top_k=TOP_K):
        q_emb = self.embedder.embed_query(query)
        res = self.store.search(q_emb, top_k=top_k)

        hits = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        for text, meta, dist in zip(docs, metas, dists):
            hits.append({
                "text": text,
                "source_file": meta.get("source_file", "unknown"),
                "page": meta.get("page", 0),
                "similarity": round(1 - dist, 4),
            })
        return hits


if __name__ == "__main__":
    r = RAGRetriever()
    for h in r.retrieve("What did RBI say about the repo rate?", top_k=3):
        print(f"\n[{h['source_file']} p{h['page']}] sim={h['similarity']}")
        print(h["text"][:300])

        