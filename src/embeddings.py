from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


class EmbeddingManager:
    def __init__(self, model_name=EMBEDDING_MODEL):
        print(f"loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts, batch_size=64, show_progress=True):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def embed_query(self, query):
        return self.model.encode([query], convert_to_numpy=True)[0]