"""
similarity.py
Handles FAISS vector search
"""
import faiss
import numpy as np

# TODO: Replace this with actual vectors from DB rather than in-memory embeddings

class VectorSearch:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.id_map = []  # mapping back to nonprofit IDs

    def add_vectors(self, vectors: np.ndarray, ids: list[int]):
        """Add vectors with associated IDs"""
        self.index.add(vectors)
        self.id_map.extend(ids)

    def search(self, query_vec: np.ndarray, k: int = 5):
        """Search for top-k most similar vectors"""
        if query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, 0)
        D, I = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append({"id": self.id_map[idx], "score": float(score)})
        return results
