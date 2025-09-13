"""
similarity.py
Handles FAISS vector search
"""
import faiss
import numpy as np
import torch

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# TODO: Replace this with actual vectors from DB rather than in-memory embeddings

class VectorSearch:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.id_map = []  # mapping back to nonprofit IDs

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """L2 normalization of vectors to unit length"""
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def add_vectors(self, vectors: np.ndarray, ids: list[int]):
        """Add vectors with associated IDs"""
        vectors = self.normalize(vectors.astype(np.float32))
        self.index.add(vectors)
        self.id_map.extend(ids)

    def search(self, query_vec: np.ndarray, k: int = 5):
        """Search for top-k most similar vectors"""
        if query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, 0)
        query_vec = self.normalize(query_vec.astype(np.float32))

        D, I = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append({"id": self.id_map[idx], "score": float(score)})
        return results
    
    def search_huggingface(self, query_vec: torch.Tensor, k: int = 5, model=model):
        """Search for top-k most similar vectors"""
        if query_vec.ndim == 1:
            query_vec = query_vec.unsqueeze(0)
        similarities = model.similarity(
            query_vec, 
            self.index.reconstruct_n(0, self.index.ntotal)
            ).squeeze(0)
        if similarities.numel() == 0:
            return []
        k = min(k, similarities.size(0))  # clamp k
        scores, indices = torch.topk(similarities, k)

        results = []
        for idx, score in zip(indices.tolist(), scores.tolist()):
            results.append({"id": self.id_map[idx], "score": float(score)})

        return results