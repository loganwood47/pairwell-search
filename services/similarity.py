"""
similarity.py
Handles FAISS vector search
"""
import faiss
import numpy as np
import torch
import math
import pickle

from sentence_transformers import SentenceTransformer
from models.two_tower import TwoTower

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

def load_model_and_preprocessing(model_path, preprocessing_path, model_class, **model_kwargs):
    """
    Load a trained model and its matching preprocessing metadata.
    """
    # Load preprocessing metadata
    with open(preprocessing_path, "rb") as f:
        preprocessing = pickle.load(f)

    vocabs = preprocessing["vocabs"]
    text_emb_dim = preprocessing["text_embed_dim"]
    embed_dim = preprocessing["embed_dim"]
    cat_emb_dim = preprocessing["cat_embed_dim"]
    
    model = TwoTower(
        n_users=len(preprocessing["user_index"]) + 1,
        n_nonprofits=len(preprocessing["nonprofit_index"]) + 1,
        city_vocab_size=len(vocabs["city"]),
        state_vocab_size=len(vocabs["state"]),
        interest_vocab_size=len(vocabs["interest"]),
        population_vocab_size=len(vocabs["population"]),
        text_emb_dim=text_emb_dim,
        embed_dim=embed_dim,
        cat_emb_dim=cat_emb_dim,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, preprocessing


def get_user_embedding(model, preprocessing, raw_user):
    vocabs = preprocessing["vocabs"]

    # numeric features (normalize with training stats)
    inc = (raw_user.get("income", 0.0) - preprocessing["inc_mean"]) / preprocessing["inc_std"]
    don = (raw_user.get("donation_budget", 0.0) - preprocessing["don_mean"]) / preprocessing["don_std"]
    feats = torch.tensor([[inc, don]], dtype=torch.float32)

    # categorical lookups (default to 0 if unseen)
    city_id = vocabs["city"].get(raw_user.get("city"), 0)
    state_id = vocabs["state"].get(raw_user.get("state"), 0)
    interest_id = vocabs["interest"].get(raw_user.get("interests", [None])[0], 0)
    pref_id = vocabs["pref"].get(raw_user.get("engagement_prefs", [None])[0], 0)

    city_id = torch.tensor([city_id])
    state_id = torch.tensor([state_id])
    interest_id = torch.tensor([interest_id])
    pref_id = torch.tensor([pref_id])

    with torch.no_grad():
        # start with numeric features
        u_vec = model.user_side(feats)

        # add categorical embeddings if they exist
        if hasattr(model, "city_emb"):
            u_vec = u_vec + model.city_emb(city_id)
        if hasattr(model, "state_emb"):
            u_vec = u_vec + model.state_emb(state_id)
        if hasattr(model, "interest_emb"):
            u_vec = u_vec + model.interest_emb(interest_id)
        if hasattr(model, "pref_emb"):
            u_vec = u_vec + model.pref_emb(pref_id)

        # pass through user MLP
        if hasattr(model, "user_mlp"):
            u_vec = model.user_mlp(u_vec)

        # normalize
        u_vec = torch.nn.functional.normalize(u_vec, dim=-1)

    return u_vec.squeeze(0).numpy()


       
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers

    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Calculate differences
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance