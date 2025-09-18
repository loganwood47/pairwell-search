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
from src.pairwell_search.models.two_tower import TwoTower

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

def load_model_and_preprocessing(model_path, preprocessing_path, **model_kwargs):
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



def preprocess_single_user(raw_user, preprocessing):
    """
    Convert raw user dict into a batch dict ready for model.forward_user().
    """
    vocabs = preprocessing["vocabs"]
    inc_mean, inc_std = preprocessing["inc_mean"], preprocessing["inc_std"]
    don_mean, don_std = preprocessing["don_mean"], preprocessing["don_std"]

    # numeric
    inc_val = (raw_user.get("income", 0.0) - inc_mean) / inc_std
    don_val = (raw_user.get("donation_budget", 0.0) - don_mean) / don_std
    user_num = torch.tensor([[inc_val, don_val]], dtype=torch.float)

    # categorical (lookup in vocabs with fallback to 0)
    def lookup(vocab, key):
        return vocab.get(raw_user.get(key, "").lower(), 0)

    city_id = lookup(vocabs["city"], "city")
    state_id = lookup(vocabs["state"], "state")

    def lookup_list(vocab, items):
        return [vocab.get(x.lower(), 0) for x in items]

    interest_ids = lookup_list(vocabs["interest"], raw_user.get("interests", []))
    pref_ids = lookup_list(vocabs["interest"], raw_user.get("engagement_prefs", []))

    # pad to fixed length (match training pad length, e.g. 5)
    pad_len = 5
    def pad_list(lst):
        return lst[:pad_len] + [0] * max(0, pad_len - len(lst))

    interest_ids = torch.tensor([pad_list(interest_ids)], dtype=torch.long)
    pref_ids = torch.tensor([pad_list(pref_ids)], dtype=torch.long)

    batch = {
        "user_idx": torch.tensor([0], dtype=torch.long),  # dummy id
        "user_num": user_num,
        "user_city": torch.tensor([city_id], dtype=torch.long),
        "user_state": torch.tensor([state_id], dtype=torch.long),
        "user_interests": interest_ids,
        "user_prefs": pref_ids,
    }
    return batch


def get_user_embedding(model, preprocessing, raw_user):
    """
    Take raw user dict, convert to batch, and run through model.forward_user.
    """
    model.eval()
    batch = preprocess_single_user(raw_user, preprocessing)
    with torch.no_grad():
        emb = model.forward_user(batch)
    return emb.squeeze(0).numpy()


       
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