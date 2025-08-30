"""
embeddings.py
Handles text and profile embedding using HuggingFace SentenceTransformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once globally
_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into vectors"""
    embeddings = _model.encode(texts, normalize_embeddings=True)
    return np.array(embeddings).astype("float32")

def embed_user_profile(profile: dict) -> np.ndarray:
    """Turn structured user data into text and embed"""
    profile_text = (
        f"User in {profile.get('geography', '')}, "
        f"income {profile.get('income', '')}, "
        f"mission statement is {profile.get('interests', '')}"
    )
    return embed_texts([profile_text])[0]
