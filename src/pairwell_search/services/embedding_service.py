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

def embed_nonprofit_profiles(profiles: list[dict]) -> np.ndarray:
    """Embed structured nonprofit profiles into vectors"""
    texts = []
    for profile in profiles:
        profile_text = (
            f"Nonprofit located in {profile.get('geography', '')}, "
            f"mission statement: {profile.get('mission_statement', '')}, "
            # TODO: add revenue stuff here
            # TODO: rework embeddings to use full nonprofit profile
            # TODO: break into individual embeddings per component (geo, mission etc)
            # then final score is weighted average of component similarities
        )
        texts.append(profile_text)
    embeddings = _model.encode(texts, normalize_embeddings=True)
    return np.array(embeddings).astype("float32")

def embed_user_profile(profile: dict) -> np.ndarray:
    """Turn structured user data into text and embed"""
    profile_text = (
        f"User located in {profile.get('geography', '')}, "
        # f"income {profile.get('income', '')}, "
        f"mission statement: {profile.get('interests', '')}"
    )
    return embed_texts([profile_text])[0]
