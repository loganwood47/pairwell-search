"""
test_embeddings.py
Check embeddings creation
"""

import numpy as np
from src.pairwell_search.services.embedding_service import embed_texts, embed_user_profile

def test_embed_texts():
    texts = ["Hello world", "Climate change nonprofit"]
    vecs = embed_texts(texts)
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape[0] == len(texts)

def test_embed_user_profile():
    profile = {"geography": "California", "income": 100000, "interests": ["education", "climate"]}
    vec = embed_user_profile(profile)
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] > 0

def test_embed_texts_returns_correct_shape():
    texts = ["Save animals worldwide.", "Provide education to children."]
    vectors = embed_texts(texts)
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape[0] == len(texts)
