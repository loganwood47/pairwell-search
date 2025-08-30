"""
test_embeddings.py
Check embeddings creation
"""

import numpy as np
from services.embedding_service import embed_texts, embed_user_profile

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
