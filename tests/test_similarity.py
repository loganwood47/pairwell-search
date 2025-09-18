"""
test_similarity.py
Check FAISS similarity search
"""
import numpy as np 
import pytest
from src.pairwell_search.services.similarity import preprocess_single_user, get_user_embedding


def test_preprocess_single_user(dummy_user, dummy_preprocessing):

    processed = preprocess_single_user(dummy_user, dummy_preprocessing)
    required_keys = {"user_idx", "user_num", "user_city", "user_state", "user_prefs"}
    assert isinstance(processed, dict)

    assert required_keys.issubset(processed.keys())
    
    assert processed["user_num"].shape == (1, 2)  # Check numeric features shape
    assert processed["user_interests"].shape == (1, 5)  # Check interests shape
    assert processed["user_prefs"].shape == (1, 5)  # Check preferences shape

def test_get_user_embedding(dummy_model, dummy_preprocessing, dummy_user):
    embedding = get_user_embedding(dummy_model, dummy_preprocessing, dummy_user)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1  # Check that the output is a 1D array
    assert embedding.shape[0] == dummy_preprocessing["embed_dim"]  # Check embedding dimension