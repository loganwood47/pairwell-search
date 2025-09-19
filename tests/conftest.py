"""
conftest.py
Shared pytest fixtures for tests
"""

import pytest
import numpy as np
import torch
from src.pairwell_search.services import embedding_service, similarity
from src.pairwell_search.services.similarity import VectorSearch
from src.pairwell_search.services.embedding_service import embed_texts

import sys
import os

# TODO: replace with better editable installation, see pyproject.toml
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# tests/conftest.py

# ---------------------------
# Sample data for tests
# ---------------------------

SAMPLE_NONPROFITS = [
    {"id": 1, "name": "Animal Rescue Org", "mission": "Save animals worldwide."},
    {"id": 2, "name": "Education for All", "mission": "Provide education to underprivileged children."},
    {"id": 3, "name": "Clean Water Initiative", "mission": "Provide clean water to communities in need."},
]

SAMPLE_USER_INTERESTS = ["animals", "education"]

# ---------------------------
# Fixtures
# ---------------------------

@pytest.fixture
def mock_nonprofits():
    return SAMPLE_NONPROFITS

@pytest.fixture
def mock_embeddings():
    vec_dim = 8
    vectors = np.random.rand(len(SAMPLE_NONPROFITS), vec_dim).astype(np.float32)
    return vectors

@pytest.fixture
def vector_search(mock_embeddings):
    vs = similarity.VectorSearch(dim=mock_embeddings.shape[1])
    vs.add_vectors(mock_embeddings, [n["id"] for n in SAMPLE_NONPROFITS])
    return vs

@pytest.fixture
def dummy_user_vector():
    vec_dim = 64
    return np.random.rand(vec_dim).astype(np.float32)

@pytest.fixture
def dummy_mission_vector():
    vec_dim = 384
    return np.random.rand(vec_dim).astype(np.float32)

@pytest.fixture
def dummy_user():
    return {
        "id": 123,
        "name": "Test User",
        "city": "Test City",
        "state": "TS",
        "income": 75000,
        "interests": ["education", "climate", "health"]
    }

@pytest.fixture
def dummy_model():
    class DummyModel:
        def encode(self, texts):
            vec_dim = 64
            if isinstance(texts, list):
                return np.random.rand(len(texts), vec_dim).astype(np.float32)
            else:
                return np.random.rand(vec_dim).astype(np.float32)
        def eval(self):
            pass
        def forward_user(self, *args, **kwargs):
            vec_dim = 64
            # return tf.tensor([np.random.rand(vec_dim).astype(np.float32)])
            return torch.tensor(np.array(np.random.rand(vec_dim).astype(np.float32)))
    return DummyModel()

@pytest.fixture
def dummy_preprocessing():
    return {
        "num_features": ["income"],
        "cat_features": ["user_city", "user_state"],
        "interest_features": ["interests"],
        "embed_dim": 64,
        "vocabs": {
            "city": {"Test City": 1},
            "state": {"TS": 1},
            "interest": {"education": 1, "climate": 2, "health": 3}
        },
        "inc_mean": 50000,
        "inc_std": 20000,
        "don_mean": 1000,
        "don_std": 500,
    }