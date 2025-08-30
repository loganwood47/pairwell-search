"""
conftest.py
Shared pytest fixtures for tests
"""

import pytest
import numpy as np
from services import embedding_service, similarity
from services.similarity import VectorSearch
from services.embedding_service import embed_texts

import sys
import os

# TODO: replace with better editable installation, see pyproject.toml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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
    vec_dim = 8
    return np.random.rand(vec_dim).astype(np.float32)
