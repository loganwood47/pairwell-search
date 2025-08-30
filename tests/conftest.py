"""
conftest.py
Shared pytest fixtures for tests
"""

import pytest
import numpy as np
from services import embedding_service, similarity
from services.similarity import VectorSearch
from services.embedding_service import embed_texts

@pytest.fixture
def sample_texts():
    return [
        "Nonprofit providing STEM education to underserved communities",
        "Organization focused on wildlife conservation",
        "Local food bank supporting families in Los Angeles"
    ]

@pytest.fixture
def sample_embeddings(sample_texts):
    return embed_texts(sample_texts)

@pytest.fixture
def vector_search(sample_embeddings):
    vs = similarity.VectorSearch(dim=sample_embeddings.shape[1])
    vs.add_vectors(sample_embeddings, ids=[1, 2, 3])
    return vs

@pytest.fixture
def sample_user_vec():
    # pretend user is interested in education
    return embed_texts(["User interested in education and youth programs"])[0]
