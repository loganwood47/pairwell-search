"""
test_recommend.py
Check recommendation blending
"""
import pytest
from services import recommend


def test_get_recommendations(vector_search, dummy_user_vector):
    results = recommend.get_recommendations(dummy_user_vector, vector_search)
    assert isinstance(results, list)
    for r in results:
        assert "id" in r and "score" in r

