"""
test_recommend.py
Check recommendation blending
"""

from services import recommend

def test_get_recommendations(vector_search, sample_user_vec):
    results = recommend.get_recommendations(sample_user_vec, vector_search, alpha=1.0, beta=0.0)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "id" in results[0]
    assert "score" in results[0]
    assert results[0]["score"] <= 1.0
