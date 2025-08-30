"""
test_similarity.py
Check FAISS similarity search
"""

def test_vector_search(vector_search, sample_user_vec):
    results = vector_search.search(sample_user_vec, k=2)
    assert isinstance(results, list)
    assert len(results) == 2
    assert "id" in results[0]
    assert "score" in results[0]
