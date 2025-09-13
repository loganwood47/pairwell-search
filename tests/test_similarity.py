"""
test_similarity.py
Check FAISS similarity search
"""

def test_vector_search(vector_search, dummy_user_vector):
    results = vector_search.search(dummy_user_vector, k=2)
    assert isinstance(results, list)
    assert len(results) == 2
    assert "id" in results[0]
    assert "score" in results[0]
