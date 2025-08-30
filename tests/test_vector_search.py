def test_vector_search_returns_valid_results(vector_search, dummy_user_vector, mock_nonprofits):
    results = vector_search.search(dummy_user_vector, k=2)
    assert len(results) <= 2
    for r in results:
        assert r["id"] in [n["id"] for n in mock_nonprofits]
        assert isinstance(r["score"], float)
