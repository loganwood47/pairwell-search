import pytest
from unittest.mock import patch
from src.pairwell_search.services import interest_expansion

def test_expand_interest_returns_string():
    interests = ["animals"]
    
    # patch the API call so it doesn't hit the real OpenRouter
    with patch("src.pairwell_search.services.interest_expansion.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Mission: Save animals."}}]
        }
        result = interest_expansion.expand_interest(interests)
        assert isinstance(result, str)
        assert "Mission" in result
