"""
test_recommend.py
Check recommendation blending
"""
import pytest
from src.pairwell_search.services import recommend
from unittest.mock import patch

@patch('src.pairwell_search.services.recommend.twoTowerRec')
def test_twoTowerRec(mock_twoTowerRec, dummy_user_vector, dummy_mission_vector):
    mock_twoTowerRec.return_value = [
        {"id": 1, "name": "Test 1", "total_similarity": 0.9, "model_similarity": 0.8, "mission_similarity": 0.7, "geo_distance_meters": 100},
        {"id": 2, "name": "Test 2", "total_similarity": 0.85, "model_similarity": 0.75, "mission_similarity": 0.65, "geo_distance_meters": 200}
    ]
    
    user_mission_embedding = dummy_mission_vector
    user_lat = 34.03956
    user_lon = -118.48194
    alpha = 0.5
    beta = 0.5
    gamma = 0.0

    results = recommend.twoTowerRec(dummy_user_vector, user_mission_embedding, user_lat, user_lon, alpha, beta, gamma)
    assert isinstance(results, list)
    required_keys = {"id", "name", "total_similarity", "model_similarity", "mission_similarity", "geo_distance_meters"}
    for r in results:
        assert required_keys.issubset(r.keys())

@patch('src.pairwell_search.services.recommend.twoTowerRec')
def test_twoTowerRec_weights(mock_twoTowerRec, dummy_user_vector, dummy_mission_vector):
    mock_twoTowerRec.side_effect = [
        [{"id": 1, "name": "Test 1", "total_similarity": 0.9, "model_similarity": 0.8, "mission_similarity": 0.7, "geo_distance_meters": 100}],
        [{"id": 2, "name": "Test 2", "total_similarity": 0.85, "model_similarity": 0.75, "mission_similarity": 0.65, "geo_distance_meters": 200}]
    ]
    
    user_mission_embedding = dummy_mission_vector 
    user_lat = 0.0
    user_lon = 0.0

    results_default = recommend.twoTowerRec(dummy_user_vector, user_mission_embedding, user_lat, user_lon)
    results_custom = recommend.twoTowerRec(dummy_user_vector, user_mission_embedding, user_lat, user_lon, alpha=0.7, beta=0.3, gamma=0.0)

    assert results_default != results_custom  # Ensure different weights yield different results
