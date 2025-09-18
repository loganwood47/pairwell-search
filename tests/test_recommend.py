"""
test_recommend.py
Check recommendation blending
"""
import pytest
from services import recommend

def test_twoTowerRec(dummy_user_vector, dummy_mission_vector):
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

def test_twoTowerRec_weights(dummy_user_vector, dummy_mission_vector):
    user_mission_embedding = dummy_mission_vector 
    user_lat = 0.0
    user_lon = 0.0

    results_default = recommend.twoTowerRec(dummy_user_vector, user_mission_embedding, user_lat, user_lon)
    results_custom = recommend.twoTowerRec(dummy_user_vector, user_mission_embedding, user_lat, user_lon, alpha=0.7, beta=0.3, gamma=0.0)

    assert results_default != results_custom  # Ensure different weights yield different results


