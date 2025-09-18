"""
test_db.py
Minimal tests for db.py
Note: This only checks if functions are callable.
For real DB, you should mock Supabase responses.
"""

from src.pairwell_search.services import db
from unittest.mock import patch

def test_db_module_has_functions():
    assert hasattr(db, "get_nonprofits")
    assert hasattr(db, "save_user")
    assert hasattr(db, "save_user_activity")


def test_get_nonprofit_vector():
    nonprofit_id = 100
    vector = db.get_nonprofit_vector(nonprofit_id)
    assert vector is not None
    assert isinstance(vector, list)

def test_get_nonprofit_vector_nonexistent():
    # Test for a nonexistent nonprofit ID
    nonprofit_id = -1
    vector = db.get_nonprofit_vector(nonprofit_id)
    assert vector is None


def test_get_nonprofits_by_id():
    nonprofit_ids = [1, 2, 3]
    nonprofits = db.get_nonprofits_by_id(ids=nonprofit_ids)
    assert nonprofits is not None
    assert isinstance(nonprofits, list)
    assert all('id' in nonprofit for nonprofit in nonprofits)

def test_get_nonprofits_by_id_nonexistent():
    nonprofit_ids = [-1, -2]
    nonprofits = db.get_nonprofits_by_id(ids=nonprofit_ids)
    assert nonprofits == []

def test_get_nonprofit_by_ein():
    ein = "88-4183627"  # Replace with a valid EIN for testing
    nonprofit = db.get_nonprofit_by_ein(ein=ein)
    assert nonprofit is not None
    assert isinstance(nonprofit, list)
    assert all('ein' in item for item in nonprofit)

def test_get_nonprofit_by_ein_nonexistent():
    ein = "000000000"  # Nonexistent EIN
    nonprofit = db.get_nonprofit_by_ein(ein=ein)
    assert nonprofit == []

# @patch('services.db.save_user_activity')
# def test_save_user_activity(mock_save_user_activity):
#     user_id = 1
#     nonprofit_id = 100
#     engagement_type = "viewed"
#     mock_save_user_activity.return_value = [{'user_id': user_id, 'nonprofit_id': nonprofit_id, 'engagement_type': engagement_type}]
    
#     response = db.save_user_activity(user_id, nonprofit_id, engagement_type)
#     assert response is not None
#     assert isinstance(response, list)
#     assert all('user_id' in item for item in response)
#     assert all('nonprofit_id' in item for item in response)
#     assert all('engagement_type' in item for item in response)

# def test_save_user_activity_invalid_user():
#     user_id = -1
#     nonprofit_id = 100
#     engagement_type = "viewed"
#     response = db.save_user_activity(user_id, nonprofit_id, engagement_type)
#     assert response is None  # Assuming the response is None for invalid user

# def test_save_user_activity_invalid_nonprofit():
#     user_id = 1
#     nonprofit_id = -1
#     engagement_type = "viewed"
#     response = db.save_user_activity(user_id, nonprofit_id, engagement_type)
#     assert response is None  # Assuming the response is None for invalid nonprofit