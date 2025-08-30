"""
test_db.py
Minimal tests for db.py
Note: This only checks if functions are callable.
For real DB, you should mock Supabase responses.
"""

from services import db

def test_db_module_has_functions():
    assert hasattr(db, "get_nonprofits")
    assert hasattr(db, "save_user")
    assert hasattr(db, "save_user_activity")
