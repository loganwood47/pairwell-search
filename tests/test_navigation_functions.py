from src.pairwell_search.services import navigation_functions
from unittest.mock import patch, MagicMock
import pandas as pd
import streamlit as st

def test_rebuild_df_edits():
    original_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    editor_state = {
        "edited_rows": {
            "0": {"A": 10},
            "1": {"B": 20}
        },
        "added_rows": [{"A": 7, "B": 8}],
        "deleted_rows": []
    }
    expected_df = pd.DataFrame({'A': [10, 2, 3, 7], 'B': [4, 20, 6, 8]})
    result_df = navigation_functions.rebuild_df(original_df, editor_state)
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))

def test_rebuild_df_added_rows():
    original_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    editor_state = {
        "edited_rows": {},
        "added_rows": [{"A": 5, "B": 6}],
        "deleted_rows": []
    }
    expected_df = pd.DataFrame({'A': [1, 2, 5], 'B': [3, 4, 6]})
    result_df = navigation_functions.rebuild_df(original_df, editor_state)
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))

def test_rebuild_df_deleted_rows():
    original_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    editor_state = {
        "edited_rows": {},
        "added_rows": [],
        "deleted_rows": [1]
    }
    expected_df = pd.DataFrame({'A': [1, 3], 'B': [4, 6]})
    result_df = navigation_functions.rebuild_df(original_df, editor_state)
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))

def test_rebuild_df_combined_operations():
    original_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    editor_state = {
        "edited_rows": {
            "0": {"A": 10},
            "1": {"B": 20}
        },
        "added_rows": [{"A": 7, "B": 8}],
        "deleted_rows": [2]
    }
    expected_df = pd.DataFrame({'A': [10, 2, 7], 'B': [4, 20, 8]})
    result_df = navigation_functions.rebuild_df(original_df, editor_state)

    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))
    
def test_on_editor_change_updates_session_state():
    st.session_state.clear()  # Clear session state for a clean test

    original_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'View Nonprofit Page': [True, False], 'id': [1, 2], 'Donation Activity': ['donation1', 'donation2']})
    editor_state = {
        "edited_rows": {},
        "added_rows": [],
        "deleted_rows": []
    }
    
    # Set up the session state
    st.session_state.original_df = original_df
    st.session_state.data_editor_output = editor_state

    # Call the function
    navigation_functions.on_editor_change()

    # Check the updated session state
    assert st.session_state.updated_df.equals(original_df)
    assert st.session_state["nonprofit_selection"] == 1
    assert st.session_state['recent_donation_data'] == 'donation1'
    assert st.session_state["navigate_to_profile"] is True

def test_on_editor_change_no_selection():
    original_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'View Nonprofit Page': [False, False], 'id': [1, 2], 'Donation Activity': ['donation1', 'donation2']})
    editor_state = {
        "edited_rows": {},
        "added_rows": [],
        "deleted_rows": []
    }
    
    # Set up the session state
    st.session_state.clear()  # Clear session state for a clean test
    st.session_state.original_df = original_df
    st.session_state.data_editor_output = editor_state
    st.session_state['navigate_to_profile'] = False

    # Call the function
    navigation_functions.on_editor_change()

    # Check the updated session state
    assert st.session_state.updated_df.equals(original_df)
    assert "nonprofit_selection" not in st.session_state
    assert "recent_donation_data" not in st.session_state
    assert st.session_state["navigate_to_profile"] is not True

def test_go_home():
    # Set up initial session state
    st.session_state["nonprofit_selection"] = "123"
    st.session_state["nonprofit_donation_data"] = ["donation1"]
    st.session_state["original_df"] = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    st.session_state["updated_df"] = pd.DataFrame({'A': [5], 'B': [6]})

    # Call the function
    navigation_functions.go_home()

    # Check the updated session state
    assert st.session_state["nonprofit_selection"] is None
    assert st.session_state["nonprofit_donation_data"] == []
    assert st.session_state["original_df"] is None
    assert st.session_state["updated_df"] is None
    assert st.session_state["navigate_home"] is True

def test_nav_monitor_initializes_missing_keys():
    """Test that nav_monitor initializes missing session state keys."""
    st.session_state.clear()
    
    # Call the function
    navigation_functions.nav_monitor()
    
    # Check that keys are initialized
    assert st.session_state["nonprofit_selection"] is None
    assert st.session_state["nonprofit_donation_data"] == []

def test_nav_monitor_preserves_existing_keys():
    """Test that nav_monitor preserves existing session state keys."""
    st.session_state.clear()
    st.session_state["nonprofit_selection"] = "123"
    st.session_state["nonprofit_donation_data"] = ["donation1"]
    
    # Call the function
    navigation_functions.nav_monitor()
    
    # Check that existing values are preserved
    assert st.session_state["nonprofit_selection"] == "123"
    assert st.session_state["nonprofit_donation_data"] == ["donation1"]

@patch('src.pairwell_search.services.navigation_functions.st.switch_page')
def test_nav_monitor_triggers_navigation(mock_switch_page):
    """Test that nav_monitor triggers navigation when navigate_to_profile is True."""
    st.session_state.clear()
    st.session_state["nonprofit_selection"] = "123"
    st.session_state["nonprofit_donation_data"] = ["donation1"]
    st.session_state["navigate_to_profile"] = True
    
    # Call the function
    navigation_functions.nav_monitor()
    
    # Check that navigation was triggered
    mock_switch_page.assert_called_once_with("pages/nonprofit_profile_page.py")
    assert st.session_state["navigate_to_profile"] is False

@patch('src.pairwell_search.services.navigation_functions.st.switch_page')
def test_nav_monitor_no_navigation_when_flag_false(mock_switch_page):
    """Test that nav_monitor does not trigger navigation when flag is False."""
    st.session_state.clear()
    st.session_state["nonprofit_selection"] = "123"
    st.session_state["navigate_to_profile"] = False
    
    # Call the function
    navigation_functions.nav_monitor()
    
    # Check that navigation was not triggered
    mock_switch_page.assert_not_called()

def test_initialize_session_df_initializes_when_missing():
    """Test that initialize_session_df initializes original_df and updated_df when missing."""
    st.session_state.clear()
    test_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    
    # Call the function
    navigation_functions.initialize_session_df(test_df)
    
    # Check that both DataFrames are initialized
    pd.testing.assert_frame_equal(st.session_state["original_df"], test_df)
    pd.testing.assert_frame_equal(st.session_state["updated_df"], test_df)

def test_initialize_session_df_initializes_when_original_df_is_none():
    """Test that initialize_session_df initializes when original_df is None."""
    st.session_state.clear()
    st.session_state["original_df"] = None
    test_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    
    # Call the function
    navigation_functions.initialize_session_df(test_df)
    
    # Check that both DataFrames are initialized
    pd.testing.assert_frame_equal(st.session_state["original_df"], test_df)
    pd.testing.assert_frame_equal(st.session_state["updated_df"], test_df)

def test_initialize_session_df_preserves_existing_original_df():
    """Test that initialize_session_df preserves existing original_df."""
    st.session_state.clear()
    existing_df = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    st.session_state["original_df"] = existing_df
    new_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    
    # Call the function
    navigation_functions.initialize_session_df(new_df)
    
    # Check that original_df is preserved
    pd.testing.assert_frame_equal(st.session_state["original_df"], existing_df)

def test_initialize_session_df_initializes_updated_df_when_missing():
    """Test that initialize_session_df initializes updated_df when missing."""
    st.session_state.clear()
    test_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    st.session_state["original_df"] = test_df
    
    # Call the function
    navigation_functions.initialize_session_df(test_df)
    
    # Check that updated_df is initialized as a copy of original_df
    pd.testing.assert_frame_equal(st.session_state["updated_df"], test_df)
    # Ensure it's a copy, not the same object
    assert st.session_state["updated_df"] is not st.session_state["original_df"]

def test_update_profile_navigation_sets_session_state():
    """Test that update_profile_navigation sets the correct session state values."""
    st.session_state.clear()
    nonprofit_id = "test_np_123"
    donation_data = ["donation1", "donation2", "donation3"]
    
    # Call the function
    navigation_functions.update_profile_navigation(nonprofit_id, donation_data)
    
    # Check that session state is set correctly
    assert st.session_state["nonprofit_selection"] == nonprofit_id
    assert st.session_state["recent_donation_data"] == donation_data

def test_update_profile_navigation_overwrites_existing_values():
    """Test that update_profile_navigation overwrites existing session state values."""
    st.session_state.clear()
    st.session_state["nonprofit_selection"] = "old_id"
    st.session_state["recent_donation_data"] = ["old_donation"]
    
    new_nonprofit_id = "new_np_456"
    new_donation_data = ["new_donation1", "new_donation2"]
    
    # Call the function
    navigation_functions.update_profile_navigation(new_nonprofit_id, new_donation_data)
    
    # Check that values are overwritten
    assert st.session_state["nonprofit_selection"] == new_nonprofit_id
    assert st.session_state["recent_donation_data"] == new_donation_data

def test_update_profile_navigation_with_empty_donation_data():
    """Test that update_profile_navigation works with empty donation data."""
    st.session_state.clear()
    nonprofit_id = "test_np_789"
    donation_data = []
    
    # Call the function
    navigation_functions.update_profile_navigation(nonprofit_id, donation_data)
    
    # Check that session state is set correctly
    assert st.session_state["nonprofit_selection"] == nonprofit_id
    assert st.session_state["recent_donation_data"] == []