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