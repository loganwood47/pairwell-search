from src.pairwell_search.services import navigation_functions
from unittest.mock import patch, MagicMock
import pandas as pd


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