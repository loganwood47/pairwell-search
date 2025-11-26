import streamlit as st
import pandas as pd


def rebuild_df(original_df, editor_state):
    """Apply edited_rows / added_rows / deleted_rows to reconstruct the DF."""
    df = original_df.copy()

    # Apply edits
    for idx, edits in editor_state.get("edited_rows", {}).items():
        idx = int(idx)
        for col, val in edits.items():
            df.at[idx, col] = val

    # Added rows
    for row in editor_state.get("added_rows", []):
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Deleted rows
    deleted = editor_state.get("deleted_rows", [])
    if deleted:
        df = df.drop(index=deleted).reset_index(drop=True)

    return df

def on_editor_change():
    state = st.session_state.data_editor_output
    updated_df = rebuild_df(st.session_state.original_df, state)
    st.session_state.updated_df = updated_df

    selected = updated_df[updated_df["View Nonprofit Page"] == True]

    if len(selected) > 0:
        np_id = selected.iloc[0].id
        st.session_state["nonprofit_selection"] = np_id
        st.session_state['recent_donation_data'] = selected.iloc[0]["Donation Activity"]
        st.session_state["navigate_to_profile"] = True

def nav_monitor():
    if "nonprofit_selection" not in st.session_state:
        st.session_state["nonprofit_selection"] = None

    if "nonprofit_donation_data" not in st.session_state:
        st.session_state["nonprofit_donation_data"] = []

    if st.session_state.get("navigate_to_profile") is True:
        st.session_state["navigate_to_profile"] = False
        st.switch_page("pages/nonprofit_profile_page.py")

def initialize_session_df(df):
    if "original_df" not in st.session_state or st.session_state["original_df"] is None:
        st.session_state["original_df"] = df

    if "updated_df" not in st.session_state:
        st.session_state.updated_df = st.session_state.original_df.copy()

def update_profile_navigation(nonprofit_id: str, donation_data: list):
    st.session_state["nonprofit_selection"] = nonprofit_id
    st.session_state["recent_donation_data"] = donation_data
    # st.session_state["navigate_to_profile"] = True  # flag triggers navigation on rerun

def go_home():
    st.session_state["nonprofit_selection"] = None
    st.session_state["nonprofit_donation_data"] = []
    st.session_state["original_df"] = None
    st.session_state["updated_df"] = None
    st.session_state["navigate_home"] = True  # flag triggers navigation on rerun
