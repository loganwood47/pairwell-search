import streamlit as st

# Ensure session state keys exist globally
if "original_df" not in st.session_state:
    st.session_state.original_df = None

if "updated_df" not in st.session_state:
    st.session_state.updated_df = None

page1 = st.Page("pages/recommendation_engine_page.py", title="Home", icon=":material/home:")
page2 = st.Page("pages/nonprofit_profile_page.py", title="Next Step", icon=":material/arrow_forward:")

nav = st.navigation([page1, page2])
nav.run()