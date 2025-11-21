import streamlit as st

st.write(
     st.session_state["nonprofit_selection"]
)

if st.session_state.get("navigate_home") is True:
    st.session_state["navigate_home"] = False
    st.switch_page("pages/recommendation_engine_page.py")


st.session_state.setdefault("nonprofit_selection", None)
st.session_state.setdefault("nonprofit_donation_data", [])

st.title("Nonprofit Profile")
st.write("Selected nonprofit ID:", st.session_state["nonprofit_selection"])

def go_home():
    st.session_state["nonprofit_selection"] = None
    st.session_state["nonprofit_donation_data"] = []
    st.session_state["navigate_home"] = True  # flag triggers navigation on rerun

st.button("Go Home", on_click=go_home)