import streamlit as st

from src.pairwell_search.services.db import get_nonprofits_by_id, get_nonprofit_projects, get_nonprofit_key_employees, get_nonprofit_board_members, get_nonprofit_financials

if st.session_state.get("navigate_home") is True:
    st.session_state["navigate_home"] = False
    st.switch_page("pages/recommendation_engine_page.py")


st.session_state.setdefault("nonprofit_selection", None)
st.session_state.setdefault("recent_donation_data", [])


nonprofit = get_nonprofits_by_id(ids=[st.session_state["nonprofit_selection"]])[0]
projects = get_nonprofit_projects(nonprofit_id=nonprofit["id"])
board_members = get_nonprofit_board_members(nonprofit_id=nonprofit["id"])

colTitle, colImage = st.columns(2)
with colTitle:
    st.title("Profile: *{}*".format(nonprofit["name"]))

with colImage:
    try:
        st.image(nonprofit["logo_url"], width=200)
    except Exception:
        pass

st.write(nonprofit["mission"])

col1, col2 = st.columns(2, border=True)

with col1:

    st.write("## Website")
    st.write("[{}]({})".format(nonprofit["website"], nonprofit["website"]))

    st.write("## Projects")
    if len(projects) == 0:
        st.write("No projects found.")
    for project in projects:
        st.subheader(project["name"])
        st.write(project["description"])
        if project["areas_served"]:
            areas_served_string = ", ".join(project["areas_served"])
            st.write("Areas served: {}".format(areas_served_string))


with col2:
    with st.container(border=True):
        st.write("## Recent Donation Activity Index")
        donation_data = st.session_state.get("recent_donation_data", [])
        if len(donation_data) == 0:
            st.write("No recent donation activity found.")
        else:
            chart = st.area_chart(donation_data)

    st.write("## Board Members")
    if len(board_members) == 0:
        st.write("No board members found.")
    for member in board_members:
        st.write("#### {} - {}".format(member["name"], member["company"]))
        if member["title"]:
            st.write("Title: {}".format(member["title"]))
        else:
            pass



def go_home():
    st.session_state["nonprofit_selection"] = None
    st.session_state["nonprofit_donation_data"] = []
    st.session_state["original_df"] = None
    st.session_state["updated_df"] = None
    st.session_state["navigate_home"] = True  # flag triggers navigation on rerun

st.button("Go Home", on_click=go_home)