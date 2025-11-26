import streamlit as st
import pandas as pd
import random

from src.pairwell_search.services.db import get_nonprofits_by_id, get_nonprofit_projects, get_nonprofit_key_employees, get_nonprofit_board_members, get_nonprofit_financials, get_np_network_edges_by_id
# from src.pairwell_search.services.visualize import plot_np_network
from src.pairwell_search.services.navigation_functions import on_editor_change, nav_monitor

nav_state = nav_monitor()

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

np_network = get_np_network_edges_by_id([nonprofit["id"]], top_k=10)
st.write("## Other Nonprofits You May Be Interested In:")

similarly_connected_np_ids = [edge["nonprofit_id_b"] for edge in np_network]

similar_nps = get_nonprofits_by_id(ids=similarly_connected_np_ids)

data = []
for r in similarly_connected_np_ids:
    np_info = [similar_nps[i] for i in range(len(similar_nps)) if similar_nps[i]['id'] == r][0]
    data.append({
        "id": r,
        "View Nonprofit Page": False,
        "Nonprofit Name": np_info['name'],
        "Mission": np_info['mission'],
        "City": np_info.get('city'),
        "State": np_info.get('state'),
        "Logo": np_info.get("logo_url"),
        "Website": np_info.get('website'),
        "Donation Activity": [random.randint(0, 100) for _ in range(30)],  # Placeholder random data to simulate donations
    })

df = pd.DataFrame(data)

st.session_state["original_df"] = df
st.session_state.updated_df = st.session_state.original_df.copy()

selectionDF = st.data_editor(st.session_state.original_df,
                                    key="data_editor_output",
                                    on_change=on_editor_change,
                                        column_config={
                                            "View Nonprofit Page": st.column_config.CheckboxColumn("View Nonprofit Page"),
                                            "Nonprofit Name": st.column_config.TextColumn("Nonprofit Name"),
                                            "Mission": st.column_config.TextColumn("Mission"),
                                            "City": st.column_config.TextColumn("City"),
                                            "State": st.column_config.TextColumn("State"),
                                            "Logo": st.column_config.ImageColumn("Logo", width=100),
                                            "Website": st.column_config.LinkColumn("Website"),
                                            "Donation Activity": st.column_config.AreaChartColumn("Recent Donation Activity")
                                        },
                                        hide_index=False,
                                        num_rows='fixed',
                                        disabled=[
                                            "id", "Nonprofit Name", "Mission", "City", "State", "Logo", "Website",
                                            "Recommendation Score", "Relevance Prediction", "Mission Similarity", "Geo Similarity"
                                        ],
                                        column_order=[
                                            "View Nonprofit Page", "Nonprofit Name", "Mission", "City", "State", "Logo", "Website",
                                            "Donation Activity", "Recommendation Score", "Relevance Prediction", "Mission Similarity", "Geo Similarity"
                                        ]
                                        )
        
st.session_state["edited_df"] = selectionDF




def go_home():
    st.session_state["nonprofit_selection"] = None
    st.session_state["nonprofit_donation_data"] = []
    st.session_state["original_df"] = None
    st.session_state["updated_df"] = None
    st.session_state["navigate_home"] = True  # flag triggers navigation on rerun

st.button("Go Home", on_click=go_home)