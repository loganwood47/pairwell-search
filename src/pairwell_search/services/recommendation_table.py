import streamlit as st
import pandas as pd
import random
from src.pairwell_search.services import db, similarity, recommend

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

def showTable(np_ids):

    nonprofits = db.get_nonprofits_by_id(ids=np_ids)

    st.title("Top Recommended Nonprofits:")

    data = []
    for r in np_ids:
        np_info = [nonprofits[i] for i in range(len(nonprofits)) if nonprofits[i]['id'] == r][0]
        result_info = [rec for rec in recs if rec['id'] == r][0]
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
            "Recommendation Score": result_info['total_similarity'],
            "Relevance Prediction": result_info['model_similarity'],
            "Mission Similarity": result_info['mission_similarity'],
            "Geo Similarity": result_info['geo_distance_meters']
        })

    df = pd.DataFrame(data)

    if "original_df" not in st.session_state or st.session_state["original_df"] is None:
        st.session_state["original_df"] = df

    if "updated_df" not in st.session_state:
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
                                        "Donation Activity": st.column_config.AreaChartColumn("Recent Donation Activity"),
                                        "Total Similarity Score": st.column_config.NumberColumn("Total Similarity Score", format="%.4f"),
                                        "Model Similarity Score": st.column_config.NumberColumn("Model Prediction Score", format="%.4f"),
                                        "Mission Similarity Score": st.column_config.NumberColumn("Mission Score", format="%.4f"),
                                        "Geo Distance (meters)": st.column_config.NumberColumn("Geo Distance (meteres)", format="%d"),
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