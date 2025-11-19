import streamlit as st
import pandas as pd

import random

from src.pairwell_search.services import db, embedding_service, similarity, recommend, visualize, interest_expansion, geocode_city
from src.pairwell_search.models.two_tower import TwoTower

model_path = "src/pairwell_search/models/two_tower_trained_1758161149.pt"
preprocessing_path = "src/pairwell_search/models/preprocessing_1758161083.pkl"

model, preprocessing = similarity.load_model_and_preprocessing(
    model_path, preprocessing_path,
    n_users=1, n_nonprofits=1,
    text_emb_dim=128, embed_dim=64, cat_emb_dim=16
)

engagement_options = ["Volunteering", "Donating", "Advocacy", "Event Participation", "Other"]

st.image("logos/Blue Long.png", width=200)

st.title("Welcome to PairWell! Tell us about yourself to find relevant nonprofits.")

city = st.text_input("City")
state = st.text_input("State")
# income = st.number_input("Income", min_value=0)
income = st.selectbox("Income", options=[0, 25000, 50000, 75000, 100000, 150000, 250000], index=0)
interests = st.text_area("Interests (comma-separated)").split(",")
engagement_prefs = st.multiselect("Preferred Engagement Types", engagement_options)


# user_id = 0
# nonprofit_ids = [random.randint(1, 5000) for _ in range(10)]  # Example nonprofit IDs




if st.button("Get Recommendations"):
    raw_user = {
        "city": city,
        "state": state,
        "income": income,
        "interests": [i.strip() for i in interests if i.strip()],
        "engagement_prefs": [e.strip() for e in engagement_prefs if e.strip()]
    }
    user_emb = similarity.get_user_embedding(model, preprocessing, raw_user)
    
    with st.spinner("Generating expanded interests and user embeddings..."):
        # Put these functions inside the button to avoid extra calls
        user_lat, user_lon = geocode_city.geocode_city(city, state) if city and state else (None, None)

        expanded_interests = interest_expansion.expand_interest([i.strip() for i in interests if i.strip()]) if interests else []

        embedded_interests = embedding_service.embed_texts(expanded_interests) if interests else None

    # recs = recommend.twoTowerRec(user_emb, embedded_interests.mean(axis=0), user_lat, user_lon, alpha=0.7, beta=0.3, gamma=0.0)
    with st.spinner("Finding best nonprofit matches..."):
        recs = recommend.twoTowerRec(
            user_emb, 
            embedded_interests, 
            user_lat, 
            user_lon, 
            alpha=0.7, # trained model prediction weight
            beta=0.4,  # mission weight
            gamma=0.3) # geo weight

        np_ids = [r["id"] for r in recs]
        nonprofits = db.get_nonprofits_by_id(ids=np_ids)

        st.title("Top Recommended Nonprofits:")

        data = []
        for r in np_ids:
            np_info = [nonprofits[i] for i in range(len(nonprofits)) if nonprofits[i]['id'] == r][0]
            result_info = [rec for rec in recs if rec['id'] == r][0]
            data.append({
            "Nonprofit Name": np_info['name'],
            "Mission": np_info['mission'],
            "City": np_info.get('city'),
            "State": np_info.get('state'),
            "Logo": np_info.get("logo_url"),
            "Website": np_info.get('website'),
            "Recommendation Score": result_info['total_similarity'],
            "Relevance Prediction": result_info['model_similarity'],
            "Mission Similarity": result_info['mission_similarity'],
            "Geo Similarity": result_info['geo_distance_meters']
            })

        df = pd.DataFrame(data)

        st.dataframe(df,
                        column_config={
                            "Nonprofit Name": st.column_config.TextColumn("Nonprofit Name"),
                            "Mission": st.column_config.TextColumn("Mission"),
                            "City": st.column_config.TextColumn("City"),
                            "State": st.column_config.TextColumn("State"),
                            "Logo": st.column_config.ImageColumn("Logo", width=100),
                            "Website": st.column_config.LinkColumn("Website"),
                            "Total Similarity Score": st.column_config.NumberColumn("Total Similarity Score", format="%.4f"),
                            "Model Similarity Score": st.column_config.NumberColumn("Model Prediction Score", format="%.4f"),
                            "Mission Similarity Score": st.column_config.NumberColumn("Mission Score", format="%.4f"),
                            "Geo Distance (meters)": st.column_config.NumberColumn("Geo Distance (meteres)", format="%d"),
                        },
                        hide_index=False)
        
        nodes, edges = visualize.fetch_edges_for_graph(np_ids, 10)
        nodes_data = visualize.fetch_node_attributes(nodes)
        graph = visualize.build_graph("You", user_emb, np_ids, nodes_data, edges)
        visualize.show_graph(graph)
