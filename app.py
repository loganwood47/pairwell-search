import streamlit as st
from services import db, embedding_service, similarity, recommend, visualize, interest_expansion
import pandas as pd

debug = st.toggle("Debug Mode", False)
if debug:
    st.write("Debug mode is ON")

    st.title("Testing Data Funcs")
    np_name = st.text_input("Nonprofit Name")
    np_city = st.text_input("City")
    np_state = st.text_input("State")
    np_mission = st.text_area("Mission Statement")

    nonprofitObj = {
        "name": np_name,
        "city": np_city,
        "state": np_state,
        "mission": np_mission
    }

    if st.button("Add Nonprofit"):
        if np_name and np_city and np_state and np_mission:
            db.add_nonprofit(nonprofitObj)
            st.success("Nonprofit added!")
        else:
            st.error("Please fill in all fields.")


    nonprofits = db.get_nonprofits()

    st.write("Current Nonprofits in DB:")
    for np in nonprofits:
        st.write(f"- {np['name']} ({np['city']}, {np['state']}, {np['mission']})")

st.title("Nonprofit Recommender")

# Step 1: User input
geo = st.text_input("Geography")
income = st.number_input("Income", min_value=0)
interests = st.text_area("Interests (comma-separated)").split(",")

if st.button("Find Recommendations"):
    # TODO: Replace these with vector search once DB implemented
    with st.spinner("Searching for nonprofits..."):
        expanded_interests = interest_expansion.expand_interest([i.strip() for i in interests if i.strip()])
        if debug:
            st.write("Mission Statement for user's ideal Nonprofit:", expanded_interests)
        profile = {"geography": geo, "income": income, "interests": expanded_interests}
        user_vec = embedding_service.embed_user_profile(profile)

        # Fetch nonprofits & embed
        nonprofits = db.get_nonprofits()
        texts = [n["mission"] for n in nonprofits]
        nonprofits_by_id = {n["id"]: n for n in nonprofits}
        nonprofit_vecs = embedding_service.embed_texts(texts)
        # nonprofit_vecs = embedding_service.embed_nonprofit_profiles(nonprofits)

        # Build vector index
        vs = similarity.VectorSearch(nonprofit_vecs.shape[1])
        vs.add_vectors(nonprofit_vecs, [n["id"] for n in nonprofits])

        # Recommend
        results = recommend.get_recommendations(user_vec, vs)

        st.title("Top Recommended Nonprofits:")

        data = []
        for r in results:
            np_info = nonprofits_by_id[r["id"]]
            data.append({
            "Nonprofit Name": np_info['name'],
            "Mission": np_info['mission'],
            "Logo": np_info.get("logo_url"),
            "Website": np_info.get('website'),
            "Score": r['score']
            })

        df = pd.DataFrame(data)

        st.dataframe(df,
                     column_config={
                         "Nonprofit Name": st.column_config.TextColumn("Nonprofit Name"),
                         "Mission": st.column_config.TextColumn("Mission"),
                         "Logo": st.column_config.ImageColumn("Logo", width=100),
                         "Website": st.column_config.LinkColumn("Website"),
                         "Score": st.column_config.NumberColumn("Score", format="%.4f")
                     },
                     hide_index=False)
                

        # Optional: visualize embeddings
        # visualize.plot_embeddings(nonprofit_vecs, [n["name"] for n in nonprofits])
