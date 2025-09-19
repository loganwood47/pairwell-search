"""
seed_user_activity.py
Synthetic multi-step activity generator for users <-> nonprofits
Simulates user interactions based on interest/mission vector similarity
"""

import random
import numpy as np
from ...db import supabase
from tqdm import tqdm
import json


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if vec1 is None or vec2 is None:
        return 0.0
    v1, v2 = np.array(vec1), np.array(vec2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    return float(np.dot(v1, v2) / denom)


def generate_activity_sequence(user: dict, nonprofit: dict, engagement_types: dict) -> list[dict]:
    """Generate activity sequence weighted by similarity between user & nonprofit."""

    # user_vec = get_user_vector(user_id)
    # print(user_vec)
    # nonprofit_vec = get_nonprofit_vector(nonprofit_id)
    sim = cosine_similarity(user['vector'], nonprofit['vector'])
    prob_factor = max(0.0, (sim + 1.0) / 2.0)  # scale -1..1 → 0..1

    activities = []

    # Always View
    et_view = [e for e in engagement_types if e['engagement_type'] == "View"][0]
    activities.append({
        "user_id": user["id"],
        "nonprofit_id": nonprofit["id"],
        "engagement_type_id": et_view["id"],
    })

    # Probability ladder
    if random.random() < 0.05 + 0.5 * prob_factor:
        et_click = [e for e in engagement_types if e['engagement_type'] == "Click"][0]
        activities.append({
            "user_id": user["id"],
            "nonprofit_id": nonprofit["id"],
            "engagement_type_id": et_click["id"],
        })

        if random.random() < 0.01 + 0.2 * prob_factor:
            et_share = [e for e in engagement_types if e['engagement_type'] == "Share"][0]
            activities.append({
                "user_id": user["id"],
                "nonprofit_id": nonprofit["id"],
                "engagement_type_id": et_share["id"],
            })

        if random.random() < 0.002 + 0.1 * prob_factor:
            et_vol = [e for e in engagement_types if e['engagement_type'] == "Volunteer"][0]
            activities.append({
                "user_id": user["id"],
                "nonprofit_id": nonprofit["id"],
                "engagement_type_id": et_vol["id"],
            })

        if random.random() < 0.005 + 0.05 * prob_factor:
            et_donate = [e for e in engagement_types if e['engagement_type'] == "Donation"][0]
            activities.append({
                "user_id": user["id"],
                "nonprofit_id": nonprofit["id"],
                "engagement_type_id": et_donate["id"],
            })

    return activities

def seed_activities(num_samples=50000, batch_size=5000):
    # Fetch engagement types
    engagement_types = {}
    et_data = supabase.table("engagement_types").select("*").execute().data
    # for et in et_data:
    #     engagement_types[et["engagement_type"]] = et
    engagement_types = et_data

    # Fetch users + vectors
    rawUsers = supabase.table("user_interest_vectors").select("user_id, vector").execute().data
    users = [{"id": u["user_id"], "vector": json.loads(u["vector"])} for u in rawUsers]
    

    # users = [{"id": u["user_id"], "vector": u["vector"]} for u in users]
    # Fetch nonprofits + vectors
    rawNonprofits = supabase.table("nonprofit_mission_vectors").select("nonprofit_id, vector").execute().data
    nonprofits = [{"id": n["nonprofit_id"], "vector": json.loads(n["vector"])} for n in rawNonprofits]

    all_activities = []
    for _ in tqdm(range(num_samples)):
        user = random.choice(users)
        nonprofit = random.choice(nonprofits)
        acts = generate_activity_sequence(user, nonprofit, engagement_types)
        all_activities.extend(acts)

        # Bulk insert when batch fills
        if len(all_activities) >= batch_size:
            supabase.table("user_activity").insert(all_activities).execute()
            all_activities = []

    # Insert leftover
    if all_activities:
        supabase.table("user_activity").insert(all_activities).execute()

    print(f"Seeded {num_samples} samples into user_activity")

seed_activities(num_samples=100000, batch_size=5000)
