"""
seed_user_activity.py
Synthetic multi-step activity generator for users <-> nonprofits
"""

import random
import numpy as np
from ...db import supabase, get_user_vector, get_nonprofit_vector


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if vec1 is None or vec2 is None:
        return 0.0
    v1, v2 = np.array(vec1), np.array(vec2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    return float(np.dot(v1, v2) / denom)


def generate_activity_sequence(user_id, nonprofit_id, engagement_types):
    """Generate activity sequence weighted by similarity between user & nonprofit."""

    user_vec = get_user_vector(user_id)
    # print(user_vec)
    nonprofit_vec = get_nonprofit_vector(nonprofit_id)
    sim = cosine_similarity(user_vec, nonprofit_vec)
    prob_factor = max(0.0, (sim + 1.0) / 2.0)  # scale -1..1 → 0..1

    activities = []

    # Always View
    et_view = [e for e in engagement_types if e['engagement_type'] == "View"][0]
    activities.append({
        "user_id": user_id,
        "nonprofit_id": nonprofit_id,
        "engagement_type_id": et_view["id"],
    })

    # Probability ladder
    if random.random() < 0.05 + 0.5 * prob_factor:
        et_click = [e for e in engagement_types if e['engagement_type'] == "Click"][0]
        activities.append({
            "user_id": user_id,
            "nonprofit_id": nonprofit_id,
            "engagement_type_id": et_click["id"],
        })

        if random.random() < 0.01 + 0.2 * prob_factor:
            et_share = [e for e in engagement_types if e['engagement_type'] == "Share"][0]
            activities.append({
                "user_id": user_id,
                "nonprofit_id": nonprofit_id,
                "engagement_type_id": et_share["id"],
            })

        if random.random() < 0.002 + 0.1 * prob_factor:
            et_vol = [e for e in engagement_types if e['engagement_type'] == "Volunteer"][0]
            activities.append({
                "user_id": user_id,
                "nonprofit_id": nonprofit_id,
                "engagement_type_id": et_vol["id"],
            })

        if random.random() < 0.005 + 0.05 * prob_factor:
            et_donate = [e for e in engagement_types if e['engagement_type'] == "Donate"][0]
            activities.append({
                "user_id": user_id,
                "nonprofit_id": nonprofit_id,
                "engagement_type_id": et_donate["id"],
            })

    return activities

engagement_types = supabase.table("engagement_types").select("*").execute().data

test_sq = generate_activity_sequence(17, 15, engagement_types)
print(test_sq)
# print(engagement_types[0])