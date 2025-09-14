"""
seed_user_activity.py
Synthetic multi-step activity generator for users <-> nonprofits
"""

import random
import numpy as np
from db import supabase, get_user_vector, get_nonprofit_vector


def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0.0
    v1, v2 = np.array(vec1), np.array(vec2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    return float(np.dot(v1, v2) / denom)


def generate_activity_sequence(user_id, nonprofit_id, engagement_types):
    """Generate activity sequence weighted by similarity between user & nonprofit."""

    user_vec = get_user_vector(user_id)
    nonprofit_vec = get_nonprofit_vector(nonprofit_id)
    sim = cosine_similarity(user_vec, nonprofit_vec)
    prob_factor = max(0.0, (sim + 1.0) / 2.0)  # scale -1..1 → 0..1

    activities = []

    # Always View
    et_view = engagement_types["View"]
    activities.append({
        "user_id": user_id,
        "nonprofit_id": nonprofit_id,
        "engagement_type_id": et_view["id"],
        "weight": et_view["weight"],
    })

    # Probability ladder
    if random.random() < 0.05 + 0.5 * prob_factor:
        et_click = engagement_types["Click"]
        activities.append({
            "user_id": user_id,
            "nonprofit_id": nonprofit_id,
            "engagement_type_id": et_click["id"],
            "weight": et_click["weight"],
        })

        if random.random() < 0.01 + 0.2 * prob_factor:
            et_share = engagement_types["Share"]
            activities.append({
                "user_id": user_id,
                "nonprofit_id": nonprofit_id,
                "engagement_type_id": et_share["id"],
                "weight": et_share["train_weight"],
            })

        if random.random() < 0.002 + 0.1 * prob_factor:
            et_vol = engagement_types["Volunteer"]
            activities.append({
                "user_id": user_id,
                "nonprofit_id": nonprofit_id,
                "engagement_type_id": et_vol["id"],
                "weight": et_vol["train_weight"],
            })

        if random.random() < 0.005 + 0.05 * prob_factor:
            et_donate = engagement_types["Donation"]
            activities.append({
                "user_id": user_id,
                "nonprofit_id": nonprofit_id,
                "engagement_type_id": et_donate["id"],
                "weight": et_donate["train_weight"],
            })

    return activities
