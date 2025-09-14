"""
seed_user_activity.py
Synthetic multi-step activity generator for users <-> nonprofits
"""

import random
from db import supabase, get_engagement_types, get_users, get_nonprofits

# --- Generator ---
def generate_activity_sequence(user_id, nonprofit_id, engagement_types):
    """
    Always logs a View, then probabilistically logs Click/Share/Donate/Volunteer.
    Returns a list of activities (dicts).
    """

    activities = []

    # Always start with a View
    et_view = engagement_types["View"]
    activities.append({
        "user_id": user_id,
        "nonprofit_id": nonprofit_id,
        "engagement_type_id": et_view["id"],
        "weight": et_view["weight"],
    })

    # Then maybe upgrade
    if random.random() < 0.2:  # 20% chance of Click after View
        et_click = engagement_types["Click"]
        activities.append({
            "user_id": user_id,
            "nonprofit_id": nonprofit_id,
            "engagement_type_id": et_click["id"],
            "weight": et_click["train_weight"],
        })

        if random.random() < 0.1:  # 10% chance of Share after Click
            et_share = engagement_types["Share"]
            activities.append({
                "user_id": user_id,
                "nonprofit_id": nonprofit_id,
                "engagement_type_id": et_share["id"],
                "weight": et_share["train_weight"],
            })

        if random.random() < 0.05:  # 5% chance of Donation after Click
            et_donate = engagement_types["Donation"]
            activities.append({
                "user_id": user_id,
                "nonprofit_id": nonprofit_id,
                "engagement_type_id": et_donate["id"],
                "weight": et_donate["train_weight"],
            })

        if random.random() < 0.02:  # 2% chance of Volunteering after Click
            et_vol = engagement_types["Volunteer"]
            activities.append({
                "user_id": user_id,
                "nonprofit_id": nonprofit_id,
                "engagement_type_id": et_vol["id"],
                "weight": et_vol["train_weight"],
            })

    return activities


def seed_user_activity(num_users=500, nonprofits_per_user=20):
    users = get_users(limit=num_users)
    nonprofits = get_nonprofits(limit=2000)
    engagement_types = get_engagement_types()

    total = 0
    for user in users:
        chosen_nonprofits = random.sample(nonprofits, nonprofits_per_user)
        for np in chosen_nonprofits:
            activities = generate_activity_sequence(user["id"], np["id"], engagement_types)
            for act in activities:
                supabase.table("user_activity").insert(act).execute()
                total += 1

    print(f"Seeded {total} user_activity rows across {num_users} users.")


if __name__ == "__main__":
    seed_user_activity(num_users=500, nonprofits_per_user=20)
