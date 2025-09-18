"""
recommend.py
Blends content-based similarity and collaborative filtering
"""
import torch
from models.two_tower import TwoTower
from services.db import supabase
import numpy as np

from services.similarity import VectorSearch, get_user_embedding
from services.embedding_service import embed_texts

# from services.collaborative import get_collaborative_scores   # (optional, later)

def twoTowerRec(user_embedding, user_mission_embedding, user_lat, user_lon, alpha=0.5, beta=0.5, gamma=0.0):
    """Get recommendations for a user using TwoTower model and vector search"""
    normAlpha = alpha / (alpha + beta + gamma)
    normBeta = beta / (alpha + beta + gamma)
    normGamma = gamma / (alpha + beta + gamma)

    resp = supabase.rpc(
        "match_nonprofits", 
        {
            "query_embedding": user_embedding.tolist(), 
            "match_count": 10,
            "query_mission_embedding": user_mission_embedding.tolist(),
            "query_lat": user_lat,
            "query_lon": user_lon,
            "model_weight": normAlpha,
            "mission_weight": normBeta,
            "geo_weight": normGamma
            }
    ).execute()

    recs = resp.data

    return recs


def get_recommendations(user_vec, vector_search: VectorSearch, alpha=0.5, beta=0.5, gamma=0.0): #alpha=0.7, beta=0.3):
    """Blend content similarity with other vector distances to get total rec"""
    # Content-based
    content_scores = vector_search.search(user_vec, k=10)

    # TODO: break into individual embeddings per component (geo, mission etc)
            # then final score is weighted average of component similarities

    # Collaborative (stub: set all to 0.0 for now)
    collab_scores = {c["id"]: 0.0 for c in content_scores}

    # Location (stub: set all to 0.0 for now)
    location_scores = {c["id"]: 0.0 for c in content_scores}

    # Blend
    results = []
    for c in content_scores:
        blended = alpha * c["score"] + beta * collab_scores.get(c["id"], 0.0) + gamma * collab_scores.get(c["id"], 0.0)
        results.append({"id": c["id"], "score": blended})

    # Sort by blended score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
