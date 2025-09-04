"""
recommend.py
Blends content-based similarity and collaborative filtering
"""

from services.similarity import VectorSearch
# from services.collaborative import get_collaborative_scores   # (optional, later)

def get_recommendations(user_vec, vector_search: VectorSearch, alpha=0.5, beta=0.5): #alpha=0.7, beta=0.3):
    """Blend content similarity with collaborative signals"""
    # Content-based
    content_scores = vector_search.search(user_vec, k=10)

    # TODO: break into individual embeddings per component (geo, mission etc)
            # then final score is weighted average of component similarities

    # Collaborative (stub: set all to 0.0 for now)
    collab_scores = {c["id"]: 0.0 for c in content_scores}

    # Blend
    results = []
    for c in content_scores:
        blended = alpha * c["score"] + beta * collab_scores.get(c["id"], 0.0)
        results.append({"id": c["id"], "score": blended})

    # Sort by blended score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
