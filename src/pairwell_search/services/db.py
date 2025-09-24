"""
db.py
Handles database connection and queries to Supabase
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client
import numpy as np
import json
import streamlit as st

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
if not SUPABASE_URL:
    SUPABASE_URL = st.secrets.SUPABASE_URL

SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_KEY:
    SUPABASE_KEY = st.secrets.SUPABASE_KEY

def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Supabase credentials not set")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase: Client = get_supabase_client()


def get_nonprofits(limit: int = 1000):
    """Fetch nonprofits from DB"""
    resp = supabase.table("nonprofits").select("*").limit(limit).execute()
    return resp.data

def get_nonprofits_by_id(limit: int = 1000, ids: list[int] = [1]):
    """Fetch nonprofits from DB"""
    resp = supabase.table("nonprofits").select("*").in_("id", ids).limit(limit).execute()
    return resp.data

def get_nonprofit_by_ein(limit: int = 1000, ein: str = ""):
    """Fetch nonprofits from DB"""
    resp = supabase.table("nonprofits").select("*").eq("ein", ein).limit(limit).execute()
    return resp.data

def add_nonprofit(nonprofit: dict):
    """Save a new nonprofit"""
    resp = supabase.table("nonprofits").insert(nonprofit).execute()
    return resp.data


def save_user(user_profile: dict):
    """Save a new user profile"""
    resp = supabase.table("users").insert(user_profile).execute()
    return resp.data


def save_user_activity(user_id: int, nonprofit_id: int, engagement_type: str):
    """Log user activity (e.g. viewed, donated, interacted)"""
    data = {"user_id": user_id, "nonprofit_id": nonprofit_id, "engagement_type": engagement_type}
    resp = supabase.table("user_activity").insert(data).execute()
    return resp.data

def get_user_by_id(user_id: int):
    """Fetch a user by ID"""
    resp = supabase.table("users").select("*").eq("id", user_id).execute()
    return resp.data

# --- USER INTEREST VECTORS ---
def store_user_vector(user_id: str, vector: list[float]) -> dict:
    """Insert or update a user's embedding vector."""
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    response = (
        supabase.table("user_interest_vectors")
        .upsert({"user_id": user_id, "vector": vector})
        .execute()
    )
    return response.data


def get_user_vector(user_id: int) -> list[float] | None:
    """Fetch a user's embedding vector by ID."""
    response = (
        supabase.table("user_interest_vectors")
        .select("vector")
        .eq("user_id", user_id)
        .execute()
    )
    if response.data:
        vector = response.data[0]["vector"]
        if isinstance(vector, str):
            vector = json.loads(vector)
        return vector
    return None


# --- NONPROFIT MISSION VECTORS ---
def store_nonprofit_vector(nonprofit_id: str, vector: list[float]) -> dict:
    """Insert or update a nonprofit's embedding vector."""
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    response = (
        supabase.table("nonprofit_mission_vectors")
        .upsert({"nonprofit_id": nonprofit_id, "vector": vector})
        .execute()
    )
    return response.data


def get_nonprofit_vector(nonprofit_id: str) -> list[float] | None:
    """Fetch a nonprofit's embedding vector by ID."""
    response = (
        supabase.table("nonprofit_mission_vectors")
        .select("vector")
        .eq("nonprofit_id", nonprofit_id)
        .execute()
    )
    if response.data:
        vector = response.data[0]["vector"]
        if isinstance(vector, str):
            vector = json.loads(vector)
        return vector
    return None

def get_users(limit: int = 1000):
    """Fetch users from DB"""
    resp = supabase.table("users").select("*").limit(limit).execute()
    return resp.data

def get_engagement_types():
    """Fetch engagement types + weights"""
    rows = supabase.table("engagement_types").select("*").execute()
    return {row["engagement_type"]: row for row in rows}

def get_np_network_edges_by_id(nonprofit_id: list[int] = [1], top_k: int = 15) -> list[dict]:
    """Fetch network edges for a given nonprofit ID"""
    resp = supabase.table("network_graph_edges").select("*").in_("nonprofit_id_a", nonprofit_id).lte("metadata->prox_rank", top_k).execute()
    return resp.data

def fetch_node_attributes(node_ids: list[int]) -> dict[int, dict]:
    """Fetch NP attributes for given nonprofit IDs for network graph"""
    resp = (
        supabase.table("nonprofits")
        .select("id,name,mission,total_revenue,ntee_codes,logo_url,embedding")
        .in_("id", node_ids)
        .execute()
    )
    return {r["id"]: r for r in resp.data}
