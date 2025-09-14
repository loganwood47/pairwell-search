"""
db.py
Handles database connection and queries to Supabase
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client
import numpy as np
import json

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_nonprofits(limit: int = 1000):
    """Fetch nonprofits from DB"""
    resp = supabase.table("nonprofits").select("*").limit(limit).execute()
    return resp.data

def get_nonprofits_by_id(limit: int = 1000, ids: list[int] = []):
    """Fetch nonprofits from DB"""
    resp = supabase.table("nonprofits").select("*").eq("id", ids).limit(limit).execute()
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


def get_user_vector(user_id: str) -> list[float] | None:
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
            vector = json.loads(vector)  # convert string → list
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
            vector = json.loads(vector)  # convert string → list
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
