"""
visualize.py
Helpers for embedding visualization in Streamlit
"""
from itertools import chain
import networkx as nx
from pyvis.network import Network
import streamlit as st
from textwrap import wrap
from sklearn.metrics.pairwise import cosine_similarity
import ast

from src.pairwell_search.services.db import get_np_network_edges_by_id, fetch_node_attributes

ntee_code_palette_map = {
    "A": "#1f77b4",  # Arts, Culture & Humanities
    "B": "#ff7f0e",  # Education
    "C": "#2ca02c",  # Environment
    "D": "#d62728",  # Animal Related
    "E": "#9467bd",  # Health Care
    "F": "#8c564b",  # Mental Health & Crisis
    "G": "#e377c2",  # Disease, Disorders & Medical
    "H": "#7f7f7f",  # Medical Research
    "I": "#bcbd22",  # Crime & Legal Related
    "J": "#17becf",  # Employment & Job Related
    "K": "#393b79",  # Food, Agriculture & Nutrition
    "L": "#637939",  # Housing & Shelter
    "M": "#8c6d31",  # Public Safety & Disaster Prep
    "N": "#843c39",  # Recreation & Sports
    "O": "#7b4173",  # Youth Development
    "P": "#3182bd",  # Human Services
    "Q": "#31a354",  # International, Foreign Affairs
    "R": "#756bb1",  # Civil Rights & Advocacy
    "S": "#636363",  # Community Improvement
    "T": "#e6550d",  # Philanthropy & Grantmaking
    "U": "#9ecae1",  # Science & Technology
    "V": "#74c476",  # Social Science & Research
    "W": "#fd8d3c",  # Public & Societal Benefit
    "X": "#969696",  # Religion Related
    "Y": "#6baed6",  # Mutual/Membership Benefit
    "Z": "#31c9b0",  # Unknown/Other
}

ntee_code_lookup = {
    "A": "Arts, Culture & Humanities",
    "B": "Education",
    "C": "Environment",
    "D": "Animal Related",
    "E": "Health Care",
    "F": "Mental Health & Crisis",
    "G": "Disease, Disorders & Medical",
    "H": "Medical Research",
    "I": "Crime & Legal Related",
    "J": "Employment & Job Related",
    "K": "Food, Agriculture & Nutrition",
    "L": "Housing & Shelter",
    "M": "Public Safety & Disaster Prep",
    "N": "Recreation & Sports",
    "O": "Youth Development",
    "P": "Human Services",
    "Q": "International, Foreign Affairs",
    "R": "Civil Rights & Advocacy",
    "S": "Community Improvement",
    "T": "Philanthropy & Grantmaking",
    "U": "Science & Technology",
    "V": "Social Science & Research",
    "W": "Public & Societal Benefit",
    "X": "Religion Related",
    "Y": "Mutual/Membership Benefit",
    "Z": "Unknown/Other"
}

def calc_sims(user_embedding, np_embedding):
    if user_embedding is not None and np_embedding is not None:
        try:
            if isinstance(np_embedding, str):
                import json
                np_embedding = [float(x) for x in ast.literal_eval(np_embedding)]
            sim = cosine_similarity(
                [user_embedding], 
                [np_embedding]
            )[0][0]
            return sim
        except Exception as e:
            print(f"Error computing similarity: {e}")

def show_color_legend():
    st.markdown("### NTEE Category Colors")
    cols = st.columns(2)
    items = list(ntee_code_palette_map.items())
    per_col = (len(items) + 3) // 4
    for c, start in zip(cols, range(0, len(items), per_col)):
        for code, color in items[start:start+per_col]:
            label = ntee_code_lookup.get(code)  # or add category names if you store them
            c.markdown(
                f"<div style='display:flex;align-items:center;margin-bottom:4px;'>"
                f"<div style='width:16px;height:16px;background:{color};"
                f"border-radius:3px;margin-right:6px;'></div>"
                f"{label}</div>",
                unsafe_allow_html=True,
            )

def compute_node_size(row, user_embedding=None, default_size=10, min_sim=0, max_sim=1):
    np_embedding = row.get("embedding")
    try:
        sim = calc_sims(user_embedding, np_embedding)
            # scale similarity [-1,1] to [10,40]
        sim = sim**4  # emphasize higher similarities
        print(sim)
        print(f"Size: {10 + ((sim - min_sim) * (30/(max_sim - min_sim))) if max_sim > min_sim else 10}")
    # return 10 + (max((sim - 0.4), 0) * 30) #TODO: get better scaling
        return 10 + ((sim - min_sim) * (30/(max_sim - min_sim))) if max_sim > min_sim else 10
    except Exception as e:
        print(f"Error computing similarity for node size: {e}")
        return default_size

def compute_node_color(row):
    codes = row.get("ntee_codes")
    if not codes: return "#888"
    main_code = codes[0]["ntee_code"][:1]  # first letter
    palette = ntee_code_palette_map
    # print(f"Using color {palette.get(main_code, '#888')} for NTEE {main_code}")
    return palette.get(main_code, "#888")

def fetch_edges_for_graph(seed_ids: list[int], top_k: int = 10) -> tuple[list[int], list[dict]]:
    all_edges = []
    all_nodes = set(seed_ids)
    for nid in seed_ids:
        edges = get_np_network_edges_by_id([nid], top_k)
        all_edges.extend(edges)
        all_nodes.update([e["nonprofit_id_b"] for e in edges])
    return list(all_nodes), all_edges

def build_graph(user_id: str, user_embedding: list[float], seed_ids: list[int], nodes_data: dict, edges: list[dict]):
    G = nx.Graph()
    
    # Add User node (distinct style)
    G.add_node(
        f"user_{user_id}",
        label=f"User {user_id}",
        size=50,            # big size
        color="#FF5733",    # distinct color
        group="user"
    )

    sims = []

    for nid, row in nodes_data.items():
        if user_embedding is not None and row.get("embedding") is not None:
            sim = calc_sims(user_embedding, row.get("embedding"))
            if sim is not None:
                sims.append(sim)

    min_sim = min(sims)**4 if sims else 0
    max_sim = max(sims)**4 if max(sims) > min_sim else min_sim + 1e-5
    
    # Add nonprofit nodes
    for nid, row in nodes_data.items():
        rev = row.get("total_revenue")
        rev_str = f"${rev:,.0f}" if isinstance(rev, (int, float)) else "N/A"
        mission = row.get("mission") or "No mission provided"
        row_color = compute_node_color(row)
        G.add_node(
            nid,
            label=row["name"][:40],   # shorter label
            title=f"{mission}<br>Revenue: {rev_str}",
            size=compute_node_size(row, user_embedding, min_sim=min_sim, max_sim=max_sim),
            color=row_color,
            group="nonprofit"
        )
        if nid in seed_ids:
            # Connect user to seed nonprofits
            G.add_edge(f"user_{user_id}", nid, weight=1.0)
    
    # Add nonprofit–nonprofit edges
    for e in edges:
        G.add_edge(
            e["nonprofit_id_a"],
            e["nonprofit_id_b"],
            weight=e["weight"]
        )
    return G

def show_graph(G):
    col_legend, col_graph = st.columns([1, 3])
    with col_graph:
        net = Network(height="750px", width="100%", bgcolor="#222", font_color="white") 
        net.from_nx(G) 
        for node in net.nodes:
            if not str(node["id"]).startswith("user_"):
                node["color"] = G.nodes[node["id"]].get("color", "#888")
        net.repulsion(node_distance=180, central_gravity=0.2, spring_length=150) 
        st.components.v1.html(net.generate_html(), height=800)
    with col_legend:
        show_color_legend()