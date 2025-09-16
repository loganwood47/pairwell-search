# models/train_two_tower_full.py
import os
import math
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from services.db import supabase
from services.embedding_service import embed_texts   # must exist and return np.ndarray
import time

# -------------------------
# Config (tweakable)
# -------------------------
EMBED_DIM = 64                  # final latent dim per tower
CAT_EMBED_DIM = 32              # embedding dim for city/state/interest/pop
TEXT_EMBED_DIM = None           # inferred from embed_texts output
MAX_INTERESTS = 6
MAX_POPULATIONS = 6
NUMERIC_USER_FIELDS = ["income", "donation_budget"]
NUMERIC_NONPROFIT_FIELDS = ["total_revenue", "employee_count"]
BATCH_SIZE = 512
EPOCHS = 6
LR = 1e-3
WRITE_BACK_BATCH = 500

# -------------------------
# Helpers: vocab, encoding
# -------------------------
def build_vocab_from_list(values: List[Any]):
    """Build shallow vocab mapping: token -> int (reserve 0 PAD, 1 UNK)"""
    counter = defaultdict(int)
    for v in values:
        if v is None:
            continue
        if isinstance(v, list):
            for t in v:
                counter[str(t)] += 1
        else:
            counter[str(v)] += 1
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok in sorted(k for k in counter.keys()):
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab

def encode_scalar_token(tok, vocab):
    if tok is None:
        return 0
    return vocab.get(str(tok), vocab["<UNK>"])

def encode_list_tokens(lst, vocab, max_len):
    if not lst:
        return [0] * max_len
    ids = [vocab.get(str(x), vocab["<UNK>"]) for x in lst]
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [0] * (max_len - len(ids))

# -------------------------
# Data fetch + preprocessing
# -------------------------
def fetch_all_users(batch_size=1000):
    offset = 0
    total_added = 0
    usersList = []
    while True:
        users = supabase.table("users").select("*").range(offset, offset + batch_size - 1).execute()
        # print(f"{len(users.data)} users fetched at offset {offset}")
        if not users.data:
            break
        usersList.extend(users.data)
        total_added += len(users.data)
        offset += batch_size
    return usersList

def fetch_all_nonprofits(batch_size=1000):
    offset = 0
    total_added = 0
    npsList = []
    while True:
        nps = supabase.table("nonprofits").select("*").range(offset, offset + batch_size - 1).execute()
        # print(f"{len(users.data)} users fetched at offset {offset}")
        if not nps.data:
            break
        npsList.extend(nps.data)
        total_added += len(nps.data)
        offset += batch_size
    return npsList

def fetch_all_interactions(batch_size=1000):
    offset = 0
    total_added = 0
    intsList = []
    while True:
        ints = supabase.table("interaction_training").select("*").range(offset, offset + batch_size - 1).execute()
        # print(f"{len(users.data)} users fetched at offset {offset}")
        if not ints.data:
            break
        intsList.extend(ints.data)
        total_added += len(ints.data)
        offset += batch_size
    return intsList

def fetch_all_from_supabase(batch_size=1000, offset=0):
    print("Fetching users, nonprofits, interactions from Supabase...")
    users = fetch_all_users(batch_size)
    nonprofits = fetch_all_nonprofits(batch_size)
    interactions = fetch_all_interactions(batch_size)
    print(f"Fetched: {len(users)} users, {len(nonprofits)} nonprofits, {len(interactions)} interactions")
    return users, nonprofits, interactions

def build_vocabs(users, nonprofits):
    print("Building vocabs for categorical features...")
    city_vals = [u.get("city") for u in users] + [n.get("city") for n in nonprofits]
    state_vals = [u.get("state") for u in users] + [n.get("state") for n in nonprofits]
    interests_vals = [i for u in users for i in (u.get("interests") or [])]
    prefs_vals = [p for u in users for p in (u.get("engagement_prefs") or [])]
    pop_vals = [p for n in nonprofits for p in (n.get("population_served_codes") or [])]
    mission_texts = [n.get("mission") or "" for n in nonprofits]

    city_vocab = build_vocab_from_list(city_vals)
    state_vocab = build_vocab_from_list(state_vals)
    interest_vocab = build_vocab_from_list(interests_vals)
    prefs_vocab = build_vocab_from_list(prefs_vals)
    population_vocab = build_vocab_from_list(pop_vals)

    vocabs = {
        "city": city_vocab,
        "state": state_vocab,
        "interest": interest_vocab,
        "prefs": prefs_vocab,
        "population": population_vocab,
        "mission_texts": mission_texts,
    }
    print("Vocab sizes:", {k: (len(v) if isinstance(v, dict) else len(v)) for k,v in vocabs.items()})
    return vocabs

def normalize_numeric_column(values: List[float]) -> np.ndarray:
    arr = np.array([0.0 if v is None else float(v) for v in values], dtype=np.float32)
    if arr.std() > 0:
        arr = (arr - arr.mean()) / (arr.std() + 1e-9)
    return arr

def precompute_mission_embeddings(nonprofits):
    """Use embed_texts to compute mission embeddings for all nonprofits (returns dict id->np.array)"""
    print("Precomputing mission text embeddings for nonprofits (this may take time)...")
    texts = [n.get("mission") or "" for n in nonprofits]
    # Batch embed in chunks to avoid huge memory usage
    EMB_BATCH = 256
    embeds = []
    for i in range(0, len(texts), EMB_BATCH):
        chunk = texts[i:i+EMB_BATCH]
        arr = embed_texts(chunk)   # must return np.ndarray shape (len(chunk), dim)
        embeds.append(arr)
    all_embs = np.vstack(embeds)
    global TEXT_EMBED_DIM
    TEXT_EMBED_DIM = all_embs.shape[1]
    id_to_emb = {int(n["id"]): all_embs[idx] for idx, n in enumerate(nonprofits)}
    print(f"Mission embeddings computed, dim={TEXT_EMBED_DIM}")
    return id_to_emb

def build_feature_tables(users, nonprofits, vocabs, mission_embs):
    """
    Returns:
      user_features: dict user_id -> dict of preprocessed features
      nonprofit_features: dict nonprofit_id -> dict ...
    """
    print("Building feature tables...")

    # USERS numeric
    incomes = normalize_numeric_column([u.get("income") for u in users])
    budgets = normalize_numeric_column([u.get("donation_budget") for u in users])

    user_features = {}
    for idx, u in enumerate(users):
        uid = int(u["id"])
        # numeric vector
        num = np.array([incomes[idx], budgets[idx]], dtype=np.float32)
        # categorical ids
        city_id = encode_scalar_token(u.get("city"), vocabs["city"])
        state_id = encode_scalar_token(u.get("state"), vocabs["state"])
        interest_ids = np.array(encode_list_tokens(u.get("interests") or [], vocabs["interest"], MAX_INTERESTS), dtype=np.int64)
        prefs_ids = np.array(encode_list_tokens(u.get("engagement_prefs") or [], vocabs["prefs"], MAX_INTERESTS), dtype=np.int64)
        user_features[uid] = {
            "num": num,
            "city": np.int64(city_id),
            "state": np.int64(state_id),
            "interests": interest_ids,
            "prefs": prefs_ids,
        }

    # NONPROFITS numeric
    revenues = normalize_numeric_column([n.get("total_revenue") for n in nonprofits])
    employees = normalize_numeric_column([n.get("employee_count") for n in nonprofits])

    nonprofit_features = {}
    for idx, n in enumerate(nonprofits):
        nid = int(n["id"])
        num = np.array([revenues[idx], employees[idx]], dtype=np.float32)
        city_id = encode_scalar_token(n.get("city"), vocabs["city"])
        state_id = encode_scalar_token(n.get("state"), vocabs["state"])
        pop_ids = np.array(encode_list_tokens(n.get("population_served_codes") or [], vocabs["population"], MAX_POPULATIONS), dtype=np.int64)
        mission_emb = mission_embs.get(nid, np.zeros(TEXT_EMBED_DIM or 384, dtype=np.float32))
        nonprofit_features[nid] = {
            "num": num,
            "city": np.int64(city_id),
            "state": np.int64(state_id),
            "populations": pop_ids,
            "mission_emb": mission_emb.astype(np.float32),
        }

    return user_features, nonprofit_features

# -------------------------
# Dataset / collate
# -------------------------
class TwoTowerDataset(Dataset):
    def __init__(self, interactions: List[dict], user_features: Dict[int, dict],
                 nonprofit_features: Dict[int, dict], valid_user_ids, valid_nonprofit_ids):
        # interactions: list of dicts from Supabase (user_id, nonprofit_id, label, weight)
        self.rows = [r for r in interactions if (int(r["user_id"]) in valid_user_ids and int(r["nonprofit_id"]) in valid_nonprofit_ids)]
        self.user_features = user_features
        self.nonprofit_features = nonprofit_features
        # Create index maps for compact embedding tables
        self.user_index = {uid: i for i, uid in enumerate(sorted(valid_user_ids))}
        self.nonprofit_index = {nid: i for i, nid in enumerate(sorted(valid_nonprofit_ids))}

        # reverse maps for writing back embeddings
        self.index_to_user = {i: uid for uid, i in self.user_index.items()}
        self.index_to_nonprofit = {i: nid for nid, i in self.nonprofit_index.items()}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        uid = int(r["user_id"]); nid = int(r["nonprofit_id"])
        uidx = self.user_index[uid]; nidx = self.nonprofit_index[nid]
        label = float(r.get("label", 0))
        weight = float(r.get("weight", 1.0))

        ufeat = self.user_features[uid]
        nfeat = self.nonprofit_features[nid]

        # pack tensors (nums + ids + lists + mission embedding)
        return {
            "user_idx": uidx,
            "nonprofit_idx": nidx,
            "user_num": torch.tensor(ufeat["num"], dtype=torch.float32),
            "user_city": torch.tensor(ufeat["city"], dtype=torch.long),
            "user_state": torch.tensor(ufeat["state"], dtype=torch.long),
            "user_interests": torch.tensor(ufeat["interests"], dtype=torch.long),  # shape (MAX_INTERESTS,)
            "user_prefs": torch.tensor(ufeat["prefs"], dtype=torch.long),
            "non_num": torch.tensor(nfeat["num"], dtype=torch.float32),
            "non_city": torch.tensor(nfeat["city"], dtype=torch.long),
            "non_state": torch.tensor(nfeat["state"], dtype=torch.long),
            "non_pops": torch.tensor(nfeat["populations"], dtype=torch.long),      # (MAX_POPULATIONS,)
            "mission_emb": torch.tensor(nfeat["mission_emb"], dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "weight": torch.tensor(weight, dtype=torch.float32),
        }

def collate_fn(batch):
    # batch: list of dicts
    out = {}
    # stack simple elements
    out["user_idx"] = torch.tensor([b["user_idx"] for b in batch], dtype=torch.long)
    out["nonprofit_idx"] = torch.tensor([b["nonprofit_idx"] for b in batch], dtype=torch.long)
    out["user_num"] = torch.stack([b["user_num"] for b in batch])
    out["user_city"] = torch.tensor([b["user_city"] for b in batch], dtype=torch.long)
    out["user_state"] = torch.tensor([b["user_state"] for b in batch], dtype=torch.long)
    out["user_interests"] = torch.stack([b["user_interests"] for b in batch])  # (B, MAX_INTERESTS)
    out["user_prefs"] = torch.stack([b["user_prefs"] for b in batch])
    out["non_num"] = torch.stack([b["non_num"] for b in batch])
    out["non_city"] = torch.tensor([b["non_city"] for b in batch], dtype=torch.long)
    out["non_state"] = torch.tensor([b["non_state"] for b in batch], dtype=torch.long)
    out["non_pops"] = torch.stack([b["non_pops"] for b in batch])
    out["mission_emb"] = torch.stack([b["mission_emb"] for b in batch])
    out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    out["weight"] = torch.tensor([b["weight"] for b in batch], dtype=torch.float32)
    return out

# -------------------------
# Model
# -------------------------
class TwoTower(nn.Module):
    def __init__(self,
                 n_users:int, n_nonprofits:int,
                 city_vocab_size:int, state_vocab_size:int,
                 interest_vocab_size:int, population_vocab_size:int,
                 text_emb_dim:int,
                 embed_dim:int = EMBED_DIM,
                 cat_emb_dim:int = CAT_EMBED_DIM):
        super().__init__()
        # ID embeddings
        self.user_id_emb = nn.Embedding(n_users, embed_dim)
        self.non_id_emb = nn.Embedding(n_nonprofits, embed_dim)
        # categorical embeddings
        self.city_emb = nn.Embedding(city_vocab_size, cat_emb_dim)
        self.state_emb = nn.Embedding(state_vocab_size, cat_emb_dim)
        self.interest_emb = nn.Embedding(interest_vocab_size, cat_emb_dim)
        self.population_emb = nn.Embedding(population_vocab_size, cat_emb_dim)
        # numeric projections
        self.user_num_proj = nn.Linear(len(NUMERIC_USER_FIELDS), embed_dim)
        self.non_num_proj = nn.Linear(len(NUMERIC_NONPROFIT_FIELDS), embed_dim)
        # text projection (mission)
        self.text_proj = nn.Linear(text_emb_dim, embed_dim)
        # final MLPs (project concat -> embed_dim)
        # We'll form concatenation: id_emb + num_proj + city + state + mean(interest) -> then MLP
        user_concat_dim = embed_dim + embed_dim + cat_emb_dim*2 + cat_emb_dim  # id + num + city + state + interests(mean)
        non_concat_dim = embed_dim + embed_dim + cat_emb_dim*2 + cat_emb_dim
        self.user_mlp = nn.Sequential(nn.Linear(user_concat_dim, 128), nn.ReLU(), nn.Linear(128, embed_dim))
        self.non_mlp = nn.Sequential(nn.Linear(non_concat_dim, 128), nn.ReLU(), nn.Linear(128, embed_dim))

    def forward(self, batch):
        # batch is dict from collate_fn
        u_id = batch["user_idx"]
        n_id = batch["nonprofit_idx"]

        # ID embeddings
        u_id_e = self.user_id_emb(u_id)            # (B, D)
        n_id_e = self.non_id_emb(n_id)

        # numeric
        u_num_e = self.user_num_proj(batch["user_num"])  # (B, D)
        n_num_e = self.non_num_proj(batch["non_num"])

        # categorical
        u_city_e = self.city_emb(batch["user_city"])
        u_state_e = self.state_emb(batch["user_state"])
        # interests: (B, L) -> (B, L, cat_dim) -> mean -> (B, cat_dim)
        u_interest_e = self.interest_emb(batch["user_interests"]).mean(dim=1)
        u_prefs_e = self.interest_emb(batch["user_prefs"]).mean(dim=1)  # reuse interest_emb for prefs (alternatively separate)

        n_city_e = self.city_emb(batch["non_city"])
        n_state_e = self.state_emb(batch["non_state"])
        n_pop_e = self.population_emb(batch["non_pops"]).mean(dim=1)

        # mission text projection
        mission_proj = self.text_proj(batch["mission_emb"])  # (B, D)

        # compose
        u_concat = torch.cat([u_id_e, u_num_e, u_city_e, u_state_e, u_interest_e], dim=1)
        n_concat = torch.cat([n_id_e, n_num_e, n_city_e, n_state_e, n_pop_e], dim=1)

        u_final = F.normalize(self.user_mlp(u_concat) + 0.0, dim=-1)
        n_final = F.normalize(self.non_mlp(n_concat) + mission_proj, dim=-1)  # add mission_proj into nonprofit side

        # similarity scores
        scores = (u_final * n_final).sum(dim=-1)
        return scores

    def encode_users(self, user_index_map, user_features, batch_size=1024, device="cpu"):
        """Return dict user_id -> embedding (numpy list)"""
        self.eval()
        id_to_emb = {}
        idxs = list(user_index_map.values())
        uids = list(user_index_map.keys())
        with torch.no_grad():
            for i in range(0, len(idxs), batch_size):
                batch_idxs = idxs[i:i+batch_size]
                # build batch tensors
                user_num = torch.stack([torch.tensor(user_features[uid]["num"]) for uid in uids[i:i+len(batch_idxs)]])
                user_city = torch.tensor([user_features[uid]["city"] for uid in uids[i:i+len(batch_idxs)]], dtype=torch.long)
                user_state = torch.tensor([user_features[uid]["state"] for uid in uids[i:i+len(batch_idxs)]], dtype=torch.long)
                user_interests = torch.stack([torch.tensor(user_features[uid]["interests"], dtype=torch.long) for uid in uids[i:i+len(batch_idxs)]])
                user_prefs = torch.stack([torch.tensor(user_features[uid]["prefs"], dtype=torch.long) for uid in uids[i:i+len(batch_idxs)]])
                batch_dict = {
                    "user_idx": torch.tensor(batch_idxs, dtype=torch.long),
                    "nonprofit_idx": torch.zeros(len(batch_idxs), dtype=torch.long),  # dummy
                    "user_num": user_num,
                    "user_city": user_city,
                    "user_state": user_state,
                    "user_interests": user_interests,
                    "user_prefs": user_prefs,
                    "non_num": torch.zeros((len(batch_idxs), len(NUMERIC_NONPROFIT_FIELDS)), dtype=torch.float32),
                    "non_city": torch.zeros(len(batch_idxs), dtype=torch.long),
                    "non_state": torch.zeros(len(batch_idxs), dtype=torch.long),
                    "non_pops": torch.zeros((len(batch_idxs), MAX_POPULATIONS), dtype=torch.long),
                    "mission_emb": torch.zeros((len(batch_idxs), TEXT_EMBED_DIM), dtype=torch.float32),
                }
                emb = self.user_mlp(torch.cat([self.user_id_emb(batch_dict["user_idx"]),
                                               self.user_num_proj(batch_dict["user_num"]),
                                               self.city_emb(batch_dict["user_city"]),
                                               self.state_emb(batch_dict["user_state"]),
                                               self.interest_emb(batch_dict["user_interests"]).mean(dim=1)], dim=1))
                emb = F.normalize(emb, dim=-1).cpu().numpy()
                for j, uid in enumerate(uids[i:i+len(batch_idxs)]):
                    id_to_emb[uid] = emb[j].tolist()
        return id_to_emb

    def encode_nonprofits(self, nonprofit_index_map, nonprofit_features, batch_size=1024, device="cpu"):
        self.eval()
        id_to_emb = {}
        n_ids = list(nonprofit_index_map.keys())
        idxs = list(nonprofit_index_map.values())
        with torch.no_grad():
            for i in range(0, len(idxs), batch_size):
                chunk_nids = n_ids[i:i+batch_size]
                # build batch tensors
                non_num = torch.stack([torch.tensor(nonprofit_features[nid]["num"]) for nid in chunk_nids])
                non_city = torch.tensor([nonprofit_features[nid]["city"] for nid in chunk_nids], dtype=torch.long)
                non_state = torch.tensor([nonprofit_features[nid]["state"] for nid in chunk_nids], dtype=torch.long)
                non_pops = torch.stack([torch.tensor(nonprofit_features[nid]["populations"], dtype=torch.long) for nid in chunk_nids])
                mission_emb = torch.stack([torch.tensor(nonprofit_features[nid]["mission_emb"], dtype=torch.float32) for nid in chunk_nids])

                # build partial batch
                batch_dict = {
                    "user_idx": torch.zeros(len(chunk_nids), dtype=torch.long),
                    "nonprofit_idx": torch.tensor([nonprofit_index_map[nid] for nid in chunk_nids], dtype=torch.long),
                    "user_num": torch.zeros((len(chunk_nids), len(NUMERIC_USER_FIELDS)), dtype=torch.float32),
                    "user_city": torch.zeros(len(chunk_nids), dtype=torch.long),
                    "user_state": torch.zeros(len(chunk_nids), dtype=torch.long),
                    "user_interests": torch.zeros((len(chunk_nids), MAX_INTERESTS), dtype=torch.long),
                    "user_prefs": torch.zeros((len(chunk_nids), MAX_INTERESTS), dtype=torch.long),
                    "non_num": non_num,
                    "non_city": non_city,
                    "non_state": non_state,
                    "non_pops": non_pops,
                    "mission_emb": mission_emb,
                }

                # compute embeddings: we add mission_proj onto non_mlp output inside forward; reuse pieces:
                with torch.no_grad():
                    n_id_vec = self.non_id_emb(batch_dict["nonprofit_idx"])
                    n_num_e = self.non_num_proj(batch_dict["non_num"])
                    n_city_e = self.city_emb(batch_dict["non_city"])
                    n_state_e = self.state_emb(batch_dict["non_state"])
                    n_pop_e = self.population_emb(batch_dict["non_pops"]).mean(dim=1)
                    n_concat = torch.cat([n_id_vec, n_num_e, n_city_e, n_state_e, n_pop_e], dim=1)
                    n_mlp_out = self.non_mlp(n_concat)
                    mission_proj = self.text_proj(batch_dict["mission_emb"])
                    final = F.normalize(n_mlp_out + mission_proj, dim=-1).cpu().numpy()
                    for j, nid in enumerate(chunk_nids):
                        id_to_emb[nid] = final[j].tolist()
        return id_to_emb

# -------------------------
# Training orchestration
# -------------------------
def train_and_write_back():
    users, nonprofits, interactions = fetch_all_from_supabase()

    # Build vocabs and mission embeddings
    vocabs = build_vocabs(users, nonprofits)
    mission_embs = precompute_mission_embeddings(nonprofits)

    # Build feature tables
    user_feats, nonprofit_feats = build_feature_tables(users, nonprofits, vocabs, mission_embs)

    # Align IDs: keep only rows where both have features
    user_ids_with_feats = set(user_feats.keys())
    nonprofit_ids_with_feats = set(nonprofit_feats.keys())
    interactions_filtered = [r for r in interactions if int(r["user_id"]) in user_ids_with_feats and int(r["nonprofit_id"]) in nonprofit_ids_with_feats]
    print(f"Filtered interactions: {len(interactions)} -> {len(interactions_filtered)} after dropping missing features")

    # Create dataset
    dataset = TwoTowerDataset(interactions_filtered, user_feats, nonprofit_feats, user_ids_with_feats, nonprofit_ids_with_feats)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # instantiate model
    n_users = len(dataset.user_index) + 1
    n_nonprofits = len(dataset.nonprofit_index) + 1
    model = TwoTower(
        n_users, n_nonprofits,
        city_vocab_size=len(vocabs["city"]),
        state_vocab_size=len(vocabs["state"]),
        interest_vocab_size=len(vocabs["interest"]),
        population_vocab_size=len(vocabs["population"]),
        text_emb_dim=TEXT_EMBED_DIM,
        embed_dim=EMBED_DIM,
        cat_emb_dim=CAT_EMBED_DIM
    )

    # train
    print("Starting training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        it = 0
        for batch in loader:
            optimizer.zero_grad()
            preds = model(batch)
            losses = loss_fn(preds, batch["label"])
            loss = (losses * batch["weight"]).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            it += 1
        print(f"Epoch {epoch+1}/{EPOCHS} avg_loss={(total_loss/it if it else 0.0):.4f}")

    # Save model weights
    ts = int(time.time())
    model_path = f"models/two_tower_trained_{ts}.pt"
    torch.save(model.state_dict(), model_path)
    print("Saved model to", model_path)

    # Compute learned embeddings and write back to Supabase in batches
    print("Computing user embeddings to write back...")
    user_id_to_emb = model.encode_users(dataset.user_index, user_feats, batch_size=1024)
    print("Computing nonprofit embeddings to write back...")
    non_id_to_emb = model.encode_nonprofits(dataset.nonprofit_index, nonprofit_feats, batch_size=1024)

    # Write back (batches)
    def batch_upsert(table, rows, batch_size=WRITE_BACK_BATCH):
        rows_list = list(rows)
        for i in range(0, len(rows_list), batch_size):
            chunk = rows_list[i:i+batch_size]
            try:
                resp = supabase.table(table).upsert(chunk).execute()
            except Exception as e:
                print("Upsert exception:", e)
                continue
        print(f"Wrote {len(rows_list)} rows to {table}")

    # Prepare payloads
    user_payloads = [{"id": uid, "embedding": emb} for uid, emb in user_id_to_emb.items()]
    nonprofit_payloads = [{"id": nid, "embedding": emb} for nid, emb in non_id_to_emb.items()]

    print("Writing user embeddings back to Supabase...")
    batch_upsert("users", user_payloads)
    print("Writing nonprofit embeddings back to Supabase...")
    batch_upsert("nonprofits", nonprofit_payloads)

    print("Done — model trained and embeddings written back.")

# -------------------------
# Entrypoint
# -------------------------
# if __name__ == "__main__":
train_and_write_back()
# test = fetch_all_interactions()
# print(test[:10])
