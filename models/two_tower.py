import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from services.db import supabase
from services.embedding_service import embed_texts
import numpy as np

# -----------------------------
# Dataset wrapper
# -----------------------------
class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, rows, user_features, nonprofit_features):
        """
        rows: list of dicts from interaction_training table
        user_features: dict[user_id] -> np.array
        nonprofit_features: dict[nonprofit_id] -> np.array
        """
        self.rows = rows
        self.user_features = user_features
        self.nonprofit_features = nonprofit_features

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        u_id = int(row["user_id"])
        n_id = int(row["nonprofit_id"])
        label = float(row["label"])
        weight = float(row["weight"])

        u_feats = self.user_features.get(u_id, np.zeros(2, dtype=np.float32))
        n_feats = self.nonprofit_features.get(n_id, np.zeros(4, dtype=np.float32))

        return (
            torch.tensor(u_id, dtype=torch.long),
            torch.tensor(n_id, dtype=torch.long),
            torch.tensor(u_feats, dtype=torch.float32),
            torch.tensor(n_feats, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )

# -----------------------------
# Two-Tower Model
# -----------------------------
class TwoTowerModel(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_nonprofits: int,
                 user_embed_dim: int = 32,
                 nonprofit_embed_dim: int = 32,
                 user_side_dim: int = 0,
                 nonprofit_side_dim: int = 0):
        super().__init__()

        # ID embeddings
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        self.nonprofit_embedding = nn.Embedding(num_nonprofits, nonprofit_embed_dim)

        # Side features
        self.user_side = nn.Linear(user_side_dim, user_embed_dim) if user_side_dim > 0 else None
        self.nonprofit_side = nn.Linear(nonprofit_side_dim, nonprofit_embed_dim) if nonprofit_side_dim > 0 else None

    def forward(self, user_ids, nonprofit_ids, user_feats=None, nonprofit_feats=None):
        # ID embeddings
        u_emb = self.user_embedding(user_ids)
        n_emb = self.nonprofit_embedding(nonprofit_ids)

        # Side features
        if self.user_side is not None and user_feats is not None:
            u_emb = u_emb + self.user_side(user_feats)

        if self.nonprofit_side is not None and nonprofit_feats is not None:
            n_emb = n_emb + self.nonprofit_side(nonprofit_feats)

        # Normalize
        u_emb = F.normalize(u_emb, dim=-1)
        n_emb = F.normalize(n_emb, dim=-1)

        # Similarity
        scores = (u_emb * n_emb).sum(dim=-1)
        return scores

# -----------------------------
# Training loop
# -----------------------------
def train_two_tower(train_loader, num_users, num_nonprofits,
                    embedding_dim=128, epochs=5, lr=1e-3,
                    user_side_dim=0, nonprofit_side_dim=0):

    model = TwoTowerModel(
        num_users=num_users,
        num_nonprofits=num_nonprofits,
        user_embed_dim=embedding_dim,
        nonprofit_embed_dim=embedding_dim,
        user_side_dim=user_side_dim,
        nonprofit_side_dim=nonprofit_side_dim,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            user_ids, nonprofit_ids, user_feats, nonprofit_feats, labels, weights = batch

            logits = model(user_ids, nonprofit_ids, user_feats, nonprofit_feats)
            losses = loss_fn(logits, labels)
            weighted_loss = (losses * weights).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            total_loss += weighted_loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return model

# -----------------------------
# Preprocessing
# -----------------------------
def normalize_column(values):
    """Convert list of values to normalized numpy array"""
    if type(values[0]) in [str, bytes]:
        # TODO: fix this, gives error
        values = embed_texts([str(v) for v in values])
    arr = np.array(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    if arr.std() > 0:
        arr = (arr - arr.mean()) / arr.std()
    return arr

def fetch_user_features():
    resp = supabase.table("users").select("id,income,donation_budget,city,state,interests,engagement_prefs").execute()
    rows = resp.data
    incomes = normalize_column([r["income"] or 0 for r in rows])
    budgets = normalize_column([r["donation_budget"] or 0 for r in rows])
    cities = normalize_column([r["city"] or 0 for r in rows])
    states = normalize_column([r["state"] or 0 for r in rows])
    interests = normalize_column([r["interests"] or 0 for r in rows])
    engagement_prefs = normalize_column([r["engagement_prefs"] or 0 for r in rows])

    features = {}
    for r, inc, bud, cit, st, intr, eng in zip(rows, incomes, budgets, cities, states, interests, engagement_prefs):
        features[int(r["id"])] = np.array([inc, bud, cit, st, intr, eng], dtype=np.float32)
    return features

def fetch_nonprofit_features():
    resp = supabase.table("nonprofits").select("id,employee_count,total_revenue,city,state,mission").execute()
    rows = resp.data
    emp = normalize_column([r["employee_count"] or 0 for r in rows])
    rev = normalize_column([r["total_revenue"] or 0 for r in rows])
    city = normalize_column([r["city"] or 0 for r in rows])
    state = normalize_column([r["state"] or 0 for r in rows])
    mission = normalize_column([r["mission"] or 0 for r in rows])

    features = {}
    for r, e, re, cit, stt, miss in zip(rows, emp, rev, city, state, mission):
        features[int(r["id"])] = np.array([e, re, cit, stt, miss], dtype=np.float32)
    return features

# -----------------------------
# Main
# -----------------------------

# Count users/nonprofits
num_users = supabase.table("users").select("id", count="exact").execute().count
num_nonprofits = supabase.table("nonprofits").select("id", count="exact").execute().count

# Fetch features
user_features = fetch_user_features()
nonprofit_features = fetch_nonprofit_features()

# Fetch training rows
resp = supabase.table("interaction_training").select("*").execute()
rows = resp.data

dataset = InteractionDataset(rows, user_features, nonprofit_features)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Train model
model = train_two_tower(
    loader,
    num_users=num_users + 1,
    num_nonprofits=num_nonprofits + 1,
    embedding_dim=128,
    epochs=10,
    user_side_dim=2,      # income + budget
    nonprofit_side_dim=4  # emp, revenue, lat, lon
)

torch.save(model.state_dict(), "models/two_tower_with_side.pth")
print("✅ Model trained and saved as models/two_tower_with_side.pth")
