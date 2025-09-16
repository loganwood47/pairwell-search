import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from services.db import supabase
from services.embedding_service import embed_texts
import numpy as np

# Utilities
def encode_value(v, vocab):
    if v is None:
        return vocab["<PAD>"]
    return vocab.get(v, vocab["<UNK>"])

def encode_list(vs, vocab, max_len=5):
    """Encode multi-value fields as list of IDs, pad/truncate to fixed length."""
    if not vs:
        vs = []
    ids = [vocab.get(v, vocab["<UNK>"]) for v in vs]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids[:max_len]

def fetch_user_features(user_rows, city_vocab, state_vocab,
                        interests_vocab, prefs_vocab,
                        max_len=5):
    """
    Convert raw Supabase user rows → model-ready tensors.
    Returns a dict of tensors keyed by feature name.
    """
    user_ids = []
    incomes = []
    budgets = []
    city_ids = []
    state_ids = []
    interests_ids = []
    prefs_ids = []

    for r in user_rows:
        user_ids.append(r["id"])
        incomes.append(float(r["income"] or 0.0))
        budgets.append(float(r["donation_budget"] or 0.0))
        city_ids.append(encode_value(r["city"], city_vocab))
        state_ids.append(encode_value(r["state"], state_vocab))
        interests_ids.append(encode_list(r.get("interests", []), interests_vocab, max_len))
        prefs_ids.append(encode_list(r.get("engagement_prefs", []), prefs_vocab, max_len))

    return {
        "user_ids": torch.tensor(user_ids, dtype=torch.long),
        "income": torch.tensor(incomes, dtype=torch.float32),
        "budget": torch.tensor(budgets, dtype=torch.float32),
        "city_ids": torch.tensor(city_ids, dtype=torch.long),
        "state_ids": torch.tensor(state_ids, dtype=torch.long),
        "interests_ids": torch.tensor(interests_ids, dtype=torch.long),  # [N, max_len]
        "prefs_ids": torch.tensor(prefs_ids, dtype=torch.long),          # [N, max_len]
    }


def fetch_nonprofit_features(nonprofit_rows, mission_vocab, max_len=5):
    """
    Convert raw Supabase nonprofit rows → model-ready tensors.
    Returns a dict of tensors keyed by feature name.
    """
    nonprofit_ids = []
    revenues = []
    budgets = []  # we’ll use expenses as a proxy for “budget”
    mission_ids = []

    for r in nonprofit_rows:
        nonprofit_ids.append(r["id"])
        revenues.append(float(r["total_revenue"] or 0.0))
        budgets.append(float(r["total_expenses"] or 0.0))
        mission_ids.append(encode_value(r["mission"], mission_vocab))

    return {
        "nonprofit_ids": torch.tensor(nonprofit_ids, dtype=torch.long),
        "revenue": torch.tensor(revenues, dtype=torch.float32),
        "budget": torch.tensor(budgets, dtype=torch.float32),
        "mission_ids": torch.tensor(mission_ids, dtype=torch.long),
    }


# -----------------------------
# Dataset wrapper
# -----------------------------
class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, rows, user_features, nonprofit_features):
        super().__init__()
        self.rows = rows
        self.user_features = user_features
        self.nonprofit_features = nonprofit_features

        # Extract unique user and nonprofit IDs
        self.user_ids = list(set(row["user_id"] for row in rows))
        self.nonprofit_ids = list(set(row["nonprofit_id"] for row in rows))

        # Create a mapping from user ID to index
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.nonprofit_id_to_index = {nonprofit_id: idx for idx, nonprofit_id in enumerate(self.nonprofit_ids)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        u_id = self.user_id_to_index[int(row["user_id"])]
        n_id = self.nonprofit_id_to_index[int(row["nonprofit_id"])]
        label = float(row["label"])
        weight = float(row["weight"])

        u_feats = self.user_features.get(int(row["user_id"]), np.zeros(6, dtype=np.float32))  # Adjusted for new features
        n_feats = self.nonprofit_features.get(int(row["nonprofit_id"]), np.zeros(5, dtype=np.float32))

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
    print(f"Normalizing column with sample values: {values[:5]}")
    if type(values[0]) in [str, bytes]:
        print(f"Embedding text values: {values[0]}")
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
