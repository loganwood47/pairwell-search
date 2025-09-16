import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from services.db import supabase

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_nonprofits, embedding_dim=128):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.nonprofit_emb = nn.Embedding(num_nonprofits, embedding_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_ids, nonprofit_ids):
        u = self.user_emb(user_ids)
        v = self.nonprofit_emb(nonprofit_ids)
        x = torch.cat([u, v], dim=-1)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)  # raw logits


def train_two_tower(train_data, num_users, num_nonprofits, embedding_dim=128, epochs=5, lr=1e-3):
    """
    train_data: list of tuples (user_id, nonprofit_id, label, weight)
        - user_id: int
        - nonprofit_id: int
        - label: binary 0/1
        - weight: float (from engagement_types.weight)
    """

    model = TwoTowerModel(num_users, num_nonprofits, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # BCEWithLogitsLoss with per-sample weights
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    for epoch in range(epochs):
        total_loss = 0.0

        # unpack training data for this epoch
        user_ids = torch.tensor([u for u, _, _, _ in train_data], dtype=torch.long)
        nonprofit_ids = torch.tensor([n for _, n, _, _ in train_data], dtype=torch.long)
        labels = torch.tensor([l for _, _, l, _ in train_data], dtype=torch.float32)
        weights = torch.tensor([w for _, _, _, w in train_data], dtype=torch.float32)

        logits = model(user_ids, nonprofit_ids)
        losses = loss_fn(logits, labels)
        weighted_loss = (losses * weights).mean()

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        total_loss = weighted_loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    return model

def label_from_engagement(engagement_type_id):
    # adjust mapping based on your engagement_types table
    if engagement_type_id in [1]:  # view
        return 0
    else:  # click, donate, volunteer etc.
        return 1

# Fetch interactions
# resp = supabase.table("user_activity").select("user_id, nonprofit_id, engagement_type_id").execute()
# rows = resp.data

# user_ids = [row["user_id"] for row in rows]
# nonprofit_ids = [row["nonprofit_id"] for row in rows]
# labels = [label_from_engagement(row["engagement_type_id"]) for row in rows]

num_users = supabase.table("users").select("id", count="exact").execute().count
num_nonprofits = supabase.table("nonprofits").select("id", count="exact").execute().count

resp = supabase.table("interaction_training").select("*").execute()
train_data = [
    (
        int(row["user_id"]),
        int(row["nonprofit_id"]),
        int(row["label"]),
        float(row["weight"]),
    )
    for row in resp.data
]

model = train_two_tower(
    train_data=train_data,
    num_users=num_users + 1,       # +1 since embeddings are 0-indexed
    num_nonprofits=num_nonprofits + 1,
    embedding_dim=128,
    epochs=10
)

torch.save(model.state_dict(), "models/two_tower.pth")
print("Model trained and saved as two_tower.pth")