import torch
import torch.nn as nn
import torch.optim as optim
from services.data_pulls.two_tower.dataloader import loader, dataset

class TwoTowerModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, 64)
        )
        self.nonprofit_tower = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, 64)
        )

    def forward(self, user_vecs, nonprofit_vecs):
        u = self.user_tower(user_vecs)
        n = self.nonprofit_tower(nonprofit_vecs)
        sim = torch.sum(u * n, dim=1)  # dot product
        return sim


# training
model = TwoTowerModel(embed_dim=dataset[0][0].shape[0])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    for user_vecs, nonprofit_vecs, labels in loader:
        preds = model(user_vecs, nonprofit_vecs)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} Loss {loss.item():.4f}")
