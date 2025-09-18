from torch.utils.data import DataLoader, Dataset
import random
import torch
import numpy as np
from ...db import supabase



class TwoTowerDataset(Dataset):
    def __init__(self, neg_ratio: int = 4, limit: int = 5000):
        """
        :param neg_ratio: how many negatives per positive
        :param limit: max # interactions to pull from DB
        """
        self.neg_ratio = neg_ratio

        # 1. Pull positive interactions
        resp = (
            supabase.table("user_activity")
            .select("user_id, nonprofit_id, engagement_type_id")
            .limit(limit)
            .execute()
        )
        self.positives = resp.data

        # 2. Get engagement weights (importance)
        weights = supabase.table("engagement_types").select("*").execute().data
        self.engagement_weights = {row["id"]: row["weight"] for row in weights}

        # 3. Cache all nonprofits + embeddings
        nonprofits = supabase.table("nonprofits").select("id, embedding").execute().data
        self.nonprofits = {n["id"]: np.array(n["embedding"]) for n in nonprofits if n["embedding"]}

        # 4. Cache all users + embeddings
        users = supabase.table("users").select("id, embedding").execute().data
        self.users = {u["id"]: np.array(u["embedding"]) for u in users if u["embedding"]}

        # 5. Build training triples (u, n, label)
        self.samples = []
        for row in self.positives:
            uid, nid = row["user_id"], row["nonprofit_id"]

            if uid not in self.users or nid not in self.nonprofits:
                continue  # skip missing embeddings

            weight = self.engagement_weights.get(row["engagement_type_id"], 1.0)
            self.samples.append((uid, nid, weight))

            # add negatives
            for _ in range(self.neg_ratio):
                neg_nid = random.choice(list(self.nonprofits.keys()))
                while neg_nid == nid:  # don’t sample same nonprofit
                    neg_nid = random.choice(list(self.nonprofits.keys()))
                self.samples.append((uid, neg_nid, 0.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, nid, label = self.samples[idx]
        user_vec = self.users[uid].astype(np.float32)
        np_vec = self.nonprofits[nid].astype(np.float32)
        return (
            torch.tensor(user_vec),
            torch.tensor(np_vec),
            torch.tensor(label, dtype=torch.float32),
        )


dataset = TwoTowerDataset(neg_ratio=3, limit=10000)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for user_vecs, nonprofit_vecs, labels in loader:
    print(user_vecs.shape, nonprofit_vecs.shape, labels.shape)
    break
