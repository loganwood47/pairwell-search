import numpy as np
from ...db import store_user_vector, supabase
from services.embedding_service import embed_texts


def get_users_with_empty_vectors(batch_size=1000, offset=0):
    resp = (
        supabase.table("users_without_vectors")
        .select("id, interests")
        .range(offset, offset + batch_size - 1)
        .execute()
    )
    return resp.data


def create_missing_user_vectors(batch_size=500):
    offset = 0
    total_added = 0

    while True:
        users = get_users_with_empty_vectors(batch_size=batch_size, offset=offset)
        print(f"{len(users)} users fetched at offset {offset}")
        if not users:
            break

        print(f"Processing batch {offset // batch_size + 1} with {len(users)} users")

        # Collect all interests in one list
        interests = [str(u.get("interests", "")) for u in users]

        # Embed all at once
        vectors = embed_texts(interests)  # shape: (batch_size, dim)

        # Store results
        rows = [
            {"user_id": user["id"], "vector": vector.tolist()}
            for user, vector in zip(users, vectors)
        ]

        supabase.table("user_interest_vectors").insert(rows).execute()

        total_added += len(users)
        offset += batch_size

        print(f"Total vectors stored: {total_added}")

    print(f"Finished: {total_added} vectors created.")


create_missing_user_vectors(batch_size=500)
