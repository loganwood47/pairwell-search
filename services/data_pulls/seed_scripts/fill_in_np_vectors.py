import numpy as np
from ...db import store_user_vector, supabase
from services.embedding_service import embed_texts


def get_nps_with_empty_vectors(batch_size=1000, offset=0):
    resp = (
        supabase.table("nps_without_vectors")
        .select("id, mission")
        .range(offset, offset + batch_size - 1)
        .execute()
    )
    return resp.data


def create_missing_np_vectors(batch_size=500):
    offset = 0
    total_added = 0

    while True:
        users = get_nps_with_empty_vectors(batch_size=batch_size, offset=offset)
        print(f"{len(users)} NPs fetched at offset {offset}")
        if not users:
            break

        print(f"Processing batch {offset // batch_size + 1} with {len(users)} NPs")

        # Collect all interests in one list
        missions = [str(u.get("mission", "")) for u in users]

        # Embed all at once
        vectors = embed_texts(missions)  # shape: (batch_size, dim)

        # Store results
        rows = [
            {"nonprofit_id": user["id"], "vector": vector.tolist()}
            for user, vector in zip(users, vectors)
        ]

        supabase.table("nonprofit_mission_vectors").insert(rows).execute()

        total_added += len(users)
        offset += batch_size

        print(f"Total vectors stored: {total_added}")

    print(f"Finished: {total_added} vectors created.")


create_missing_np_vectors(batch_size=500)
