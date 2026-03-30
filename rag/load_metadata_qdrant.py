#!/usr/bin/env python3
"""
Load embedded CMIP6 metadata into Qdrant collections.

Creates 3 collections (cmip6_variables, cmip6_sources, cmip6_experiments)
with 768-dim dense vectors + metadata payloads.

Usage:
    python rag/load_metadata_qdrant.py
"""

import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType, OptimizersConfigDiff,
)

QDRANT_URL = "http://localhost:6333"
DIM = 768

RAG_DIR = Path(__file__).parent

COLLECTIONS = {
    "cmip6_variables": {
        "file": RAG_DIR / "metadata_variables_embedded.jsonl",
        "payload_keys": ["long_name", "realm", "units", "source"],
        "metadata_from": "metadata",
    },
    "cmip6_sources": {
        "file": RAG_DIR / "metadata_sources_embedded.jsonl",
        "payload_keys": ["institution_id", "release_year", "label_extended", "activity_participation"],
        "metadata_from": "metadata",
    },
    "cmip6_experiments": {
        "file": RAG_DIR / "metadata_experiments_embedded.jsonl",
        "payload_keys": ["activity_id", "tier", "experiment_name", "parent_experiment_id", "min_years"],
        "metadata_from": "metadata",
    },
}


def load_collection(client: QdrantClient, name: str, config: dict):
    filepath = config["file"]
    if not filepath.exists():
        print(f"SKIP: {filepath} not found")
        return 0

    # Read records
    with open(filepath) as f:
        records = [json.loads(line) for line in f]
    print(f"\n{'='*60}")
    print(f"Loading {name}: {len(records)} records")

    # Recreate collection
    if client.collection_exists(name):
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=DIM,
            distance=Distance.COSINE,
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=0,  # immediate indexing for small collections
        ),
    )

    # Build points
    points = []
    for idx, rec in enumerate(records):
        # Build payload
        payload = {
            "id": rec["id"],
            "text": rec["rewritten"],
            "original": rec.get("original", ""),
        }
        # Copy metadata fields
        meta = rec.get("metadata", {})
        for key in config["payload_keys"]:
            if key in meta:
                val = meta[key]
                # Flatten lists to strings for payload indexing
                if isinstance(val, list):
                    payload[key] = val
                else:
                    payload[key] = val

        # Also add the 'source' field (= id) for re-ranking compatibility
        payload["source"] = rec["id"]

        points.append(PointStruct(
            id=idx,
            vector=rec["embedding"],
            payload=payload,
        ))

    # Upsert in one batch (small enough)
    client.upsert(collection_name=name, points=points)

    # Create payload index on 'id' for fast filtering
    client.create_payload_index(
        collection_name=name,
        field_name="id",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    info = client.get_collection(name)
    print(f"  ✅ {info.points_count} points loaded")
    return info.points_count


def main():
    client = QdrantClient(url=QDRANT_URL)

    # Verify Qdrant is running
    try:
        collections = client.get_collections()
        print(f"Qdrant connected. Existing collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"ERROR: Cannot connect to Qdrant at {QDRANT_URL}: {e}")
        print("Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        return

    total = 0
    for name, config in COLLECTIONS.items():
        total += load_collection(client, name, config)

    print(f"\nDone! {total} total points across {len(COLLECTIONS)} collections.")

    # Summary
    print("\nCollections:")
    for c in client.get_collections().collections:
        info = client.get_collection(c.name)
        print(f"  {c.name}: {info.points_count} points")


if __name__ == "__main__":
    main()
