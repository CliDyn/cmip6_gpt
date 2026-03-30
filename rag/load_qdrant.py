#!/usr/bin/env python3
"""
load_qdrant.py — Load embedded CMIP6 chunks into Qdrant with Hybrid Search.

Creates a collection with:
  - "dense" named vector (768-dim, Cosine) from Gemini Embedding 2
  - "sparse" named vector (BM25 via FastEmbed) for keyword search
  - Payload indexes for year, journal, tier, chunk_type, paper_id

Usage:
  python load_qdrant.py                  # Load all chunks
  python load_qdrant.py --limit 1000     # Test with 1000 chunks
  python load_qdrant.py --recreate       # Drop and recreate collection
"""

import argparse
import json
import sys
import uuid
import time
from pathlib import Path

from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QDRANT_URL = "http://localhost:6333"
COLLECTION = "cmip6_papers"
DENSE_DIM = 768
INPUT_JSONL = Path(__file__).parent / "chunks_embedded.jsonl"
BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# BM25 Sparse Encoder
# ---------------------------------------------------------------------------
print("Loading BM25 sparse encoder (FastEmbed)...")
bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
print("  BM25 model loaded.")


def text_to_sparse(text: str) -> models.SparseVector:
    """Convert text to a BM25 sparse vector."""
    result = list(bm25_model.embed([text]))[0]
    return models.SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


# ---------------------------------------------------------------------------
# Collection Setup
# ---------------------------------------------------------------------------
def create_collection(client: QdrantClient, recreate: bool = False):
    """Create (or recreate) the Qdrant collection with named vectors."""
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION in collections:
        if recreate:
            print(f"Dropping existing collection '{COLLECTION}'...")
            client.delete_collection(COLLECTION)
        else:
            info = client.get_collection(COLLECTION)
            print(f"Collection '{COLLECTION}' exists: {info.points_count} points")
            return False  # already exists

    print(f"Creating collection '{COLLECTION}'...")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_DIM,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,  # BM25-style IDF weighting
            ),
        },
        # Scalar quantization for 75% RAM savings on dense vectors
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
            ),
        ),
    )
    print("  Collection created with Cosine + BM25 + Scalar Quantization.")

    # Create payload indexes
    print("Creating payload indexes...")
    for field, schema in [
        ("year", models.PayloadSchemaType.INTEGER),
        ("journal", models.PayloadSchemaType.KEYWORD),
        ("tier", models.PayloadSchemaType.KEYWORD),
        ("paper_id", models.PayloadSchemaType.KEYWORD),
        ("chunk_type", models.PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=schema,
        )
        print(f"  Indexed: {field} ({schema})")

    return True  # newly created


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data(client: QdrantClient, input_path: Path, limit: int | None = None):
    """Load chunks into Qdrant in batches."""
    # Check how many points already exist
    info = client.get_collection(COLLECTION)
    existing_count = info.points_count
    print(f"Existing points: {existing_count}")

    points_buffer = []
    total_loaded = 0
    skipped = 0
    t0 = time.time()

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            chunk = json.loads(line)

            # Skip if no embedding
            embedding = chunk.get("embedding")
            if not embedding:
                skipped += 1
                continue

            # Generate sparse BM25 vector from raw text
            text_raw = chunk.get("text_raw", chunk.get("text_with_prefix", ""))
            sparse_vec = text_to_sparse(text_raw)

            # Build point
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
            point = models.PointStruct(
                id=point_id,
                vector={
                    "dense": embedding,
                    "sparse": sparse_vec,
                },
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": chunk.get("paper_id", ""),
                    "title": chunk.get("title", ""),
                    "doi": chunk.get("doi", ""),
                    "year": chunk.get("year"),
                    "journal": chunk.get("journal", ""),
                    "tier": chunk.get("tier", ""),
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "section": chunk.get("section", ""),
                    "text_raw": text_raw[:2000],  # truncate for storage
                },
            )
            points_buffer.append(point)

            # Batch upsert
            if len(points_buffer) >= BATCH_SIZE:
                client.upsert(
                    collection_name=COLLECTION,
                    points=points_buffer,
                )
                total_loaded += len(points_buffer)
                elapsed = time.time() - t0
                rate = total_loaded / elapsed if elapsed > 0 else 0
                print(f"  [{total_loaded:,}] loaded ({rate:.0f} pts/sec)")
                points_buffer = []

    # Flush remaining
    if points_buffer:
        client.upsert(
            collection_name=COLLECTION,
            points=points_buffer,
        )
        total_loaded += len(points_buffer)

    elapsed = time.time() - t0
    print(f"\nDone! {total_loaded:,} points loaded in {elapsed:.1f}s")
    print(f"  Skipped (no embedding): {skipped}")

    # Final count
    info = client.get_collection(COLLECTION)
    print(f"  Collection total: {info.points_count:,} points")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Load CMIP6 chunks into Qdrant")
    parser.add_argument("--limit", type=int, default=None, help="Limit chunks to load")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection")
    parser.add_argument("--input", type=str, default=str(INPUT_JSONL), help="Input JSONL")
    parser.add_argument("--url", type=str, default=QDRANT_URL, help="Qdrant URL")
    args = parser.parse_args()

    client = QdrantClient(url=args.url)
    print(f"Connected to Qdrant at {args.url}")

    create_collection(client, recreate=args.recreate)
    load_data(client, Path(args.input), limit=args.limit)


if __name__ == "__main__":
    main()
