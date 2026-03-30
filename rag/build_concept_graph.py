#!/usr/bin/env python3
"""
CMIP6 Concept Graph Builder — LightRAG Entity-Relationship Extraction.

Reads all paper chunks from Qdrant, feeds them to LightRAG for entity+relationship
extraction using Gemini 3 Flash, and persists the concept graph locally.

Usage:
  python rag/build_concept_graph.py                    # Full corpus
  python rag/build_concept_graph.py --limit 1000       # First 1000 chunks (test)
  python rag/build_concept_graph.py --resume            # Resume from last checkpoint

Cost estimate (full corpus):
  101,828 chunks × ~1,520 chars × ~380 tokens → ~39M input tokens
  At gemini-3-flash ($0.50/1M input): ~$19.3 input + ~$10 output ≈ $25-30 total
"""

import os
import sys
import json
import time
import asyncio
import warnings
import logging
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "cmip6_papers"
WORKING_DIR = Path(__file__).parent / "concept_graph_store"
CHECKPOINT_FILE = WORKING_DIR / "checkpoint.json"
BATCH_SIZE = 50  # chunks per LightRAG insert call
LLM_MODEL = "gemini-3-flash-preview"
EMBEDDING_MODEL = "models/gemini-embedding-2-preview"
EMBEDDING_DIM = 768


def get_api_key():
    """Load API key from environment or .env file."""
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("GOOGLE_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["GOOGLE_API_KEY"] = key
                    break
    if not key:
        raise ValueError("GOOGLE_API_KEY not set")
    os.environ["GEMINI_API_KEY"] = key
    return key


def load_chunks_from_qdrant(limit=None):
    """Load all text chunks from Qdrant."""
    from qdrant_client import QdrantClient

    log.info("Loading chunks from Qdrant...")
    qdrant = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    
    all_chunks = []
    offset = None
    while True:
        results, offset = qdrant.scroll(
            collection_name=COLLECTION,
            limit=100,
            offset=offset,
            with_payload=["text_raw", "paper_id", "title", "year", "chunk_type"],
        )
        for r in results:
            text = r.payload.get("text_raw", "")
            if text and len(text) > 50:  # skip tiny chunks
                all_chunks.append({
                    "id": r.id,
                    "text": text,
                    "paper_id": r.payload.get("paper_id", ""),
                    "title": r.payload.get("title", ""),
                    "year": r.payload.get("year", ""),
                })
        if offset is None:
            break
        if limit and len(all_chunks) >= limit:
            all_chunks = all_chunks[:limit]
            break

    log.info(f"Loaded {len(all_chunks)} chunks from Qdrant")
    return all_chunks


def create_lightrag_instance(api_key):
    """Create LightRAG with Gemini 3 Flash + custom embedding."""
    from lightrag import LightRAG
    from lightrag.llm.gemini import gemini_model_complete
    from lightrag.utils import wrap_embedding_func_with_attrs
    import google.genai as genai

    @wrap_embedding_func_with_attrs(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=2048,
        model_name="gemini-embedding-2-preview",
    )
    async def custom_gemini_embed(texts: list[str], **kwargs) -> np.ndarray:
        client = genai.Client(api_key=api_key)
        from google.genai import types
        response = await client.aio.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBEDDING_DIM,
            ),
        )
        embeddings = np.array(
            [np.array(e.values, dtype=np.float32) for e in response.embeddings]
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=gemini_model_complete,
        llm_model_name=LLM_MODEL,
        embedding_func=custom_gemini_embed,
        embedding_batch_num=10,
        embedding_func_max_async=4,
        llm_model_max_async=4,
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
    )
    return rag


def load_checkpoint():
    """Load processing checkpoint."""
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"processed_ids": [], "batch_idx": 0, "total_entities": 0, "total_relations": 0}


def save_checkpoint(state):
    """Save processing checkpoint."""
    CHECKPOINT_FILE.write_text(json.dumps(state, indent=2))


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="CMIP6 Concept Graph Builder")
    parser.add_argument("--limit", type=int, help="Limit number of chunks to process")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    print("=" * 70)
    print("CMIP6 Concept Graph Builder (LightRAG + Gemini 3 Flash)")
    print("=" * 70)

    # 1. Setup
    api_key = get_api_key()
    chunks = load_chunks_from_qdrant(limit=args.limit)

    # 2. Load or create checkpoint
    state = load_checkpoint() if args.resume else {
        "processed_ids": [], "batch_idx": 0,
        "total_entities": 0, "total_relations": 0,
    }
    processed_set = set(state["processed_ids"])

    # Filter already processed
    if args.resume:
        chunks = [c for c in chunks if str(c["id"]) not in processed_set]
        log.info(f"Resuming: {len(processed_set)} already done, {len(chunks)} remaining")

    if not chunks:
        log.info("Nothing to process!")
        return

    # 3. Create LightRAG
    rag = create_lightrag_instance(api_key)
    await rag.initialize_storages()

    # 4. Process in batches
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(chunks))
        batch = chunks[batch_start:batch_end]

        # Prepare documents: prefix with metadata for better extraction
        docs = []
        for c in batch:
            header = f"[Paper: {c['title']} ({c['year']}), DOI: {c['paper_id']}]\n"
            docs.append(header + c["text"])

        log.info(
            f"Batch {batch_idx + 1}/{total_batches} "
            f"({batch_start}-{batch_end}/{len(chunks)}) | "
            f"Elapsed: {time.time() - t0:.0f}s"
        )

        try:
            await rag.ainsert(docs)
        except Exception as e:
            log.error(f"Batch {batch_idx + 1} failed: {e}")
            # Save checkpoint and continue
            save_checkpoint(state)
            continue

        # Update checkpoint
        for c in batch:
            state["processed_ids"].append(str(c["id"]))
        state["batch_idx"] = batch_idx + 1

        # Save checkpoint every 5 batches
        if (batch_idx + 1) % 5 == 0:
            save_checkpoint(state)
            elapsed = time.time() - t0
            rate = (batch_idx + 1) * BATCH_SIZE / elapsed
            eta = (len(chunks) - batch_end) / rate if rate > 0 else 0
            log.info(
                f"  Checkpoint saved. "
                f"Rate: {rate:.0f} chunks/s | ETA: {eta / 60:.0f} min"
            )

    # 5. Final save
    save_checkpoint(state)
    elapsed = time.time() - t0

    # 6. Stats
    graph_file = WORKING_DIR / "graph_chunk_entity_relation.graphml"
    graph_size = graph_file.stat().st_size / 1e6 if graph_file.exists() else 0

    print(f"\n{'=' * 70}")
    print(f"CONCEPT GRAPH BUILD COMPLETE")
    print(f"{'=' * 70}")
    print(f"Chunks processed:  {len(state['processed_ids'])}")
    print(f"Graph file:        {graph_file} ({graph_size:.1f} MB)")
    print(f"Working dir:       {WORKING_DIR}")
    print(f"Total time:        {elapsed / 60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
