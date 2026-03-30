#!/usr/bin/env python3
"""
embed_and_index.py — Vectorize CMIP6 chunks with Gemini Embedding 2 (Vertex AI).

Two modes:
  1) realtime  — embed via streaming API (good for testing, ~5 req/s)
  2) batch     — submit to Batch API at 50% cost (production, ~24h turnaround)

Usage:
  # Test on 10 chunks (realtime)
  python embed_and_index.py --mode realtime --limit 10

  # Full production run (batch, 50% cheaper)
  python embed_and_index.py --mode batch

  # Check batch job status
  python embed_and_index.py --mode status --resume batches/123456789

Environment:
  GOOGLE_API_KEY in ../.env
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = "gemini-embedding-2-preview"  # Newest, natively multimodal, pre-normalized at 768
TASK_TYPE = "RETRIEVAL_DOCUMENT"
OUTPUT_DIM = 768  # MRL: 768 recommended for cost/quality balance
INPUT_JSONL = Path(__file__).parent / "chunks_enriched.jsonl"
OUTPUT_JSONL = Path(__file__).parent / "chunks_embedded.jsonl"
BATCH_INPUT_FILE = Path(__file__).parent / "batch_embed_input.jsonl"

# Realtime batching
REALTIME_BATCH_SIZE = 20  # texts per API call
REALTIME_SLEEP = 1.2       # ~50 RPM, safe for free tier


def get_client():
    """Create a Gemini API client with api_key."""
    from google import genai
    return genai.Client(api_key=os.environ['GOOGLE_API_KEY'])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector. Required for sub-3072 MRL dimensions."""
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def load_chunks(path: Path, limit: int | None = None) -> list[dict]:
    """Load chunks from JSONL."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            chunks.append(json.loads(line))
    return chunks


def estimate_cost(chunks: list[dict]) -> dict:
    """Estimate embedding cost."""
    total_tokens = sum(int(c.get("token_count", 0)) for c in chunks)
    cost_realtime = total_tokens / 1_000_000 * 0.25
    cost_batch = total_tokens / 1_000_000 * 0.125
    return {
        "chunks": len(chunks),
        "total_tokens": total_tokens,
        "cost_realtime_usd": round(cost_realtime, 2),
        "cost_batch_usd": round(cost_batch, 2),
    }


# ---------------------------------------------------------------------------
# Realtime Embedding
# ---------------------------------------------------------------------------
def embed_realtime(chunks: list[dict], output_path: Path):
    """Embed chunks using the realtime (streaming) API via Vertex AI."""
    from google.genai import types

    client = get_client()

    total = len(chunks)
    n_batches = math.ceil(total / REALTIME_BATCH_SIZE)
    embedded_count = 0

    # Check for existing progress (resume support)
    already_done = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                c = json.loads(line)
                already_done.add(c["chunk_id"])
        print(f"  Resuming: {len(already_done)} chunks already embedded.")

    with open(output_path, "a", encoding="utf-8") as fout:
        for batch_idx in range(n_batches):
            start = batch_idx * REALTIME_BATCH_SIZE
            end = min(start + REALTIME_BATCH_SIZE, total)
            batch_chunks = chunks[start:end]

            # Skip already-done chunks
            batch_chunks = [c for c in batch_chunks if c["chunk_id"] not in already_done]
            if not batch_chunks:
                continue

            texts = [c["text_with_prefix"] for c in batch_chunks]

            try:
                result = client.models.embed_content(
                    model=MODEL,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=TASK_TYPE,
                        output_dimensionality=OUTPUT_DIM,
                    ),
                )

                for chunk, emb in zip(batch_chunks, result.embeddings):
                    vec = l2_normalize(emb.values)
                    chunk["embedding"] = vec
                    fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    embedded_count += 1

                if embedded_count % 200 == 0 or batch_idx == n_batches - 1:
                    print(f"  [{embedded_count}/{total}] embedded")

            except Exception as e:
                # Exponential backoff for rate limits
                for retry in range(1, 4):
                    wait = 10 * (2 ** (retry - 1))  # 10s, 20s, 40s
                    print(f"  ERROR (attempt {retry}/3): {str(e)[:100]}", file=sys.stderr)
                    print(f"  Waiting {wait}s...", file=sys.stderr)
                    time.sleep(wait)
                    try:
                        result = client.models.embed_content(
                            model=MODEL,
                            contents=texts,
                            config=types.EmbedContentConfig(
                                task_type=TASK_TYPE,
                                output_dimensionality=OUTPUT_DIM,
                            ),
                        )
                        for chunk, emb in zip(batch_chunks, result.embeddings):
                            vec = l2_normalize(emb.values)
                            chunk["embedding"] = vec
                            fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                            embedded_count += 1
                        break  # success
                    except Exception as e2:
                        e = e2
                else:
                    print(f"  FATAL: skipping {len(batch_chunks)} chunks.", file=sys.stderr)

            time.sleep(REALTIME_SLEEP)

    print(f"\nDone! {embedded_count} chunks embedded → {output_path}")


# ---------------------------------------------------------------------------
# Batch API Embedding
# ---------------------------------------------------------------------------
def prepare_batch_input(chunks: list[dict], batch_input_path: Path):
    """Prepare JSONL input file for Batch API."""
    print(f"Preparing batch input file: {batch_input_path}")

    with open(batch_input_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            # Each line is an EmbedContentRequest
            request = {
                "key": chunk["chunk_id"],
                "request": {
                    "contents": [
                        {"parts": [{"text": chunk["text_with_prefix"]}]}
                    ],
                    "config": {
                        "task_type": TASK_TYPE,
                        "output_dimensionality": OUTPUT_DIM,
                    },
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    size_mb = batch_input_path.stat().st_size / (1024 * 1024)
    print(f"  Written {len(chunks)} requests ({size_mb:.1f} MB)")


def submit_batch_job(batch_input_path: Path) -> str:
    """Upload input file and submit batch embedding job."""
    client = get_client()

    # Upload the file
    print("Uploading batch input file...")
    uploaded = client.files.upload(
        file=str(batch_input_path),
        config={"display_name": "cmip6_embed_input", "mime_type": "jsonl"},
    )
    print(f"  Uploaded: {uploaded.name}")

    # Create batch job
    print("Submitting batch embedding job...")
    batch_job = client.batches.create_embeddings(
        model=MODEL,
        src={"file_name": uploaded.name},
        config={"display_name": "cmip6_rag_embeddings"},
    )
    print(f"  Job created: {batch_job.name}")
    print(f"  Status: {batch_job.state}")
    return batch_job.name


def check_batch_status(job_name: str):
    """Check status of a batch job."""
    client = get_client()
    job = client.batches.get(name=job_name)
    print(f"Job: {job.name}")
    print(f"  State: {job.state}")
    if hasattr(job, "batch_stats") and job.batch_stats:
        stats = job.batch_stats
        print(f"  Total requests: {getattr(stats, 'total_request_count', '?')}")
        print(f"  Succeeded: {getattr(stats, 'succeeded_request_count', '?')}")
        print(f"  Failed: {getattr(stats, 'failed_request_count', '?')}")
    return job


def download_batch_results(job_name: str, chunks: list[dict], output_path: Path):
    """Download batch results and merge with chunk metadata."""
    client = get_client()
    job = client.batches.get(name=job_name)

    if str(job.state) not in ("JOB_STATE_SUCCEEDED", "SUCCEEDED", "4"):
        print(f"Job not complete yet. State: {job.state}")
        return

    # Build chunk_id -> chunk lookup
    chunk_map = {c["chunk_id"]: c for c in chunks}

    # Download results
    print(f"Downloading results from {job.name}...")
    result_file_name = job.dest.file_name if hasattr(job, 'dest') and job.dest else None

    if result_file_name:
        # Download via Files API
        result_content = client.files.download(name=result_file_name)
        results = result_content.decode("utf-8").strip().split("\n")
    else:
        print("  No result file found. Trying inline results...")
        results = []

    embedded_count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for line in results:
            try:
                resp = json.loads(line)
                chunk_id = resp.get("custom_metadata", "")
                embedding_values = resp.get("embedding", {}).get("values", [])

                if chunk_id in chunk_map and embedding_values:
                    chunk = chunk_map[chunk_id]
                    chunk["embedding"] = l2_normalize(embedding_values)
                    fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    embedded_count += 1
            except json.JSONDecodeError:
                continue

    print(f"\nDone! {embedded_count}/{len(chunks)} chunks embedded → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Embed CMIP6 chunks with Gemini Embedding 2")
    parser.add_argument("--mode", choices=["realtime", "batch", "status", "download"],
                        default="realtime", help="Embedding mode")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of chunks (for testing)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Batch job name to resume/check (e.g. batches/123)")
    parser.add_argument("--input", type=str, default=str(INPUT_JSONL),
                        help="Input JSONL path")
    parser.add_argument("--output", type=str, default=str(OUTPUT_JSONL),
                        help="Output JSONL path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.mode in ("status", "download") and not args.resume:
        print("ERROR: --resume <job_name> required for status/download mode")
        sys.exit(1)

    # Status check
    if args.mode == "status":
        check_batch_status(args.resume)
        return

    # Load chunks
    print(f"Loading chunks from {input_path}...")
    chunks = load_chunks(input_path, args.limit)
    cost_info = estimate_cost(chunks)
    print(f"  Chunks: {cost_info['chunks']:,}")
    print(f"  Total tokens: {cost_info['total_tokens']:,}")
    print(f"  Est. cost (realtime): ${cost_info['cost_realtime_usd']}")
    print(f"  Est. cost (batch):    ${cost_info['cost_batch_usd']}")
    print()

    if args.mode == "realtime":
        print(f"=== REALTIME EMBEDDING (batch_size={REALTIME_BATCH_SIZE}) ===")
        embed_realtime(chunks, output_path)

    elif args.mode == "batch":
        if args.resume:
            print("=== CHECKING BATCH JOB ===")
            check_batch_status(args.resume)
        else:
            print("=== BATCH EMBEDDING (50% cost) ===")
            prepare_batch_input(chunks, BATCH_INPUT_FILE)
            job_name = submit_batch_job(BATCH_INPUT_FILE)
            print(f"\nJob submitted: {job_name}")
            print(f"Check status:   python embed_and_index.py --mode status --resume {job_name}")
            print(f"Download:       python embed_and_index.py --mode download --resume {job_name}")

    elif args.mode == "download":
        print("=== DOWNLOADING BATCH RESULTS ===")
        download_batch_results(args.resume, chunks, output_path)


if __name__ == "__main__":
    main()
