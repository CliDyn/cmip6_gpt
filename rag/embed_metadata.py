#!/usr/bin/env python3
"""
Embed CMIP6 metadata (variables/sources/experiments) with gemini-embedding-2-preview.
Reads rewritten JSONL files, embeds text, outputs embedded JSONL.

Usage:
    python rag/embed_metadata.py
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

MODEL = "gemini-embedding-2-preview"
TASK_TYPE = "RETRIEVAL_DOCUMENT"
OUTPUT_DIM = 768
BATCH_SIZE = 100  # max texts per API call

INPUT_DIR = ROOT / "tests" / "original_json" / "rewritten" / "run_09f1d7bbf87b"
OUTPUT_DIR = ROOT / "rag"

FILES = {
    "variables": INPUT_DIR / "cmip6_variables_rewritten.jsonl",
    "sources": INPUT_DIR / "cmip6_sources_rewritten.jsonl",
    "experiments": INPUT_DIR / "cmip6_experiments_rewritten.jsonl",
}


def get_client():
    from google import genai
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def l2_normalize(vec):
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm > 0 else arr.tolist()


def embed_file(client, name: str, input_path: Path, output_path: Path):
    """Embed a single JSONL file."""
    from google.genai import types

    print(f"\n{'='*60}")
    print(f"Embedding {name}: {input_path.name}")

    with open(input_path) as f:
        records = [json.loads(line) for line in f]

    print(f"  Records: {len(records)}")

    embedded = []
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        texts = [r["rewritten"] for r in batch]

        result = client.models.embed_content(
            model=MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=TASK_TYPE,
                output_dimensionality=OUTPUT_DIM,
            ),
        )

        for rec, emb in zip(batch, result.embeddings):
            rec["embedding"] = l2_normalize(emb.values)
            embedded.append(rec)

        print(f"  [{len(embedded)}/{len(records)}] embedded")
        time.sleep(0.5)  # gentle rate limiting

    # Write output
    with open(output_path, "w") as f:
        for rec in embedded:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  → {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")
    return len(embedded)


def main():
    client = get_client()
    total = 0

    for name, input_path in FILES.items():
        if not input_path.exists():
            print(f"SKIP: {input_path} not found")
            continue
        output_path = OUTPUT_DIR / f"metadata_{name}_embedded.jsonl"
        total += embed_file(client, name, input_path, output_path)

    print(f"\nDone! {total} records embedded across {len(FILES)} files.")


if __name__ == "__main__":
    main()
