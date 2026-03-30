#!/usr/bin/env python3
"""
MinerU VLM Batch Runner — processes PDFs in small batches to avoid CPU/RAM crash.

MinerU CLI has no built-in "process N files at a time" option — it scans the
entire input directory at once, which crashes Jupyter when given 5,000+ PDFs.

This script creates temporary symlink directories of BATCH_SIZE PDFs each,
calls `mineru` on each batch, and moves on. Already-parsed PDFs are skipped.

Usage (Jupyter cell):
    %run ~/cmip6_gpt/rag/run_mineru_batched.py

Or from terminal:
    python3 ~/cmip6_gpt/rag/run_mineru_batched.py
"""

import os
import subprocess
import shutil
import time
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
CORE_DIR = Path.home() / "core"
OUTPUT_DIR = Path.home() / "parsed_output"
BATCH_SIZE = 50           # PDFs per batch — tweak if still OOMs
BACKEND = "vlm-auto-engine"
GPUS = "0,1"              # which GPUs to use
DATA_PARALLEL = 2         # vLLM data-parallel size (= number of GPUs)
TEMP_BATCH_DIR = Path.home() / "_current_batch"
# Full path to mineru binary inside conda env (no activate needed)
MINERU_BIN = Path.home() / "miniforge3" / "envs" / "mineru" / "bin" / "mineru"

OUTPUT_DIR.mkdir(exist_ok=True)

# ── Collect all PDFs ──────────────────────────────────────────────────────────
all_pdfs = sorted(CORE_DIR.glob("*.pdf"))
print(f"Total PDFs in {CORE_DIR}: {len(all_pdfs)}")

# ── Filter out already-parsed PDFs ────────────────────────────────────────────
# MinerU creates a subdirectory per PDF in output_dir, named after the PDF stem
already_parsed = set()
if OUTPUT_DIR.exists():
    already_parsed = {d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()}

remaining = [p for p in all_pdfs if p.stem not in already_parsed]
print(f"Already parsed: {len(already_parsed)}")
print(f"Remaining: {len(remaining)}")

if not remaining:
    print("✅ All PDFs already parsed!")
    exit(0)

# ── Process in batches ────────────────────────────────────────────────────────
total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
print(f"\nWill process {len(remaining)} PDFs in {total_batches} batches of {BATCH_SIZE}")
print(f"Backend: {BACKEND} | GPUs: {GPUS} | Data-parallel: {DATA_PARALLEL}")
print("=" * 60)

t_start = time.time()

for batch_idx in range(total_batches):
    batch_start = batch_idx * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, len(remaining))
    batch_pdfs = remaining[batch_start:batch_end]

    # Create temp symlink directory for this batch
    if TEMP_BATCH_DIR.exists():
        shutil.rmtree(TEMP_BATCH_DIR)
    TEMP_BATCH_DIR.mkdir()

    for pdf in batch_pdfs:
        (TEMP_BATCH_DIR / pdf.name).symlink_to(pdf)

    print(f"\n[Batch {batch_idx + 1}/{total_batches}] "
          f"PDFs {batch_start + 1}-{batch_end} / {len(remaining)}")

    # Build mineru command
    cmd = (
        f"CUDA_VISIBLE_DEVICES={GPUS} {MINERU_BIN} "
        f"-p {TEMP_BATCH_DIR} "
        f"-o {OUTPUT_DIR} "
        f"-b {BACKEND} "
        f"-d cuda "
        f"--data-parallel-size {DATA_PARALLEL}"
    )

    t_batch = time.time()

    try:
        result = subprocess.run(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=3600  # 1 hour timeout per batch
        )

        elapsed = time.time() - t_batch
        parsed_now = len([d for d in OUTPUT_DIR.iterdir() if d.is_dir()])
        print(f"  ✅ Done in {elapsed:.0f}s | Total parsed: {parsed_now}/{len(all_pdfs)}")

        # Print last few lines of output for visibility
        lines = result.stdout.strip().split("\n")
        for line in lines[-5:]:
            print(f"  | {line}")

    except subprocess.TimeoutExpired:
        print(f"  ⚠️ Batch timed out after 1 hour, moving on...")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Cleanup temp dir
    if TEMP_BATCH_DIR.exists():
        shutil.rmtree(TEMP_BATCH_DIR)

# ── Final report ──────────────────────────────────────────────────────────────
total_time = time.time() - t_start
final_parsed = len([d for d in OUTPUT_DIR.iterdir() if d.is_dir()])
print(f"\n{'=' * 60}")
print(f"DONE!")
print(f"Total parsed: {final_parsed}/{len(all_pdfs)}")
print(f"Total time: {total_time / 3600:.1f} hours")
print(f"{'=' * 60}")
