#!/usr/bin/env python3
"""
Build a citation graph for the CMIP6 corpus using OpenAlex API.

Fetches `referenced_works` for all papers in Qdrant, filters to internal
edges (both papers in corpus), and saves a NetworkX DiGraph as JSON.

Usage:
    python build_citation_graph.py          # build graph
    python build_citation_graph.py --stats  # show graph statistics
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

# ── Config ────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "cmip6_papers"
METADATA_CSV = Path(__file__).parent.parent / "raw_papers" / "_metadata" / "all_cmip6_metadata.csv"
GRAPH_PATH = Path(__file__).parent / "citation_graph.json"
OPENALEX_EMAIL = "dmpantiu@awi.de"
OPENALEX_BATCH = 50  # IDs per API request (max ~50 with filter pipe)


def get_qdrant_papers():
    """Get unique paper_ids from Qdrant."""
    from qdrant_client import QdrantClient

    qdrant = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    papers = set()
    offset = None
    while True:
        results = qdrant.scroll(
            collection_name=COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=["paper_id", "title", "year", "journal"],
        )
        for p in results[0]:
            pid = p.payload.get("paper_id", "")
            if pid:
                papers.add(pid)
        offset = results[1]
        if offset is None:
            break
    return papers


def load_doi_to_openalex():
    """Build DOI → OpenAlex ID mapping from metadata CSV."""
    doi_to_oa = {}
    oa_to_meta = {}
    with open(METADATA_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            doi_raw = row.get("doi", "")
            oa_id = row.get("id", "")
            if doi_raw and oa_id:
                # Normalize DOI: https://doi.org/10.1234/abc → 10_1234_abc
                doi_norm = doi_raw.replace("https://doi.org/", "").replace("/", "_").replace(".", "_")
                oa_short = oa_id.replace("https://openalex.org/", "")
                doi_to_oa[doi_norm] = oa_short
                oa_to_meta[oa_short] = {
                    "doi": doi_norm,
                    "title": row.get("title", ""),
                    "year": int(row.get("publication_year", 0) or 0),
                    "journal": row.get("journal", ""),
                    "cited_by_count": int(row.get("cited_by_count", 0) or 0),
                }
    return doi_to_oa, oa_to_meta


def fetch_referenced_works(oa_ids: list[str]) -> dict[str, list[str]]:
    """Batch-fetch referenced_works from OpenAlex API."""
    result = {}
    batches = [oa_ids[i:i + OPENALEX_BATCH] for i in range(0, len(oa_ids), OPENALEX_BATCH)]

    for i, batch in enumerate(batches):
        pipe_ids = "|".join(batch)
        url = (
            f"https://api.openalex.org/works?"
            f"filter=openalex:{pipe_ids}&"
            f"select=id,referenced_works&"
            f"per_page=200&"
            f"mailto={OPENALEX_EMAIL}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": f"cmip6-rag/1.0 (mailto:{OPENALEX_EMAIL})"})

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                for work in data.get("results", []):
                    oa_short = work["id"].replace("https://openalex.org/", "")
                    refs = [r.replace("https://openalex.org/", "") for r in work.get("referenced_works", [])]
                    result[oa_short] = refs
        except Exception as e:
            print(f"  ⚠️ Batch {i + 1}/{len(batches)} failed: {e}")

        if (i + 1) % 10 == 0 or i == len(batches) - 1:
            print(f"  Fetched {i + 1}/{len(batches)} batches ({len(result)} papers)")

        # Rate limiting: OpenAlex allows 10 req/sec with mailto
        time.sleep(0.12)

    return result


def build_graph():
    """Main: build citation graph and save."""
    print("=== CMIP6 Citation Graph Builder ===\n")

    # 1. Get our papers from Qdrant
    print("1. Loading paper IDs from Qdrant...")
    qdrant_papers = get_qdrant_papers()
    print(f"   {len(qdrant_papers)} unique papers in Qdrant\n")

    # 2. Map DOI → OpenAlex ID
    print("2. Loading DOI → OpenAlex mapping...")
    doi_to_oa, oa_to_meta = load_doi_to_openalex()
    print(f"   {len(doi_to_oa)} DOIs mapped to OpenAlex IDs\n")

    # 3. Find OpenAlex IDs for our Qdrant papers
    our_oa_ids = []
    our_doi_to_oa = {}
    missing = 0
    for doi in qdrant_papers:
        oa_id = doi_to_oa.get(doi)
        if oa_id:
            our_oa_ids.append(oa_id)
            our_doi_to_oa[doi] = oa_id
        else:
            missing += 1

    print(f"3. Matched {len(our_oa_ids)}/{len(qdrant_papers)} papers to OpenAlex")
    if missing:
        print(f"   ({missing} papers not found in OpenAlex CSV)")
    print()

    # Build reverse map: OpenAlex ID → DOI (for our papers only)
    oa_to_doi = {v: k for k, v in our_doi_to_oa.items()}
    our_oa_set = set(our_oa_ids)

    # 4. Fetch referenced_works from OpenAlex
    print(f"4. Fetching referenced_works from OpenAlex API ({len(our_oa_ids)} papers)...")
    refs_map = fetch_referenced_works(our_oa_ids)
    print(f"   Got references for {len(refs_map)} papers\n")

    # 5. Build NetworkX DiGraph
    print("5. Building citation graph...")
    G = nx.DiGraph()

    # Add nodes (our papers)
    for doi in qdrant_papers:
        oa_id = our_doi_to_oa.get(doi)
        meta = oa_to_meta.get(oa_id, {}) if oa_id else {}
        G.add_node(doi, **{
            "title": meta.get("title", ""),
            "year": meta.get("year", 0),
            "journal": meta.get("journal", ""),
            "cited_by_count": meta.get("cited_by_count", 0),
        })

    # Add edges (A cites B, only if both in corpus)
    edge_count = 0
    external_refs = 0
    for doi, oa_id in our_doi_to_oa.items():
        refs = refs_map.get(oa_id, [])
        for ref_oa in refs:
            ref_doi = oa_to_doi.get(ref_oa)
            if ref_doi and ref_doi in qdrant_papers:
                G.add_edge(doi, ref_oa_to_doi := ref_doi)  # doi CITES ref_doi
                edge_count += 1
            else:
                external_refs += 1

    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Internal edges (CITES): {G.number_of_edges()}")
    print(f"   External references (skipped): {external_refs}")
    print(f"   Avg citations per paper (internal): {G.number_of_edges() / max(G.number_of_nodes(), 1):.1f}")
    print()

    # 6. Save
    print(f"6. Saving to {GRAPH_PATH}...")
    graph_data = json_graph.node_link_data(G)
    with open(GRAPH_PATH, "w") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    size_mb = GRAPH_PATH.stat().st_size / 1024 / 1024
    print(f"   Saved ({size_mb:.1f} MB)\n")

    # 7. Stats
    print_stats(G)

    return G


def print_stats(G=None):
    """Print citation graph statistics."""
    if G is None:
        if not GRAPH_PATH.exists():
            print("No graph found. Run without --stats first.")
            return
        with open(GRAPH_PATH) as f:
            data = json.load(f)
        G = json_graph.node_link_graph(data)

    print("=== Citation Graph Stats ===\n")
    print(f"Nodes (papers): {G.number_of_nodes()}")
    print(f"Edges (CITES):  {G.number_of_edges()}")
    print(f"Density:        {nx.density(G):.6f}")
    print()

    # Most cited within corpus (in-degree)
    in_deg = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)
    print("Top-15 most cited within corpus:")
    for doi, deg in in_deg[:15]:
        title = G.nodes[doi].get("title", "")[:55]
        year = G.nodes[doi].get("year", "?")
        print(f"  {deg:4d} citations | {title} ({year})")

    print()

    # Most citing (out-degree)
    out_deg = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)
    print("Top-5 papers citing most corpus papers:")
    for doi, deg in out_deg[:5]:
        title = G.nodes[doi].get("title", "")[:55]
        year = G.nodes[doi].get("year", "?")
        print(f"  {deg:4d} refs     | {title} ({year})")

    print()

    # Isolated nodes
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    print(f"Isolated papers (no internal citations): {len(isolated)}")

    # Components
    weakly_connected = list(nx.weakly_connected_components(G))
    print(f"Weakly connected components: {len(weakly_connected)}")
    largest = max(weakly_connected, key=len)
    print(f"Largest component: {len(largest)} papers ({100 * len(largest) / G.number_of_nodes():.1f}%)")

    # Year distribution
    years = Counter(G.nodes[n].get("year", 0) for n in G.nodes())
    print(f"\nPapers by year (top-5):")
    for y, c in years.most_common(5):
        print(f"  {y}: {c}")


def main():
    parser = argparse.ArgumentParser(description="CMIP6 Citation Graph Builder")
    parser.add_argument("--stats", action="store_true", help="Show stats for existing graph")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    else:
        build_graph()


if __name__ == "__main__":
    main()
