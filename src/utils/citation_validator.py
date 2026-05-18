"""
Citation Integrity Validator
=============================
Post-generation guardrail against DOI hallucination.

Background
----------
The audit of UC4 Q1 revealed that Gemini fabricated 7 out of 21 cited DOIs:
strings that follow the `10.xxxx/yyyy` pattern but never appeared in any
`cmip6_literature_search` result. This is the classic "parametric bleeding"
failure mode: the LLM, under a strict-citation system prompt, knows it MUST
emit DOIs, and when the RAG retrieval lacks an exact match for its desired
claim, it confabulates a plausible-looking DOI from its pre-training
weights.

The CMIP6 Forge backend already maintains a session blackboard of RAG-grounded
DOIs under `cite.<paper_id>` keys (see `literature_service.cmip6_literature_search`,
which auto-populates the blackboard from each search's top-10 hits). This
module reads that blackboard and validates that every DOI in the agent's
final text is grounded there.

Usage
-----
    from src.utils.citation_validator import validate_citations, format_violation

    result = validate_citations(final_text, blackboard)
    if not result.ok:
        # Return correction message to agent for regeneration
        correction = format_violation(result)
        # ... inject correction as SystemMessage and re-invoke agent

Design notes
------------
- Single source of truth = the session blackboard `cite.*` keys, NOT raw
  tool_result content. The blackboard is what the agent actually "saw" in
  its working memory and what the prompt explicitly told it to cite.
- DOI normalisation: lowercased, trailing punctuation stripped, no URL prefix.
- Tolerant of common formatting variants (parentheses, `( 10.xxxx )`, trailing
  periods, the `.1` vs `_1` AMS journal quirk).
- Conservative: only flags strings that match the DOI regex but are NOT in
  the retrieved set. Doesn't try to flag missing-citation cases (claims
  without any DOI) — that's a different audit dimension.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

# RFC-compatible DOI regex (Crossref-compatible). Matches the bulk of real-world
# DOIs including the AMS `jcli-d-XX-XXXX.1` and Springer `s00XXX-YYYY-NNNNN-N` forms.
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[a-zA-Z0-9._/()-]+")

# AMS journals publish DOIs as `10.1175/jcli-d-21-0468.1`. Some metadata systems
# (and occasionally Gemini's parametric memory) store them as `10.1175/jcli-d-21-0468_1`.
# Treat the two as identical when validating.
_AMS_VARIANT_RE = re.compile(r"_(\d+)$")


def _normalise_doi(raw: str) -> str:
    """Lowercase, strip trailing punctuation/whitespace, normalise AMS `_N` → `.N`."""
    s = raw.lower().strip().rstrip(".,);:]'\" ")
    # AMS variant: ..._1 → ....1 (last underscore followed by digits at end)
    s = _AMS_VARIANT_RE.sub(r".\1", s)
    return s


def _extract_dois(text: str) -> Set[str]:
    """Return the set of normalised DOIs found anywhere in the given text."""
    return {_normalise_doi(m) for m in DOI_PATTERN.findall(text or "")}


def _extract_blackboard_dois(blackboard: Optional[Dict[str, str]]) -> Set[str]:
    """Pull DOIs out of every `cite.*` blackboard entry.

    Each `cite.<paper_id>` value is a compact JSON like
    `{"doi":"10.xxxx/yyyy","title":"...", ...}`. We parse it and pull the DOI
    field. Falls back to a regex scan if the JSON is malformed.
    """
    grounded: Set[str] = set()
    if not blackboard:
        return grounded
    for key, value in blackboard.items():
        if not key.startswith("cite."):
            continue
        if not value:
            continue
        # Try structured parse first
        try:
            obj = json.loads(value) if isinstance(value, str) else value
            doi = obj.get("doi") if isinstance(obj, dict) else None
            if doi:
                grounded.add(_normalise_doi(str(doi)))
                continue
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback: regex scan the raw value
        grounded.update(_extract_dois(str(value)))
    return grounded


@dataclass
class CitationValidationResult:
    """Outcome of validating one block of generated text."""

    ok: bool
    cited: Set[str] = field(default_factory=set)
    grounded: Set[str] = field(default_factory=set)
    fabricated: Set[str] = field(default_factory=set)
    n_cited: int = 0
    n_fabricated: int = 0

    def summary(self) -> str:
        return (
            f"{self.n_cited} DOIs cited, "
            f"{self.n_fabricated} not in RAG retrieval"
            + (" — REJECT" if not self.ok else "")
        )


def validate_citations(
    text: str,
    blackboard: Optional[Dict[str, str]] = None,
    *,
    extra_grounded_dois: Optional[Iterable[str]] = None,
) -> CitationValidationResult:
    """Validate that every DOI in `text` is present in the RAG blackboard.

    Parameters
    ----------
    text
        The agent's final assistant response (or any text to validate).
    blackboard
        Per-session blackboard dict, typically `state.get("blackboard")` from
        the LangGraph agent state. Each `cite.<paper_id>` entry contains a
        compact JSON with the retrieved DOI.
    extra_grounded_dois
        Optional iterable of additional DOIs known to be grounded (e.g. from a
        cached prior turn, or explicitly whitelisted DOIs like the AR6 report).
        Use sparingly — defeats the purpose if filled with the model's own
        memory.

    Returns
    -------
    CitationValidationResult
        `.ok == True` iff every cited DOI is in the grounded set.
    """
    cited = _extract_dois(text)
    grounded = _extract_blackboard_dois(blackboard)
    if extra_grounded_dois:
        grounded.update(_normalise_doi(d) for d in extra_grounded_dois)

    fabricated = cited - grounded
    return CitationValidationResult(
        ok=len(fabricated) == 0,
        cited=cited,
        grounded=grounded,
        fabricated=fabricated,
        n_cited=len(cited),
        n_fabricated=len(fabricated),
    )


def format_violation(result: CitationValidationResult, *, max_listed: int = 10) -> str:
    """Render a correction message that can be injected back into the agent.

    Designed to be sent as a SystemMessage on the agent's NEXT turn so it
    regenerates the affected passages.
    """
    if result.ok:
        return ""
    listed = sorted(result.fabricated)[:max_listed]
    extra = (
        f"\n  …and {len(result.fabricated) - max_listed} more"
        if len(result.fabricated) > max_listed else ""
    )
    return (
        "[CITATION INTEGRITY VIOLATION]\n"
        f"Your previous response cited {result.n_cited} DOIs, of which "
        f"{result.n_fabricated} do NOT appear in this session's RAG retrieval "
        f"(the `cite.*` blackboard). These DOIs are HALLUCINATED:\n"
        + "\n".join(f"  - {d}" for d in listed)
        + extra
        + "\n\nRegenerate the response. For each fabricated DOI: either (a) drop "
          "the quantitative claim it backs, or (b) replace the citation with a "
          "DOI that actually appears in the RAG retrieval. Do not invent new "
          "DOIs from memory. If no retrieved paper supports a number, omit the "
          "number — coverage is not worth fabricated provenance."
    )


__all__ = [
    "CitationValidationResult",
    "DOI_PATTERN",
    "validate_citations",
    "format_violation",
]
