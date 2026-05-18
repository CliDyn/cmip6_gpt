"""Shared blackboard helpers — capacity enforcement, formatting.

Lives outside src/agents to avoid circular imports between the agent
factory and tools that auto-populate the blackboard (literature_service,
methodology_rag_service, etc.).
"""
from typing import Dict, Optional

from src.config import _CONFIG_DATA

_BB_CFG = _CONFIG_DATA.get("blackboard", {}) or {}
BB_ENABLED = bool(_BB_CFG.get("enabled", True))
BB_MAX_ENTRIES = int(_BB_CFG.get("max_entries", 30))
BB_MAX_TOTAL_CHARS = int(_BB_CFG.get("max_total_chars", 8000))
BB_MAX_VALUE_CHARS = int(_BB_CFG.get("max_value_chars", 500))


def merge_blackboard(left: Optional[dict], right: Optional[dict]) -> dict:
    """Reducer for the blackboard field.
    - Right wins on collision.
    - A right value of None means 'delete this key' (forget()).
    - Capacity is enforced inside the writer tools, not here.
    """
    out: Dict[str, str] = dict(left or {})
    for k, v in (right or {}).items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = v
    return out


def apply_capacity(current: Dict[str, str],
                   updates: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """Trim `updates` so the resulting blackboard fits the configured caps.
    - New keys are dropped silently if entry-count cap is hit.
    - New keys are dropped if total-chars cap is hit.
    - Existing-key updates are always allowed (keeps semantics of overwrite).
    - Deletes (None) are always allowed.
    """
    accepted: Dict[str, Optional[str]] = {}
    projected = dict(current)
    total_chars = sum(len(v) for v in projected.values())
    for k, v in updates.items():
        if v is None:
            accepted[k] = None
            if k in projected:
                total_chars -= len(projected[k])
                projected.pop(k)
            continue
        capped = v[:BB_MAX_VALUE_CHARS]
        delta_chars = len(capped) - len(projected.get(k, ""))
        new_total = total_chars + delta_chars
        is_new_key = k not in projected
        if is_new_key and len(projected) >= BB_MAX_ENTRIES:
            continue
        if new_total > BB_MAX_TOTAL_CHARS:
            continue
        accepted[k] = capped
        projected[k] = capped
        total_chars = new_total
    return accepted


def format_blackboard(blackboard: Dict[str, str]) -> str:
    """Render the blackboard as a compact text block for SystemMessage injection.
    Keys are grouped by category prefix (everything before the first '.')."""
    if not blackboard:
        return ""
    grouped: Dict[str, list] = {}
    for k, v in blackboard.items():
        cat, _, sub = k.partition(".")
        grouped.setdefault(cat or "misc", []).append((sub or k, v))
    lines = ["[BLACKBOARD — short-term session memory; entries persist across tool calls]"]
    for cat in sorted(grouped):
        lines.append(f"  {cat}:")
        for sub, v in grouped[cat]:
            lines.append(f"    - {sub}: {v}")
    lines.append("Use save_to_memory(category, key, value) to add. Use forget(key) to remove.")
    return "\n".join(lines)
