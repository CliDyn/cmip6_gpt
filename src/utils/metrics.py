"""
Structured logging and metrics collection for the CMIP6 GPT pipeline.

Provides:
- PipelineMetrics: collects per-query metrics (timing, RAG scores, validation hits/misses)
- step_timer: context manager for timing pipeline steps
- pipeline_logger: structured logger for consistent formatting

Usage:
    metrics = PipelineMetrics()
    with metrics.step("select_facets"):
        result = select_facets(query)
    metrics.record_rag_scores("variable_id", scores=[0.2, 0.4, 0.8])
    metrics.record_validation("variable_id", requested="tas", accepted=True)
    report = metrics.summary()
"""

import time
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

# ─── Structured Logger ────────────────────────────────────────────────
# Replaces raw print() calls with leveled, prefixed logging

_logger = logging.getLogger("cmip6_pipeline")

if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-7s │ %(name)s │ %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    _logger.addHandler(_handler)
    _logger.setLevel(logging.DEBUG)

pipeline_logger = _logger


# ─── Step Timer ───────────────────────────────────────────────────────

@contextmanager
def step_timer(step_name: str, metrics: Optional["PipelineMetrics"] = None):
    """Context manager that logs and optionally records step duration."""
    pipeline_logger.info(f"▶ {step_name}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        pipeline_logger.info(f"◀ {step_name} ({elapsed:.3f}s)")
        if metrics is not None:
            metrics._step_times[step_name] = elapsed


# ─── RAG Score Entry ──────────────────────────────────────────────────

@dataclass
class RAGScoreEntry:
    """Stores score statistics for one facet's RAG retrieval."""
    facet: str
    total_candidates: int = 0
    accepted_candidates: int = 0
    rejected_candidates: int = 0
    scores: List[float] = field(default_factory=list)
    score_threshold: float = 0.0

    @property
    def min_score(self) -> Optional[float]:
        return min(self.scores) if self.scores else None

    @property
    def max_score(self) -> Optional[float]:
        return max(self.scores) if self.scores else None

    @property
    def mean_score(self) -> Optional[float]:
        return sum(self.scores) / len(self.scores) if self.scores else None


# ─── Validation Entry ────────────────────────────────────────────────

@dataclass
class ValidationEntry:
    """Records whether a selected facet value was in the dynamic schema."""
    facet: str
    value: Any
    accepted: bool
    corrected_value: Optional[Any] = None


# ─── Pipeline Metrics ────────────────────────────────────────────────

@dataclass
class PipelineMetrics:
    """
    Collects structured metrics for a single query through the CMIP6 pipeline.

    Designed for both real-time debugging and post-hoc benchmark analysis.
    Call .summary() to get a JSON-serializable dict of all collected data.
    """
    query: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Internal storage
    _step_times: Dict[str, float] = field(default_factory=dict)
    _rag_scores: Dict[str, RAGScoreEntry] = field(default_factory=dict)
    _validations: List[ValidationEntry] = field(default_factory=list)
    _facets_selected: List[str] = field(default_factory=list)
    _facet_values: Dict[str, Any] = field(default_factory=dict)
    _total_datasets: int = 0
    _api_success: bool = False
    _errors: List[str] = field(default_factory=list)

    # ── Recording methods ──

    def set_query(self, query: str):
        self.query = query

    def record_facets_selected(self, facets: List[str]):
        self._facets_selected = facets
        pipeline_logger.debug(f"Facets selected: {facets}")

    def record_rag_scores(
        self,
        facet: str,
        scores: List[float],
        total: int,
        accepted: int,
        threshold: float,
    ):
        entry = RAGScoreEntry(
            facet=facet,
            total_candidates=total,
            accepted_candidates=accepted,
            rejected_candidates=total - accepted,
            scores=scores,
            score_threshold=threshold,
        )
        self._rag_scores[facet] = entry
        pipeline_logger.debug(
            f"RAG [{facet}]: {accepted}/{total} accepted "
            f"(threshold={threshold:.2f}, "
            f"range={entry.min_score:.4f}–{entry.max_score:.4f})"
            if scores else f"RAG [{facet}]: no scores"
        )

    def record_validation(
        self, facet: str, value: Any, accepted: bool, corrected_value: Any = None
    ):
        entry = ValidationEntry(
            facet=facet, value=value, accepted=accepted, corrected_value=corrected_value
        )
        self._validations.append(entry)
        if accepted:
            pipeline_logger.debug(f"Validation ✓ {facet}={value}")
        else:
            pipeline_logger.warning(
                f"Validation ✗ {facet}={value}"
                + (f" → corrected to {corrected_value}" if corrected_value else " → rejected")
            )

    def record_facet_values(self, facet_values: Dict[str, Any]):
        self._facet_values = facet_values

    def record_api_result(self, total_datasets: int):
        self._total_datasets = total_datasets
        self._api_success = total_datasets > 0

    def record_error(self, error: str):
        self._errors.append(error)
        pipeline_logger.error(f"Pipeline error: {error}")

    @contextmanager
    def step(self, step_name: str):
        """Context manager shortcut for timing a step."""
        with step_timer(step_name, metrics=self):
            yield

    # ── Reporting ──

    @property
    def validation_accuracy(self) -> Optional[float]:
        if not self._validations:
            return None
        accepted = sum(1 for v in self._validations if v.accepted)
        return accepted / len(self._validations)

    @property
    def hallucination_count(self) -> int:
        return sum(1 for v in self._validations if not v.accepted)

    @property
    def total_pipeline_time(self) -> float:
        return sum(self._step_times.values())

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary of all collected metrics."""
        return {
            "query": self.query,
            "timestamp": self.timestamp,
            "step_times": self._step_times,
            "total_pipeline_time_s": round(self.total_pipeline_time, 3),
            "facets_selected": self._facets_selected,
            "facet_values": self._facet_values,
            "rag": {
                facet: {
                    "total": e.total_candidates,
                    "accepted": e.accepted_candidates,
                    "rejected": e.rejected_candidates,
                    "threshold": e.score_threshold,
                    "score_min": round(e.min_score, 4) if e.min_score is not None else None,
                    "score_max": round(e.max_score, 4) if e.max_score is not None else None,
                    "score_mean": round(e.mean_score, 4) if e.mean_score is not None else None,
                }
                for facet, e in self._rag_scores.items()
            },
            "validation": {
                "total_checks": len(self._validations),
                "accepted": sum(1 for v in self._validations if v.accepted),
                "hallucinations": self.hallucination_count,
                "accuracy": round(self.validation_accuracy, 4) if self.validation_accuracy is not None else None,
                "details": [
                    {
                        "facet": v.facet,
                        "value": v.value,
                        "accepted": v.accepted,
                        "corrected": v.corrected_value,
                    }
                    for v in self._validations
                ],
            },
            "result": {
                "total_datasets": self._total_datasets,
                "api_success": self._api_success,
            },
            "errors": self._errors,
        }

    def log_summary(self):
        """Pretty-print summary to the pipeline logger."""
        s = self.summary()
        pipeline_logger.info("═" * 60)
        pipeline_logger.info(f"PIPELINE SUMMARY for: {s['query']}")
        pipeline_logger.info(f"  Total time    : {s['total_pipeline_time_s']:.3f}s")
        for step, t in s["step_times"].items():
            pipeline_logger.info(f"    {step:30s} {t:.3f}s")
        if s["rag"]:
            pipeline_logger.info(f"  RAG filtering :")
            for facet, r in s["rag"].items():
                pipeline_logger.info(
                    f"    {facet:20s} {r['accepted']}/{r['total']} accepted "
                    f"(mean={r['score_mean']}, threshold={r['threshold']})"
                )
        v = s["validation"]
        pipeline_logger.info(
            f"  Validation    : {v['accepted']}/{v['total_checks']} accepted, "
            f"{v['hallucinations']} hallucinations"
        )
        pipeline_logger.info(f"  API result    : {s['result']['total_datasets']} datasets")
        if s["errors"]:
            pipeline_logger.warning(f"  Errors        : {s['errors']}")
        pipeline_logger.info("═" * 60)

    def to_json(self, indent: int = 2) -> str:
        """Serialize summary to JSON string (for file export / benchmarks)."""
        return json.dumps(self.summary(), indent=indent, default=str)
