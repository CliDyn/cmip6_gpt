"""
Structured Review Schema
=========================
Pydantic models for structured peer review output.

These schemas formalize the review process so that issues have
explicit severity, evidence type, confidence, and verification flags.
This prevents the "confidently wrong reviewer" failure mode.

Currently created as importable models + documentation of the target.
The current reviewer tools remain text-based since forcing structured
output from external LLMs requires `with_structured_output()` which
may not be stable across all reviewer model providers.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class ReviewIssue(BaseModel):
    """A single issue identified by a reviewer."""
    severity: Literal["critical", "major", "minor"] = Field(
        description="Severity of the issue"
    )
    claim: str = Field(
        description="What the reviewer claims is wrong"
    )
    evidence_type: Literal["stdout_proven", "code_logic", "hypothesis", "figure_visual"] = Field(
        description=(
            "How the reviewer supports this claim:\n"
            "- stdout_proven: The STDOUT data directly proves the issue\n"
            "- code_logic: The code has a clear logical bug\n"
            "- hypothesis: The reviewer suspects an issue but cannot prove it\n"
            "- figure_visual: The figure shows something anomalous"
        )
    )
    evidence: str = Field(
        description="Specific evidence (line numbers, values from stdout, etc.)"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="How confident the reviewer is. LOW confidence + CRITICAL severity is INVALID."
    )
    proposed_fix: Optional[str] = Field(
        default=None,
        description="Specific code change to fix the issue"
    )
    requires_verification: bool = Field(
        default=True,
        description=(
            "Whether the fix must be empirically tested before accepting. "
            "True for any mathematical change. False for cosmetic changes."
        )
    )


class ReviewReport(BaseModel):
    """Complete review report from a single reviewer."""
    issues: List[ReviewIssue] = Field(default_factory=list)
    verdict: Literal["accept", "revise", "reject"] = Field(
        description="Overall verdict"
    )


class VerificationDecision(BaseModel):
    """
    The worker agent's decision on a reviewer's claim.
    
    This is the output of the Empirical Defiance Protocol:
    for each reviewer claim, the worker must decide whether to
    accept, reject, or verify the claim.
    """
    claim: str = Field(description="The reviewer's original claim")
    decision: Literal["accept", "reject", "verify"] = Field(
        description="Worker's decision on this claim"
    )
    rationale: str = Field(
        description="Why the worker made this decision"
    )
    before_stats: Optional[str] = Field(
        default=None,
        description="Summary stats BEFORE applying the reviewer's fix"
    )
    after_stats: Optional[str] = Field(
        default=None,
        description="Summary stats AFTER applying the reviewer's fix (if verified)"
    )
    violated_invariants: List[str] = Field(
        default_factory=list,
        description="Physical invariants violated by the fix (e.g., 'precipitation < 0')"
    )
