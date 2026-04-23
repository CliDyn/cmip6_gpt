"""
Task Specification and Method Contract Schemas
================================================
Typed Pydantic models that formalize scientific constraints BEFORE
code generation begins. These schemas catch failures like:
- Using 'tas' when the task requires 'tasmax' (variable substitution)
- Computing chronological RMSE on free-running models
- Applying image-space gradient operators to geophysical grids

These models are created here for documentation and future enforcement.
They are NOT wired into the agent loop yet — that requires a LangGraph
StateGraph migration which is a separate, larger effort.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum


class VariableRole(str, Enum):
    """Role of a variable in the analysis."""
    PRIMARY = "primary"         # The variable being analyzed
    REFERENCE = "reference"     # Observational reference (e.g., ERA5)
    AUXILIARY = "auxiliary"     # Supporting variable (e.g., areacella for weighting)


class TemporalSemantics(str, Enum):
    """How time alignment should be handled."""
    CLIMATOLOGICAL = "climatological"     # Compare long-term means (safe for free-running models)
    CHRONOLOGICAL = "chronological"       # Direct year-by-year comparison (only for initialized/reanalysis)
    DISTRIBUTIONAL = "distributional"     # Compare distributions/spectra (always safe)


class VariableNeed(BaseModel):
    """A specific variable required by the analysis."""
    variable: str = Field(description="Exact CMIP6 variable_id (e.g., 'tasmax', 'prsn', 'siconc')")
    role: VariableRole = Field(description="Role in the analysis")
    required: bool = Field(default=True, description="Whether this variable is mandatory")
    frequency: str = Field(default="mon", description="Required temporal frequency")
    reason: str = Field(description="Why this specific variable is needed (prevents substitution)")


class MetricNeed(BaseModel):
    """A metric to be computed in the analysis."""
    name: str = Field(description="Metric name (e.g., 'spatial_mean', 'RMSE', 'trend')")
    weighting: Literal["area_weighted", "unweighted", "population_weighted"] = Field(
        default="area_weighted",
        description="Required weighting scheme"
    )
    temporal_semantics: TemporalSemantics = Field(
        default=TemporalSemantics.CLIMATOLOGICAL,
        description="How temporal alignment should be handled"
    )
    allowed_for_free_running_models: bool = Field(
        default=True,
        description="Whether this metric is valid for uninitialized CMIP6 runs"
    )


class TaskSpec(BaseModel):
    """
    Typed specification of a scientific analysis task.
    
    Generated BEFORE code, this schema codifies:
    - What phenomenon is being analyzed
    - What variables are needed (exact, not proxy)
    - What metrics are appropriate
    - What shortcuts are forbidden
    
    The coverage linter can then check this spec for consistency.
    """
    objective: str = Field(description="One-sentence description of the analysis goal")
    phenomenon: str = Field(description="Physical phenomenon (e.g., 'lake-effect snow', 'ENSO teleconnections')")
    domain: Optional[str] = Field(default=None, description="Geographic domain (e.g., 'Great Lakes region, 41-48°N, 75-92°W')")
    periods: List[str] = Field(
        default_factory=list,
        description="Time periods (e.g., ['1985-2014', '2070-2099'])"
    )
    experiments: List[str] = Field(
        default_factory=list,
        description="CMIP6 experiment_ids (e.g., ['historical', 'ssp245', 'ssp585'])"
    )
    required_variables: List[VariableNeed] = Field(
        default_factory=list,
        description="All variables needed for this analysis"
    )
    metrics: List[MetricNeed] = Field(
        default_factory=list,
        description="Metrics to compute"
    )
    forbidden_shortcuts: List[str] = Field(
        default_factory=list,
        description="Explicitly banned approaches (e.g., 'Do not substitute tas for tasmax')"
    )
    can_execute: bool = Field(
        default=True,
        description="Whether the spec passes coverage linting"
    )
    blocking_reason: Optional[str] = Field(
        default=None,
        description="Why can_execute is False (if applicable)"
    )


class MethodContract(BaseModel):
    """
    Codifies the analysis methodology as a machine-readable contract.
    
    This is checked AFTER code generation to verify that the implementation
    matches the contracted methodology.
    """
    unit_conversions: List[str] = Field(
        default_factory=list,
        description="All unit conversions applied (e.g., 'pr: kg/m2/s → mm/day × 86400')"
    )
    spatial_weighting: str = Field(
        default="cosine_latitude",
        description="Method used for area weighting"
    )
    temporal_aggregation: str = Field(
        default="monthly_climatology",
        description="How time is aggregated (e.g., 'monthly_climatology', 'annual_mean', 'DJF_seasonal')"
    )
    baseline_period: Optional[str] = Field(
        default=None,
        description="Reference period for anomalies (e.g., '1981-2010')"
    )
    regridding_method: Optional[str] = Field(
        default=None,
        description="Method used for regridding (e.g., 'bilinear', 'conservative', 'none')"
    )
    bias_correction: Optional[str] = Field(
        default=None,
        description="Bias correction method applied (e.g., 'QDM', 'EQM', 'none')"
    )
