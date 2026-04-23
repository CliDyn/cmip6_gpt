"""
Coverage Linter for TaskSpec
==============================
Validates a TaskSpec against known scientific rules BEFORE
any code is generated. Catches failures like:
- Heatwave analysis without 'tasmax' at daily frequency
- Lake-effect snow without 'prsn' or ice data
- Gradient analysis without physical-space computation
- Regional means without area weighting

This is a rule-based system (no LLM calls, no cost).
"""

import re
from typing import List
from src.models.task_spec import TaskSpec, TemporalSemantics


def lint_task_spec(spec: TaskSpec) -> TaskSpec:
    """
    Validate a TaskSpec and set can_execute=False if critical issues found.
    Modifies the spec in-place and returns it.
    """
    issues = []
    phenomenon_lower = spec.phenomenon.lower()
    objective_lower = spec.objective.lower()
    combined = f"{phenomenon_lower} {objective_lower}"
    
    # Collect all variable names in the spec
    var_ids = {v.variable for v in spec.required_variables}
    var_freqs = {v.variable: v.frequency for v in spec.required_variables}

    # ─── Rule 1: Heatwave / extreme heat → require tasmax + daily ───
    if any(kw in combined for kw in ["heatwave", "heat wave", "extreme heat", "hot day"]):
        if "tasmax" not in var_ids:
            issues.append(
                "Heatwave analysis requires 'tasmax' (daily max temperature), "
                "not 'tas' (daily mean). Add VariableNeed(variable='tasmax', "
                "frequency='day', role='primary')"
            )
        elif var_freqs.get("tasmax") != "day":
            issues.append(
                "Heatwave analysis requires DAILY 'tasmax'. "
                "Monthly data cannot detect heat extremes."
            )

    # ─── Rule 2: Lake-effect snow → require snowfall + ice ───
    if "lake" in combined and "snow" in combined:
        if "prsn" not in var_ids:
            issues.append(
                "Lake-effect snow analysis requires 'prsn' (snowfall flux), "
                "not 'pr' (total precipitation). Snowfall ≠ rainfall."
            )
        if "siconc" not in var_ids and "sic" not in var_ids:
            issues.append(
                "Lake-effect snow is driven by lake ice cover. "
                "Consider adding 'siconc' or lake ice fraction."
            )

    # ─── Rule 3: Gradient/front analysis → ban image-space filters ───
    if any(kw in combined for kw in ["gradient", "front", "frontal", "baroclin"]):
        for shortcut in spec.forbidden_shortcuts:
            pass  # These are already explicitly forbidden
        # Auto-add if not present
        bad_tools = ["sobel", "cv2", "skimage", "ndimage"]
        existing_bans = " ".join(spec.forbidden_shortcuts).lower()
        if not any(t in existing_bans for t in bad_tools):
            spec.forbidden_shortcuts.append(
                "Do not use image-space gradient operators (sobel, cv2, skimage). "
                "Use physical-space derivatives with Earth radius scaling."
            )

    # ─── Rule 4: Regional / global means → require area weighting ───
    if any(kw in combined for kw in ["mean", "average", "rmse", "bias"]):
        for metric in spec.metrics:
            if metric.weighting == "unweighted":
                issues.append(
                    f"Metric '{metric.name}' is unweighted. "
                    "Spatial statistics on Earth grids MUST use area weighting."
                )
                metric.weighting = "area_weighted"

    # ─── Rule 5: Internal variability → forbid chronological RMSE ───
    if any(kw in combined for kw in ["internal variability", "free-running", "uninitialized"]):
        for metric in spec.metrics:
            if metric.temporal_semantics == TemporalSemantics.CHRONOLOGICAL:
                issues.append(
                    f"Metric '{metric.name}' uses chronological alignment, "
                    "but the task involves free-running models. Chronological RMSE "
                    "is scientifically invalid — use climatological or distributional."
                )
                metric.allowed_for_free_running_models = False

    # ─── Rule 6: Chronological RMSE on CMIP6 free-running models ───
    for metric in spec.metrics:
        if (metric.name.lower() in ["rmse", "mae", "mse"]
            and metric.temporal_semantics == TemporalSemantics.CHRONOLOGICAL
            and "historical" in spec.experiments):
            issues.append(
                f"Metric '{metric.name}' with chronological alignment against "
                "'historical' experiment is likely invalid for uninitialized models. "
                "Consider using climatological comparison instead."
            )

    # ─── Apply results ───
    if issues:
        spec.can_execute = False
        spec.blocking_reason = " | ".join(issues)
    else:
        spec.can_execute = True
        spec.blocking_reason = None

    return spec
