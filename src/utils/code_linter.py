"""
Climate Code AST Linter
========================
Static analysis of generated Python code to catch common geophysical
and epistemic errors BEFORE execution.

This is NOT a case-specific monkey-patch. It is a generic quarantine
for entire classes of geophysical errors identified during the
CMIP6 Forge hallucination audit.
"""

import ast
import re
from typing import List, Dict


# Variable name patterns that suggest "this should be real model data".
# Used by the AST-level fabrication detector below.
_MODEL_VAR_RE = re.compile(
    r"^("
    r"(awi|mpi|cesm|gfdl|ec[_-]?earth|hadgem|noresm|miroc|ipsl|access|canesm|ukesm|fgoals|cnrm|inm)"
    r"[_a-z0-9]*"
    r"|model_?(data|out|field|output|sst|tas|pr|tos|zos|psl|uo|vo)\w*"
    r"|cmip6?_?\w*"
    r"|(historical|ssp\d{3}|projection)_?(data|fld|out)\w*"
    r")$",
    re.IGNORECASE,
)


class ClimateCodeLinter(ast.NodeVisitor):
    """AST visitor that flags dangerous patterns in climate analysis code."""

    def __init__(self):
        self.issues: List[Dict] = []

    # ─── Synthetic data fabrication (AST-level structural detection) ───

    def _flag_random_model_assignment(self, target_name: str, location: str = "") -> None:
        self.issues.append({
            "severity": "critical",
            "issue": (
                f"DATA FABRICATION: variable '{target_name}' looks like a CMIP6/model "
                f"field but is being populated from np.random / synthetic source"
                f"{(' (' + location + ')') if location else ''}. "
                "FORBIDDEN: agents must NEVER fabricate model output when real fetch fails. "
                "Stop, report the failure, and surface the load error to the user."
            ),
        })

    def _is_random_call(self, value: ast.AST) -> bool:
        """Detect np.random.*, numpy.random.*, random.*, torch.randn-style calls."""
        if isinstance(value, ast.Call):
            f = value.func
            # np.random.normal(...), np.random.uniform(...)
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Attribute):
                if f.value.attr == "random":
                    return True
            # np.random.<x> directly (when imported as `from numpy import random`)
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                if f.value.id in ("random",) and f.attr in (
                    "normal", "uniform", "randn", "rand", "standard_normal",
                    "choice", "randint",
                ):
                    return True
            # np.random.default_rng().normal(...)
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call):
                inner = f.value.func
                if (isinstance(inner, ast.Attribute)
                        and inner.attr in ("default_rng", "RandomState")):
                    return True
        return False

    def visit_Assign(self, node):
        """Catch `awi_tos = np.random.normal(...)` and friends."""
        if self._is_random_call(node.value):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and _MODEL_VAR_RE.match(tgt.id):
                    self._flag_random_model_assignment(tgt.id, location="np.random call")
                elif isinstance(tgt, ast.Tuple):
                    for elt in tgt.elts:
                        if isinstance(elt, ast.Name) and _MODEL_VAR_RE.match(elt.id):
                            self._flag_random_model_assignment(elt.id, location="np.random tuple")
        # xr.DataArray(np.random.normal(...), ...) bound to a model-named var
        if isinstance(node.value, ast.Call):
            f = node.value.func
            is_xr_dataarray = (
                isinstance(f, ast.Attribute) and f.attr in ("DataArray", "Dataset")
            ) or (isinstance(f, ast.Name) and f.id in ("DataArray", "Dataset"))
            if is_xr_dataarray and node.value.args:
                first = node.value.args[0]
                if self._is_random_call(first):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name) and _MODEL_VAR_RE.match(tgt.id):
                            self._flag_random_model_assignment(tgt.id, location="xr.DataArray(np.random...)")
        self.generic_visit(node)

    def visit_Call(self, node):
        # --- Ban image-space gradient operators on geophysical data ---
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr == "sobel":
                self.issues.append({
                    "severity": "critical",
                    "issue": (
                        "EPISTEMIC ERROR: scipy.ndimage.sobel computes index-space gradients. "
                        "This is physically invalid on Earth grids — pixel spacing varies with "
                        "latitude. Use metric-aware derivatives (xr.differentiate with physical "
                        "distance scaling) or regrid to a regular grid first."
                    ),
                })
            # Also catch cv2/skimage gradient ops
            if attr in {"Sobel", "Laplacian", "Scharr"}:
                self.issues.append({
                    "severity": "critical",
                    "issue": (
                        f"EPISTEMIC ERROR: {attr}() is an image-processing operator. "
                        "Do NOT apply to geophysical lat/lon grids."
                    ),
                })

        # --- Flag unweighted spatial means ---
        if isinstance(node.func, ast.Attribute) and node.func.attr == "mean":
            # Skip if this is chained after .weighted() — that IS correct
            is_weighted = (
                isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Attribute)
                and node.func.value.func.attr == "weighted"
            )
            if not is_weighted:
                # Check if dim argument contains lat/lon
                for kw in node.keywords:
                    if kw.arg == "dim":
                        dim_str = ast.dump(kw.value).lower()
                        if any(d in dim_str for d in ["lat", "lon", "latitude", "longitude"]):
                            self.issues.append({
                                "severity": "warning",
                                "issue": (
                                    "Unweighted spatial mean over lat/lon detected. "
                                    "Grid cells shrink toward the poles. Use "
                                    ".weighted(np.cos(np.deg2rad(lat))).mean() or "
                                    "aligned_weighted_mean() for physically correct averages."
                                ),
                            })
                # Also check positional dim argument
                if node.args:
                    arg_str = ast.dump(node.args[0]).lower()
                    if any(d in arg_str for d in ["lat", "lon", "latitude", "longitude"]):
                        self.issues.append({
                            "severity": "warning",
                            "issue": (
                                "Unweighted spatial mean over lat/lon detected. "
                                "Use .weighted(cos(lat)).mean() for correct Earth averaging."
                            ),
                        })

        # --- Flag suspicious set_ylim / set_xlim ---
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"set_ylim", "set_xlim"}:
            self.issues.append({
                "severity": "info",
                "issue": (
                    f"Explicit axis limits ({node.func.attr}) detected. "
                    "Verify they do not hide required lines, baselines, or data."
                ),
            })

        self.generic_visit(node)

    def visit_AugAssign(self, node):
        """Catch scalar guardrail hacking like `data -= bias` or `data += 0.63`."""
        if isinstance(node.op, (ast.Add, ast.Sub)):
            # Only flag if the right side is a simple number or name (not array ops)
            if isinstance(node.value, (ast.Constant, ast.Name)):
                self.issues.append({
                    "severity": "warning",
                    "issue": (
                        "Additive in-place scalar shift detected (e.g. data -= bias). "
                        "Check whether this is a legitimate unit conversion or a "
                        "guardrail hack to force baselines to match."
                    ),
                })
        self.generic_visit(node)


def lint_climate_code(code_str: str) -> List[Dict]:
    """
    Parse and lint a code string for common climate analysis errors.

    Returns a list of dicts with 'severity' and 'issue' keys.
    Severity levels: 'critical', 'warning', 'info'
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return []  # Can't lint unparseable code; let exec() handle the error

    linter = ClimateCodeLinter()
    linter.visit(tree)
    return linter.issues


def format_linter_warnings(issues: List[Dict]) -> str:
    """Format linter issues into a readable string for REPL stdout."""
    if not issues:
        return ""

    lines = ["", "=" * 60, "⚠️  CLIMATE CODE LINTER WARNINGS", "=" * 60]
    for issue in issues:
        sev = issue["severity"].upper()
        lines.append(f"  [{sev}] {issue['issue']}")
    lines.append("=" * 60)
    lines.append("")
    return "\n".join(lines)
