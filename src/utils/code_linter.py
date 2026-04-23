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
from typing import List, Dict


class ClimateCodeLinter(ast.NodeVisitor):
    """AST visitor that flags dangerous patterns in climate analysis code."""

    def __init__(self):
        self.issues: List[Dict] = []

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
