"""
Audited Scientific Primitives for Climate Data Analysis
========================================================
Pre-tested helper functions for common geophysical operations.
These are loaded into the Python REPL namespace so the agent
can use them instead of reinventing error-prone ad-hoc solutions.

All functions are designed to FAIL LOUDLY on common mistakes
rather than silently producing wrong results.
"""

import numpy as np
import xarray as xr
from typing import List, Optional, Dict, Any

EARTH_RADIUS_M = 6_371_000.0


# ─── Weighted Spatial Mean ───────────────────────────────────────────

def aligned_weighted_mean(
    da: xr.DataArray,
    weights: xr.DataArray,
    dims: List[str],
) -> xr.DataArray:
    """Area-weighted mean with strict alignment and denominator checks.
    
    Prevents the UC4-class bug where data is masked to a region but
    weights cover the entire globe, artificially crushing the result.
    
    Args:
        da: Data to average (may contain NaNs from masking)
        weights: Weight array (e.g., cos(lat) or areacella)
        dims: Dimensions to average over (e.g., ['lat', 'lon'])
    
    Raises:
        ValueError: If dimensions are missing or denominator is non-positive
    """
    da, weights = xr.align(da, weights, join="exact")

    missing = [d for d in dims if d not in da.dims]
    if missing:
        raise ValueError(f"Missing dimensions in data: {missing}")

    # Mask weights to match data NaNs
    w = weights.where(da.notnull())
    denom = w.sum(dims, skipna=True)

    if float(denom.min()) <= 0:
        raise ValueError(
            "Weighted mean denominator is non-positive after masking. "
            "Your mask/domain mismatch is likely — ensure weights are "
            "masked identically to data."
        )

    num = (da * w).sum(dims, skipna=True)
    return num / denom


# ─── Physical-Space Gradients ────────────────────────────────────────

def lonlat_gradient_magnitude(
    da: xr.DataArray,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """Compute gradient magnitude using physical distances on the sphere.
    
    Unlike scipy.ndimage.sobel (which computes per-pixel gradients),
    this function accounts for the fact that 1° of longitude shrinks
    toward the poles.
    
    Args:
        da: 2D+ DataArray with lat/lon coordinates
        lat_name: Name of latitude coordinate
        lon_name: Name of longitude coordinate
    
    Returns:
        Gradient magnitude in units of [da_units / metre]
    """
    lat = da[lat_name]

    d_da_dlat = da.differentiate(lat_name)
    d_da_dlon = da.differentiate(lon_name)

    meters_per_deg_lat = np.pi * EARTH_RADIUS_M / 180.0
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(lat))

    d_da_dy = d_da_dlat / meters_per_deg_lat
    d_da_dx = d_da_dlon / meters_per_deg_lon

    return np.hypot(d_da_dx, d_da_dy)


# ─── Seasonal Helpers (DJF Boundary Safe) ────────────────────────────

def add_season_year(ds):
    """Add a season_year coordinate that keeps DJF together.
    
    December gets assigned to the NEXT year's winter, so that
    Dec 2014 + Jan 2015 + Feb 2015 are all season_year=2015.
    This prevents the historical/SSP boundary from splitting winters.
    """
    month = ds.time.dt.month
    year = ds.time.dt.year
    season_year = xr.where(month == 12, year + 1, year)
    return ds.assign_coords(season_year=("time", season_year.data))


def concat_historical_and_scenario(hist, fut):
    """Concatenate historical and SSP datasets on time axis.
    
    The correct order for seasonal analysis across the 2014/2015
    boundary: concatenate FIRST, then aggregate.
    """
    return xr.concat([hist, fut], dim="time").sortby("time")


def seasonal_mean_continuous(ds, months: List[int]):
    """Compute seasonal means that respect year boundaries.
    
    Uses season_year grouping so DJF 2014-15 is not split.
    
    Args:
        ds: Dataset or DataArray with time dimension
        months: List of months defining the season (e.g., [12, 1, 2] for DJF)
    """
    ds = add_season_year(ds)
    subset = ds.sel(time=ds.time.dt.month.isin(months))
    return subset.groupby("season_year").mean("time")


# ─── Figure Metadata QA ─────────────────────────────────────────────

def extract_figure_metadata(fig) -> Dict[str, Any]:
    """Extract machine-readable metadata from a matplotlib figure.
    
    Used by figure QA to detect hidden lines, missing legends,
    blank maps, and other visual issues WITHOUT relying on
    vision models (which miss these systematically).
    """
    meta = {"axes": []}
    for ax in fig.axes:
        ax_meta = {
            "title": ax.get_title(),
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "xlim": tuple(map(float, ax.get_xlim())),
            "ylim": tuple(map(float, ax.get_ylim())),
            "legend_labels": [],
            "lines": [],
        }

        leg = ax.get_legend()
        if leg:
            ax_meta["legend_labels"] = [t.get_text() for t in leg.texts]

        for line in ax.get_lines():
            y = np.asarray(line.get_ydata(), dtype=float)
            finite = np.isfinite(y)
            ax_meta["lines"].append({
                "label": line.get_label(),
                "n_points": int(finite.sum()),
                "ymin": float(np.nanmin(y)) if finite.any() else None,
                "ymax": float(np.nanmax(y)) if finite.any() else None,
            })

        meta["axes"].append(ax_meta)
    return meta


def hidden_lines_qa(meta: Dict) -> List[str]:
    """Check if any plotted line is fully outside the visible y-axis limits.
    
    This catches the UC11-class bug where set_ylim clips the
    ERA5 observational baseline off the visible area.
    
    Returns:
        List of issue strings (empty = no issues found)
    """
    issues = []
    for ax in meta["axes"]:
        lo, hi = ax["ylim"]
        for line in ax["lines"]:
            ymin, ymax = line.get("ymin"), line.get("ymax")
            if ymin is None or ymax is None:
                continue
            if ymax < lo or ymin > hi:
                issues.append(
                    f"Line '{line['label']}' lies fully outside y-limits "
                    f"[{lo:.2f}, {hi:.2f}] (data range: [{ymin:.2f}, {ymax:.2f}]). "
                    f"This may indicate a hidden baseline."
                )
    return issues
