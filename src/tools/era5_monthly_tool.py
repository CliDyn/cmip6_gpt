"""
ERA5 Monthly Retrieval Tool for CMIP6 Forge
=============================================
Retrieves ERA5 monthly-averaged reanalysis data on single levels
directly from the Copernicus Climate Data Store (CDS) API.

Dataset: reanalysis-era5-single-levels-monthly-means
Resolution: 0.25° x 0.25° global grid
Coverage: 1940–present (updated monthly)

Requires:
  - pip install cdsapi
  - ~/.cdsapirc with valid CDS credentials
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)


# ============================================================================
# VARIABLE CATALOGUE (ERA5 single-level monthly means)
# ============================================================================

ERA5_MONTHLY_VARIABLES = {
    # Temperature
    "2m_temperature":           {"short": "t2m",  "units": "K",     "long": "2-metre temperature"},
    "skin_temperature":         {"short": "skt",  "units": "K",     "long": "Skin temperature"},
    "sea_surface_temperature":  {"short": "sst",  "units": "K",     "long": "Sea surface temperature"},
    # Precipitation
    "total_precipitation":      {"short": "tp",   "units": "m",     "long": "Total precipitation"},
    "convective_precipitation": {"short": "cp",   "units": "m",     "long": "Convective precipitation"},
    "large_scale_precipitation":{"short": "lsp",  "units": "m",     "long": "Large-scale precipitation"},
    # Pressure
    "mean_sea_level_pressure":  {"short": "msl",  "units": "Pa",    "long": "Mean sea level pressure"},
    "surface_pressure":         {"short": "sp",   "units": "Pa",    "long": "Surface pressure"},
    # Wind
    "10m_u_component_of_wind":  {"short": "u10",  "units": "m/s",   "long": "10-metre U wind component"},
    "10m_v_component_of_wind":  {"short": "v10",  "units": "m/s",   "long": "10-metre V wind component"},
    # Radiation
    "surface_solar_radiation_downwards":        {"short": "ssrd",  "units": "J/m²", "long": "Surface solar radiation downwards"},
    "surface_thermal_radiation_downwards":      {"short": "strd",  "units": "J/m²", "long": "Surface thermal radiation downwards"},
    "top_net_solar_radiation":                  {"short": "tsr",   "units": "J/m²", "long": "Top net solar radiation"},
    "top_net_thermal_radiation":                {"short": "ttr",   "units": "J/m²", "long": "Top net thermal radiation"},
    # Cloud & moisture
    "total_cloud_cover":        {"short": "tcc",  "units": "0-1",   "long": "Total cloud cover"},
    "total_column_water_vapour":{"short": "tcwv", "units": "kg/m²", "long": "Total column water vapour"},
    # Snow & ice
    "snow_depth":               {"short": "sd",   "units": "m",     "long": "Snow depth (water equivalent)"},
    "sea_ice_cover":            {"short": "siconc","units": "0-1",  "long": "Sea-ice area fraction"},
    # Evaporation & heat flux
    "evaporation":              {"short": "e",    "units": "m",     "long": "Evaporation"},
    "surface_sensible_heat_flux":{"short": "sshf", "units": "J/m²", "long": "Surface sensible heat flux"},
    "surface_latent_heat_flux": {"short": "slhf",  "units": "J/m²", "long": "Surface latent heat flux"},
}


def _resolve_variable(user_input: str) -> str:
    """Resolve a user-provided variable name to the CDS canonical name."""
    low = user_input.strip().lower().replace("-", "_").replace(" ", "_")
    # Direct match
    if low in ERA5_MONTHLY_VARIABLES:
        return low
    # Match by short name
    for cds_name, meta in ERA5_MONTHLY_VARIABLES.items():
        if meta["short"] == low:
            return cds_name
    # Fuzzy substring match
    for cds_name in ERA5_MONTHLY_VARIABLES:
        if low in cds_name:
            return cds_name
    return user_input  # Pass through, let CDS API validate


def list_era5_variables() -> str:
    """Return a formatted list of available ERA5 monthly variables."""
    lines = ["Available ERA5 monthly single-level variables:", ""]
    for cds_name, meta in ERA5_MONTHLY_VARIABLES.items():
        lines.append(f"  {meta['short']:6s}  {meta['long']} [{meta['units']}]  (CDS: {cds_name})")
    return "\n".join(lines)


# ============================================================================
# ARGUMENT SCHEMA
# ============================================================================

class ERA5MonthlyArgs(BaseModel):
    """Arguments for ERA5 monthly data retrieval from Copernicus CDS."""

    variable: str = Field(
        description=(
            "ERA5 variable to retrieve. Use short names or full CDS names.\n"
            "Common: t2m (2m temperature), sst (sea surface temp), tp (precipitation),\n"
            "msl (mean sea level pressure), u10/v10 (10m wind), tcc (cloud cover),\n"
            "siconc (sea ice cover), tcwv (total column water vapour)"
        )
    )

    year_start: int = Field(
        description="Start year (e.g. 1979). Data available from 1940.",
        ge=1940
    )

    year_end: int = Field(
        description="End year (e.g. 2023). Must be >= year_start.",
        ge=1940
    )

    months: Optional[List[int]] = Field(
        default=None,
        description=(
            "List of months to retrieve (1-12). Default: all 12 months.\n"
            "Example: [6, 7, 8] for JJA (summer only)."
        )
    )

    area: Optional[List[float]] = Field(
        default=None,
        description=(
            "Geographic bounding box as [North, West, South, East] in degrees.\n"
            "Example: [70, -20, 30, 40] for Europe.\n"
            "Default: global. Latitude: -90 to 90, Longitude: -180 to 180."
        )
    )

    @field_validator("year_end")
    @classmethod
    def validate_year_range(cls, v, info):
        if "year_start" in info.data and v < info.data["year_start"]:
            raise ValueError(f"year_end ({v}) must be >= year_start ({info.data['year_start']})")
        return v


# ============================================================================
# MAIN RETRIEVAL FUNCTION
# ============================================================================

def retrieve_era5_monthly(
    variable: str,
    year_start: int,
    year_end: int,
    months: Optional[List[int]] = None,
    area: Optional[List[float]] = None,
) -> str:
    """
    Retrieve ERA5 monthly-averaged single-level data from Copernicus CDS.

    Downloads NetCDF to the session sandbox and returns the file path
    so the Python REPL can load it with xarray.

    Returns:
        Success message with file path and metadata, or error message.
    """
    try:
        import cdsapi
    except ImportError:
        return (
            "Error: 'cdsapi' library not installed.\n"
            "Please run: pip install cdsapi\n"
            "And configure ~/.cdsapirc with your CDS credentials."
        )

    # Resolve variable
    cds_variable = _resolve_variable(variable)
    meta = ERA5_MONTHLY_VARIABLES.get(cds_variable)

    # Build month list
    if months is None:
        month_list = [f"{m:02d}" for m in range(1, 13)]
    else:
        month_list = [f"{m:02d}" for m in sorted(set(months))]

    # Build year list
    year_list = [str(y) for y in range(year_start, year_end + 1)]

    # Output path
    output_dir = Path("era5_monthly_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    area_tag = "global" if area is None else f"{area[0]}N_{area[2]}S_{area[1]}W_{area[3]}E"
    filename = f"era5_{cds_variable}_{year_start}-{year_end}_{area_tag}.nc"
    output_path = output_dir / filename

    # Check cache
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        return (
            f"CACHE HIT — File already exists\n"
            f"  Variable: {cds_variable}"
            + (f" ({meta['long']}, {meta['units']})" if meta else "") +
            f"\n  Period: {year_start}–{year_end}\n"
            f"  File: {output_path}  ({size_mb:.1f} MB)\n\n"
            f"Load with:\n"
            f"  import xarray as xr\n"
            f"  ds = xr.open_dataset('{output_path}')"
        )

    # Build CDS request
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [cds_variable],
        "year": year_list,
        "month": month_list,
        "time": ["00:00"],
        "data_format": "netcdf",
    }

    if area is not None:
        request["area"] = area

    # Log request
    n_months_total = len(year_list) * len(month_list)
    logger.info(f"CDS request: {cds_variable}, {year_start}-{year_end}, "
                f"{n_months_total} months, area={area or 'global'}")

    print(f"\n{'='*60}")
    print(f"DOWNLOADING ERA5 MONTHLY DATA FROM COPERNICUS CDS")
    print(f"{'='*60}")
    print(f"  Variable : {cds_variable}" + (f" ({meta['long']})" if meta else ""))
    print(f"  Period   : {year_start}–{year_end} ({len(year_list)} years)")
    print(f"  Months   : {', '.join(month_list)}")
    print(f"  Area     : {area or 'global'}")
    print(f"  Output   : {output_path}")
    print(f"{'='*60}")

    # Retrieve
    try:
        client = cdsapi.Client()
        start_time = time.time()

        client.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            request,
            str(output_path),
        )

        elapsed = time.time() - start_time
        size_mb = output_path.stat().st_size / (1024 * 1024)

        print(f"\n{'='*60}")
        print(f"DOWNLOAD COMPLETE  ({elapsed:.1f}s, {size_mb:.1f} MB)")
        print(f"{'='*60}\n")

        result = (
            f"SUCCESS — ERA5 monthly data downloaded\n"
            f"{'='*50}\n"
            f"  Variable : {cds_variable}"
        )
        if meta:
            result += f" ({meta['long']}, {meta['units']})"
        result += (
            f"\n  Period   : {year_start}–{year_end}\n"
            f"  Months   : {', '.join(month_list)}\n"
            f"  Area     : {area or 'global'}\n"
            f"  Size     : {size_mb:.1f} MB\n"
            f"  Time     : {elapsed:.1f}s\n"
            f"  File     : {output_path}\n"
            f"{'='*50}\n\n"
            f"Load with:\n"
            f"  import xarray as xr\n"
            f"  ds = xr.open_dataset('{output_path}')\n"
            f"  print(ds)\n"
            f"  data = ds['{meta['short'] if meta else cds_variable}']"
        )
        return result

    except Exception as e:
        # Clean up partial download
        if output_path.exists():
            output_path.unlink()

        error_msg = str(e)
        logger.error(f"CDS retrieval failed: {error_msg}")

        return (
            f"Error retrieving ERA5 data from CDS: {error_msg}\n\n"
            f"Troubleshooting:\n"
            f"1. Check ~/.cdsapirc exists with valid CDS API key\n"
            f"2. Verify variable name: {cds_variable}\n"
            f"3. Check date range (data available from 1940)\n"
            f"4. Try a smaller area or shorter time range\n\n"
            f"Available variables:\n{list_era5_variables()}"
        )


# ============================================================================
# LANGCHAIN TOOL
# ============================================================================

era5_monthly_tool = StructuredTool.from_function(
    func=retrieve_era5_monthly,
    name="retrieve_era5_monthly",
    description=(
        "Downloads ERA5 monthly-averaged reanalysis data on single levels "
        "from the Copernicus Climate Data Store (CDS).\n\n"
        "USE THIS TOOL when the user needs observational/reanalysis climate data "
        "for comparison with CMIP6 model output or for standalone analysis.\n\n"
        "VARIABLES: t2m (2m temperature), sst (sea surface temp), tp (precipitation), "
        "msl (mean sea level pressure), u10/v10 (10m wind), tcc (cloud cover), "
        "siconc (sea ice cover), tcwv (water vapour), and more.\n\n"
        "COVERAGE: Global, 0.25° resolution, 1940–present, monthly means.\n\n"
        "Returns NetCDF file path. Load with: xr.open_dataset('path')\n"
        "Use the Python REPL to analyze the downloaded data."
    ),
    args_schema=ERA5MonthlyArgs,
)


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print(list_era5_variables())
    print("\n\nTesting retrieval...")
    result = retrieve_era5_monthly(
        variable="t2m",
        year_start=2023,
        year_end=2023,
        months=[1],
        area=[70, -20, 30, 40],  # Europe
    )
    print(result)
