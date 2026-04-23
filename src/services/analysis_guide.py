"""
CMIP6 Analysis Guide Tool
==========================
Provides methodological guidance for CMIP6 climate data analysis using python_repl.

This tool returns TEXT INSTRUCTIONS (not executable code!) for:
- What approach to take with CMIP6 multi-model data
- How to structure the analysis
- Quality checks and pitfalls
- Best practices for visualization

The agent uses python_repl to execute the actual analysis.
"""

from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool


# =============================================================================
# ANALYSIS GUIDES
# =============================================================================

ANALYSIS_GUIDES = {
    # -------------------------------------------------------------------------
    # DATA OPERATIONS
    # -------------------------------------------------------------------------
    "load_data": """
## Loading CMIP6 Data from Google Cloud Storage

### When to use
- Initializing any analysis after `cmip6_datasets_search` has found datasets

### Workflow
1. **Query the catalog** — Filter the Pangeo CMIP6 CSV catalog:
   ```
   df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
   df_sub = df.query("variable_id=='tos' & source_id=='MPI-ESM1-2-LR' & experiment_id=='historical'")
   ```
2. **Open Zarr store** — `ds = xr.open_zarr(df_sub.zstore.values[0], consolidated=True, storage_options={'token':'anon'})`
3. **Inspect** — Check `ds.coords`, `ds.data_vars`, `ds.attrs`
4. **Convert units** immediately:
   - Temperature (`tas`, `ts`, `tos`): Usually already in K → subtract 273.15 → °C
   - Precipitation (`pr`): kg/m²/s → multiply by 86400 → mm/day
   - Pressure (`psl`): Pa → divide by 100 → hPa
   - Sea ice fraction (`siconc`): 0–100% (check if fraction 0–1)
5. **Handle model calendars** — CMIP6 models use `noleap`, `360_day`, `all_leap`, etc.
   Use `xr.decode_cf()` or `cftime` for time indexing.

### Quality Checklist
- [ ] Data loaded lazily (no `.load()` on full dataset)
- [ ] Units converted before any aggregation
- [ ] Coordinate names checked (`lat`/`latitude`/`nav_lat` vary by model)
- [ ] Calendar type checked: `ds.time.encoding.get('calendar', 'standard')`
- [ ] Grid type checked: regular vs curvilinear (affects plotting)

### Common Pitfalls
- ⚠️ CMIP6 Zarr stores on GCS can be multi-GB — keep operations lazy until subsetted.
- ⚠️ Some models use `latitude`/`longitude`, others use `lat`/`lon`, and ocean models may use `nav_lat`/`nav_lon` or `nlat`/`nlon`. Always check with `.coords`.
- ⚠️ Non-standard calendars (`360_day`, `noleap`) cause errors with `pd.Timestamp`. Use `cftime`-aware operations or `ds.convert_calendar('standard')`.
- ⚠️ Ocean variables (`tos`, `sos`, `zos`) are on ocean grids with NaN over land — do NOT expect rectangular lat/lon.
- ⚠️ CMIP6 stores the SAME variable across multiple `table_id` values (e.g., `Amon` vs `Amon` vs `day`). Always filter by `table_id` to avoid duplicates.

### ⚠️ CRITICAL: Unstructured Grids (FESOM / AWI-CM)
AWI-CM uses FESOM with **unstructured triangular meshes** (dim: `ncells`). These are HUGE (~830k cells).
**NEVER .compute() the full array!** Follow this pattern:
1. Open lazily: `da = xr.open_zarr(...)['uo'].isel(depth=0)` — stays as dask array
2. Subset time lazily: `da = da.sel(time=slice('1990','2014'))` — STILL lazy
3. Compute climatology lazily: `clim = da.groupby('time.month').mean('time')`
4. Compute anomalies lazily: `anom = da.groupby('time.month') - clim`
5. Compute FINAL result only: `result = (0.5 * anom**2).mean('time').compute()` — THIS materializes
6. For annual time series, compute YEAR BY YEAR:
   ```
   annual = []
   for yr in range(1990, 2015):
       chunk = da.sel(time=str(yr)).compute()  # ~10 months, manageable
       annual.append(float(chunk.mean()))
   ```
7. For scatter plots, SUBSAMPLE: `idx = np.random.choice(len(lon), 200000, replace=False)`
8. NEVER write a single code block that does everything — SPLIT into multiple REPL calls:
   - Call 1: Open + inspect
   - Call 2: Compute result arrays
   - Call 3: Plot
""",

    "spatial_subset": """
## Spatial Subsetting

### When to use
- Focusing on a specific region, country, or ocean basin
- Reducing data size before heavy analysis

### Workflow
1. **Determine bounds** — Find min/max latitude and longitude.
2. **Check coordinate orientation** — Some models have ascending lat, some descending.
3. **Slice data** — `.sel(lat=slice(south, north), lon=slice(west, east))` (adjust for orientation).
4. **For curvilinear ocean grids** — Use `.where()` masking instead of `.sel()`.

### Quality Checklist
- [ ] Latitude orientation verified: `ds.lat[0] > ds.lat[-1]` → descending → `slice(north, south)`
- [ ] Longitude format checked (0–360 vs -180–180) and converted if needed
- [ ] Result is not empty — verify with `.shape`
- [ ] For ocean grids: check if grid is regular or curvilinear

### Common Pitfalls
- ⚠️ Slicing incorrectly on descending coords → empty array.
- ⚠️ Some CMIP6 ocean models use tripolar grids — `.sel()` won't work. Use `where((lat > S) & (lat < N) & (lon > W) & (lon < E))`.
- ⚠️ Longitude wrapping: if model uses 0–360 but you need -180–180, use `ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')`.
""",

    "temporal_subset": """
## Temporal Subsetting & Aggregation

### When to use
- Isolating specific periods, seasons, or decades
- Computing annual/seasonal means from monthly data

### Workflow
1. **Time slice** — `.sel(time=slice('1950-01-01', '2014-12-31'))`.
2. **Season filter** — `.sel(time=ds.time.dt.season == 'DJF')`.
3. **Annual mean** — `.groupby('time.year').mean('time')` or `.resample(time='YS').mean()`.
4. **Seasonal mean** — `.groupby('time.dt.season').mean('time')` for climatological seasonal means.

### Quality Checklist
- [ ] Aggregation matches variable: `.mean()` for temperature/pressure, `.sum()` for precip (after converting to mm/month)
- [ ] Model calendar handled (use `cftime` if `360_day` or `noleap`)
- [ ] Historical period ends at 2014 in CMIP6; SSP scenarios start at 2015

### Common Pitfalls
- ⚠️ DJF wraps across years — December 2013 is part of DJF 2013-14.
- ⚠️ `.resample()` may fail on `cftime` calendars — use `.groupby('time.year')` instead.
- ⚠️ CMIP6 historical stops at 2014-12-31. For 2015+, you need SSP scenario data and must concatenate with `xr.concat([hist, ssp], dim='time')`.
- ⚠️ Precipitation in CMIP6 is a RATE (kg/m²/s). To get monthly totals, multiply by seconds-per-month, NOT by 86400×30.
""",

    # -------------------------------------------------------------------------
    # STATISTICAL ANALYSIS
    # -------------------------------------------------------------------------
    "anomalies": """
## Anomaly Analysis

### When to use
- "How unusual was this period?"
- Comparing model output to climatological average
- Any "above/below average" question

### Workflow
1. **Define baseline** — Historical period: 1850–2014 (or sub-period like 1981–2010).
2. **Compute climatology** — `clim = ds.sel(time=slice('1981','2010')).groupby('time.month').mean('time')`.
3. **Subtract** — `anomaly = ds.groupby('time.month') - clim`.
4. **Convert units** — Report in °C, mm/day (not K, kg/m²/s).
5. **Multi-model** — Compute anomaly per model, then ensemble mean ± spread.

### Quality Checklist
- [ ] Baseline ≥20 years for stable climatology
- [ ] Same calendar grouping for climatology and data
- [ ] Units converted for readability
- [ ] For multi-model: anomalies computed PER MODEL before averaging

### Common Pitfalls
- ⚠️ Computing climatology across models BEFORE anomalies mixes model biases.
- ⚠️ CRITICAL: Temperature anomaly in K = anomaly in °C. Do NOT subtract 273.15 from a temperature difference.
- ⚠️ For SSP scenarios, the baseline MUST come from the corresponding historical run, not the SSP itself.

### Interpretation
- Positive = warmer/wetter than baseline. ±1σ = common, ±2σ = unusual, ±3σ = extreme.
- Maps: Use `RdBu_r` centered at zero via `TwoSlopeNorm`.
""",

    "zscore": """
## Z-Score Analysis (Standardized Anomalies)

### When to use
- Comparing extremity across different variables or models
- Standardizing across regions with different variability
- Identifying statistically significant departures

### Workflow
1. **Compute baseline mean** — Grouped by month for seasonality.
2. **Compute baseline std** — Same period, same grouping.
3. **Standardize** — `z = (value - mean) / std`.

### Quality Checklist
- [ ] Standard deviation is non-zero everywhere
- [ ] Baseline period matches for mean and std
- [ ] Computed per model, not across models

### Common Pitfalls
- ⚠️ Precipitation is NOT normally distributed — use percentiles instead of raw Z-scores.
- ⚠️ Z-scores near coastlines can be extreme due to mixed land/ocean grid cells.
- ⚠️ Some CMIP6 models have very low variability in certain regions → inflated Z-scores.

### Interpretation
- Z = 0: average. ±1: normal (68%). ±2: unusual (5%). ±3: extreme (0.3%).
""",

    "trend_analysis": """
## Linear Trend Analysis

### When to use
- "Is temperature increasing over time?"
- Detecting climate change signals in model projections
- Comparing trend magnitude across models/scenarios

### Workflow
1. **Downsample** — Convert to annual means: `.groupby('time.year').mean('time')`.
2. **Regress** — `scipy.stats.linregress` or `np.polyfit(degree=1)` per grid point.
3. **Significance** — Extract p-value.
4. **Scale** — Multiply annual slope × 10 → "per decade".
5. **Multi-model** — Compute trend per model, report median ± spread.

### Quality Checklist
- [ ] Period stated clearly (e.g., 2015–2100 for SSP)
- [ ] Seasonal cycle removed before fitting
- [ ] Significance tested (p < 0.05)
- [ ] Report trend as units/decade

### Common Pitfalls
- ⚠️ Trend on monthly data without removing seasonality → dominated by annual cycle.
- ⚠️ Short periods (< 30 years) have large uncertainty — report confidence intervals.
- ⚠️ If p > 0.05, state the trend is NOT statistically significant.
- ⚠️ Different CMIP6 models have wildly different climate sensitivities — always report per-model trends, not just multi-model mean.

### Interpretation
- Report as °C/decade. Maps: stipple significant grid cells.
""",

    "eof_analysis": """
## EOF/PCA Analysis

### When to use
- Finding dominant spatial patterns (ENSO, NAO, PDO-like modes)
- Dimensionality reduction of spatiotemporal data
- Comparing modes across different CMIP6 models

### Workflow
1. **Deseasonalize** — Compute anomalies.
2. **Latitude weighting** — Multiply by `np.sqrt(np.cos(np.deg2rad(lat)))`.
3. **Decompose** — PCA on flattened space dimensions.
4. **Reconstruct** — Map PCs back to spatial grid (EOFs).

### Quality Checklist
- [ ] Seasonal cycle removed
- [ ] Latitude weighting applied
- [ ] Variance explained (%) calculated per mode and shown in title
- [ ] Coastlines added to maps (use Cartopy)

### Common Pitfalls
- ⚠️ Unweighted EOFs inflate polar regions artificially.
- ⚠️ EOFs from different models are NOT directly comparable — they may have opposite sign conventions.
- ⚠️ Ocean variables on curvilinear grids need regridding before EOF analysis.

### Interpretation
- EOF1: dominant spatial pattern. PC1: its temporal evolution.
- EOF1 explaining >20% variance is highly dominant.
""",

    "correlation_analysis": """
## Correlation Analysis

### When to use
- Spatial/temporal correlation mapping
- Teleconnection exploration in model output
- Relating a climate index to spatial patterns

### Workflow
1. **Deseasonalize** — Remove seasonal cycle from both variables.
2. **Align** — Ensure identical time axes.
3. **Correlate** — `xr.corr(var1, var2, dim='time')`.
4. **Significance** — Compute p-values, mask insignificant areas.

### Quality Checklist
- [ ] Both variables deseasonalized
- [ ] p-values computed (p < 0.05)
- [ ] Sample size ≥30 time points

### Common Pitfalls
- ⚠️ Correlating raw data captures seasonal cycle — everything correlates with summer.
- ⚠️ Spatial autocorrelation inflates field significance — apply FDR correction.
- ⚠️ Different models may have different grid resolutions — regrid before cross-model correlation.

### Interpretation
- Plot spatial R maps with `RdBu_r`. Stipple significant areas.
""",

    "composite_analysis": """
## Composite Analysis

### When to use
- Average conditions during El Niño vs La Niña in model output
- Spatial fingerprint of specific events
- Multi-model composite agreement

### Workflow
1. **Define events** — Boolean mask (e.g., Niño3.4 > 0.5°C).
2. **Subset** — `.where(mask, drop=True)`.
3. **Average** — Time mean of subset = composite.
4. **Compare** — Subtract climatology → composite anomaly.

### Quality Checklist
- [ ] Sample size ≥10 events
- [ ] Significance tested (bootstrap or t-test)
- [ ] Multi-model: composite computed per model before averaging

### Common Pitfalls
- ⚠️ Few events (N < 5) → noise, not signal.
- ⚠️ Mixing seasons obscures signal.
""",

    "seasonal_decomposition": """
## Seasonal Decomposition

### When to use
- Separating the seasonal cycle from interannual variability
- Visualizing how a model year deviates from the mean cycle

### Workflow
1. **Compute climatology** — `.groupby('time.month').mean('time')`.
2. **Extract anomalies** — Subtract climatology from raw data.
3. **Smooth trend** — Apply 12-month rolling mean for multi-year trends.

### Quality Checklist
- [ ] Baseline robust (≥20 years of historical data)
- [ ] cftime-aware if non-standard calendar

### Common Pitfalls
- ⚠️ `360_day` calendar: months have 30 days each — beware of `.groupby('time.dayofyear')`.
- ⚠️ Use `.groupby('time.month')` consistently for CMIP6 monthly data.
""",

    "spatial_statistics": """
## Spatial Statistics & Area Averaging

### When to use
- Computing global mean temperature time series
- Area-weighted regional averages
- Model intercomparison of area means

### Workflow
1. **Latitude weights** — `weights = np.cos(np.deg2rad(ds.lat))`.
2. **Apply** — `ds.weighted(weights).mean(dim=['lat', 'lon'])`.
3. **For ocean** — Use `areacello` cell area if available, or mask land with NaN.

### Quality Checklist
- [ ] Latitude weighting applied BEFORE spatial averaging
- [ ] Land/ocean mask applied if computing ocean/land-only means
- [ ] Check if model provides `areacella` / `areacello` for exact cell areas

### Common Pitfalls
- ⚠️ Unweighted averages bias toward poles (smaller cells over-counted).
- ⚠️ Global mean SST must exclude land cells — mask with `tos.where(tos.notnull())`.
- ⚠️ Different coordinate names: some models use `lat`/`lon`, others `latitude`/`longitude`. Generalize with: `lat_name = [c for c in ds.coords if 'lat' in c.lower()][0]`.
""",

    "climatology_normals": """
## Climatology Normals

### When to use
- Computing reference climatologies for comparison
- "Departure from normal" analysis

### Workflow
1. **Select base period** — CMIP6 historical: 1981–2010 or 1850–1900 (pre-industrial).
2. **Monthly climatology** — `normals = baseline.groupby('time.month').mean('time')`.
3. **Departure** — `departure = data.groupby('time.month') - normals`.

### Quality Checklist
- [ ] Pre-industrial baseline: 1850–1900 (for warming assessments)
- [ ] Modern baseline: 1981–2010 or 1991–2020
- [ ] Baseline computed from historical experiment only

### Common Pitfalls
- ⚠️ Using SSP data for baseline computation is wrong — always use historical.
- ⚠️ Pre-industrial baseline (1850–1900) may have model drift in some models.
""",

    "climate_indices": """
## Climate Indices from CMIP6

### When to use
- Computing ENSO, NAO, AMO, PDO from model output
- Comparing index behavior across models/scenarios

### Key Indices
- **ENSO (Niño 3.4)**: `tos` anomaly, 5°S–5°N, 170°W–120°W. El Niño > +0.5°C, La Niña < -0.5°C.
- **NAO**: `psl` difference, Azores High minus Icelandic Low.
- **PDO**: Leading EOF of North Pacific `tos` (north of 20°N).
- **AMO**: Detrended North Atlantic `tos` average (0–60°N, 80°W–0°).
- **AMOC**: Use `msftyz` (meridional overturning streamfunction) at 26.5°N.

### Workflow
1. **Extract region** — Use standard geographic bounds.
2. **Compute anomaly** — Area-averaged, against pre-industrial or 30yr baseline.
3. **Smooth** — 3-to-5 month rolling mean.

### Quality Checklist
- [ ] Standard geographic bounds used
- [ ] Rolling mean applied to filter noise
- [ ] Latitude-weighted area average
- [ ] Per-model index computed, then multi-model spread shown

### Common Pitfalls
- ⚠️ Ocean variable `tos` uses ocean grids — may need regridding for precise box extraction.
- ⚠️ Different models simulate ENSO with different amplitude/frequency — this IS a finding, not a bug.
""",

    "extremes": """
## Extreme Event Analysis

### When to use
- Threshold exceedance frequency in projections
- How extremes change under different scenarios

### Workflow
1. **Daily data required** — Use `table_id='day'` when searching.
2. **Define threshold** — Absolute (e.g., T > 35°C) or percentile (>95th from historical).
3. **Count** — Exceedance days per year.
4. **Compare** — Historical vs SSP scenario frequencies.

### Quality Checklist
- [ ] Using DAILY data (not monthly — monthly hides extremes entirely)
- [ ] Percentile threshold from historical baseline, applied to future
- [ ] Per-model analysis before multi-model summary

### Common Pitfalls
- ⚠️ Monthly data CANNOT detect extremes. You must search for `table_id='day'`.
- ⚠️ Absolute thresholds are region-dependent — prefer percentile-based.
- ⚠️ Daily CMIP6 data is very large — subset spatially and temporally first.
""",

    # -------------------------------------------------------------------------
    # CMIP6-SPECIFIC ANALYSIS
    # -------------------------------------------------------------------------
    "multi_model_comparison": """
## Multi-Model Comparison

### When to use
- Comparing the same variable across multiple CMIP6 models
- Assessing model agreement/disagreement
- Creating multi-model ensemble means

### Workflow
1. **Load multiple models** — Loop through model list, `xr.open_zarr` each.
2. **Regrid to common grid** — Use `xesmf` or select one model's grid and interpolate others.
   Alternatively, compute global/regional means to avoid regridding.
3. **Compute per-model statistic** — Anomaly, trend, mean for each model separately.
4. **Ensemble statistics** — Mean across models, ± 1 std for spread.
5. **Plot** — Individual model lines + ensemble mean (thick) + spread shading.

### Quality Checklist
- [ ] Each model processed INDEPENDENTLY before combining
- [ ] Same variable, experiment, and time period across all models
- [ ] Ensemble mean weighted equally (unless using model weighting scheme)
- [ ] Number of models stated in figure title/caption
- [ ] Individual model lines shown (thin, semi-transparent) behind ensemble mean

### Common Pitfalls
- ⚠️ Models have different grids — you CANNOT simply average raw grids. Regrid or use area means.
- ⚠️ Not all models provide all variables or time periods — handle missing models gracefully.
- ⚠️ Model independence is debatable — some "models" share components. Report which models used.
- ⚠️ For time series: don't just show ensemble mean — always show individual model lines or spread.

### Interpretation
- Narrow spread = high model agreement = robust signal.
- Wide spread = structural uncertainty = less confidence.
- If >80% of models agree on sign, the signal is considered robust (IPCC convention).
""",

    "scenario_comparison": """
## Scenario Comparison (Historical vs SSP)

### When to use
- Comparing future projections under different emission pathways
- "How much warming under SSP2-4.5 vs SSP5-8.5?"
- Fan/spaghetti plots of future trajectories

### Workflow
1. **Load historical** — `experiment_id='historical'` (1850–2014).
2. **Load SSP scenarios** — `experiment_id='ssp126'`, `'ssp245'`, `'ssp370'`, `'ssp585'`.
3. **Concatenate** — `xr.concat([hist, ssp], dim='time')` per model.
4. **Compute anomaly** — Relative to pre-industrial baseline (1850–1900).
5. **Plot** — Time series with scenario-colored bands (SSP1-2.6=blue, SSP2-4.5=orange, SSP5-8.5=red).

### Standard SSP Colors (IPCC Convention)
- SSP1-2.6: blue (#1b9e77)
- SSP2-4.5: orange/amber (#d95f02)
- SSP3-7.0: red (#e7298a)  
- SSP5-8.5: dark red (#7570b3)

### Quality Checklist
- [ ] Historical and SSP from the SAME model and variant (`r1i1p1f1`)
- [ ] Concatenation creates continuous time axis (2014–2015 boundary)
- [ ] Anomalies relative to consistent baseline (1850–1900 pre-industrial)
- [ ] Multiple models per scenario → show spread (shading)

### Common Pitfalls
- ⚠️ Mixing variant labels across historical/SSP creates discontinuities.
- ⚠️ Some models don't provide all SSP scenarios — check availability.
- ⚠️ The historical-SSP boundary (2014–2015) may show a small jump due to different forcings.
- ⚠️ Always label scenarios clearly. Never show unlabeled colored lines.

### Interpretation
- SSP1-2.6 ≈ Paris-compliant (~1.5–2°C). SSP5-8.5 ≈ worst case (~4–5°C by 2100).
- Scenario spread dominates uncertainty after ~2050. Model spread dominates before.
""",

    "ensemble_analysis": """
## Ensemble Analysis (Internal Variability)

### When to use
- Analyzing within-model ensemble spread (r1, r2, ... r10)
- Separating forced signal from internal variability
- "Is this trend robust or just noise?"

### Workflow
1. **Load multiple members** — Same model, same experiment, different `member_id` (r1i1p1f1, r2i1p1f1, ...).
2. **Compute per-member** — Time series or maps for each.
3. **Ensemble mean** — Average across members → forced signal.
4. **Ensemble spread** — Std across members → internal variability.
5. **Signal-to-noise** — `SNR = abs(ensemble_mean) / ensemble_std`.

### Quality Checklist
- [ ] All members from SAME model, experiment, and physics version
- [ ] At least 3 members (preferably ≥5) for meaningful spread
- [ ] SNR computed to assess robustness

### Common Pitfalls
- ⚠️ Small ensembles (N < 3) underestimate internal variability.
- ⚠️ Not all models provide large ensembles. CESM2-LE has ~100, most have 3–10.
- ⚠️ Different `i` values (initialization) may matter for decadal predictions.

### Interpretation
- Ensemble mean = forced response to greenhouse gases.
- Ensemble spread = what would happen due to natural variability alone.
- SNR > 1 = forced signal detectable above noise.
""",

    "model_evaluation": """
## Model Evaluation (Comparison to Observations)

### When to use
- "How well does this model reproduce observed climate?"
- Bias assessment: model minus ERA5/observations
- Taylor diagrams for multi-model skill summary

### Workflow
1. **Load model historical** — Same period as observational reference.
2. **Load reference** — ERA5 reanalysis or observational dataset.
3. **Regrid** — Interpolate to common grid (coarser of the two).
4. **Bias map** — `bias = model_clim - obs_clim`.
5. **Statistics** — RMSE, pattern correlation, bias.

### Quality Checklist
- [ ] Same time period for model and observations
- [ ] Both on compatible grids (regridded)
- [ ] Bias map uses diverging colormap (`RdBu_r`) centered at zero
- [ ] Report area-weighted global mean bias

### Common Pitfalls
- ⚠️ ERA5 is a reanalysis, not pure observations — it has its own biases.
- ⚠️ Model ocean grids often differ from atmospheric grids — be explicit about which you compare.
- ⚠️ Comparing daily extremes requires daily data from both model and reference.

### Interpretation
- Warm bias = model too warm. Cold bias = too cool. Pattern correlation > 0.9 = good skill.
""",

    "cmip6_unit_conventions": """
## CMIP6 Variable Names & Unit Conventions

### When to use
- Before any analysis — check variable names and units
- When converting between CMIP6 and human-readable formats

### Atmosphere Variables (table: Amon)
| Variable | Long Name | Native Unit | Convert To |
|----------|-----------|-------------|------------|
| `tas` | Near-surface air temp | K | −273.15 → °C |
| `tasmax` | Daily max temp | K | −273.15 → °C |
| `tasmin` | Daily min temp | K | −273.15 → °C |
| `pr` | Precipitation rate | kg/m²/s | ×86400 → mm/day |
| `psl` | Sea level pressure | Pa | ÷100 → hPa |
| `hurs` | Near-surface RH | % | no conversion |
| `sfcWind` | Surface wind speed | m/s | no conversion |
| `rsds` | Downwelling SW radiation | W/m² | no conversion |
| `clt` | Total cloud fraction | % | no conversion |

### Ocean Variables (table: Omon)
| Variable | Long Name | Native Unit | Convert To |
|----------|-----------|-------------|------------|
| `tos` | Sea surface temp | °C or K | check `ds.tos.attrs['units']` |
| `sos` | Sea surface salinity | PSU | no conversion |
| `zos` | Sea surface height | m | no conversion |
| `siconc` | Sea ice concentration | % | no conversion |

### Quality Checklist
- [ ] Always check `ds[var].attrs['units']` — conventions vary by model
- [ ] Convert BEFORE any computation (mean, anomaly, etc.)
- [ ] Precipitation: `pr × 86400` = mm/day; `pr × 86400 × 30` ≈ mm/month

### Common Pitfalls
- ⚠️ `tos` is sometimes in K, sometimes in °C depending on the model. CHECK `attrs['units']`.
- ⚠️ `pr` in kg/m²/s is a tiny number (≈ 0.00005). Forgetting to convert makes plots look flat.
- ⚠️ CRITICAL: Temperature DIFFERENCES in K = differences in °C. Do NOT subtract 273.15 from anomalies.
""",

    "warming_levels": """
## Global Warming Levels

### When to use
- "When does this model reach 1.5°C / 2°C / 3°C warming?"
- Time-of-crossing analysis
- Sampling conditions at specific warming levels (pattern scaling)

### Workflow
1. **Global mean temperature** — Area-weighted mean of `tas`: `gmst = tas.weighted(np.cos(np.deg2rad(tas.lat))).mean(dim=['lat','lon'])`.
2. **Annual mean** — `.groupby('time.year').mean('time')`.
3. **Anomaly** — Relative to 1850–1900 pre-industrial baseline.
4. **Smoothing** — 20-year running mean to identify crossing year.
5. **Crossing year** — First year where 20yr mean exceeds threshold.

### Quality Checklist
- [ ] Baseline: 1850–1900 (IPCC standard for pre-industrial)
- [ ] 20-year running mean for smoothed crossing (avoids single-year noise)
- [ ] Historical + SSP concatenated for continuous record
- [ ] Report crossing year ± spread across models

### Common Pitfalls
- ⚠️ Annual global mean temperature has ±0.2°C interannual variability — don't use single-year crossing.
- ⚠️ Low-emission scenarios (SSP1-2.6) may never reach 2°C — handle gracefully.
- ⚠️ Different ECS (equilibrium climate sensitivity) across models → wide spread in crossing years.

### Interpretation
- 1.5°C: ~2030 (most models, most scenarios). 2°C: ~2040–2060 (scenario dependent).
- Same warming level ≠ same impacts — regional patterns differ.
""",

    # -------------------------------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------------------------------
    "visualization_spatial": """
## Spatial Map Visualization

### When to use
- Mapping absolute climate fields (temperature, wind, precipitation, pressure)

### Workflow
1. **Figure** — `fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})`.
2. **Plot** — `ax.pcolormesh(lon, lat, data, cmap=..., shading='auto', transform=ccrs.PlateCarree())`.
3. **Coastlines** — `ax.coastlines(linewidth=0.5)` + `ax.add_feature(cfeature.BORDERS, linewidth=0.3)`.
4. **Colorbar** — ALWAYS: `plt.colorbar(mesh, ax=ax, label='Units', shrink=0.8)`.
5. **Title** — Include model name, experiment, period.

### Colormap Standards
- Temperature: `RdYlBu_r`
- Precipitation: `YlGnBu`
- Wind speed: `YlOrRd`
- Pressure: `viridis`
- Cloud cover: `Blues`
- Anomalies: `RdBu_r` (ALWAYS with `TwoSlopeNorm`)

### Quality Checklist
- [ ] Figure 12×8 for maps
- [ ] CARTOPY MANDATORY: `ccrs` projection + coastlines + borders
- [ ] `transform=ccrs.PlateCarree()` passed to plot call
- [ ] NEVER use `jet` colormap
- [ ] Colorbar has label with units
- [ ] Title includes: variable, model, experiment, period

### Common Pitfalls
- ⚠️ Diverging colormap on absolute data is misleading — diverging ONLY for anomalies.
- ⚠️ Ocean curvilinear grids need `pcolormesh(nav_lon, nav_lat, data)` not simple 1D coords.
- ⚠️ `plt.contourf` may fail on NaN-heavy ocean data — use `pcolormesh` instead.
""",

    "visualization_timeseries": """
## Time Series Visualization

### When to use
- Temporal evolution of global/regional mean
- Multi-model spaghetti plots
- Scenario fan plots

### Workflow
1. **Area average** — With latitude weighting.
2. **Figure** — `fig, ax = plt.subplots(figsize=(12, 6))`.
3. **Individual models** — Thin lines (`linewidth=0.8, alpha=0.4`).
4. **Ensemble mean** — Thick line (`linewidth=2.5`).
5. **Spread** — `ax.fill_between(time, mean-std, mean+std, alpha=0.2)`.
6. **Date formatting** — `fig.autofmt_xdate(rotation=30)`.

### Quality Checklist
- [ ] Figure 12×6 minimum
- [ ] Y-axis has explicit units
- [ ] Legend included
- [ ] Grid lines: `ax.grid(True, alpha=0.3)`
- [ ] If showing scenarios: use IPCC standard colors
- [ ] If showing trend: dashed line with slope annotation (°C/decade)
- [ ] Title includes variable, region/scope

### Common Pitfalls
- ⚠️ Only showing ensemble mean hides model uncertainty. ALWAYS show individual lines or spread.
- ⚠️ `cftime` dates may not work with `matplotlib` directly — convert to float years.
- ⚠️ Y-axis too narrow exaggerates variability. Keep ranges physically reasonable.
""",

    "visualization_anomaly_map": """
## Anomaly Map Visualization

### When to use
- Diverging data: departures, trends, z-scores, biases
- Any map with positive AND negative values

### Workflow
1. **Center at zero** — `from matplotlib.colors import TwoSlopeNorm`.
2. **Robust limits** — `vmax = np.nanpercentile(np.abs(data), 98)`, `vmin = -vmax`.
3. **Plot** — `pcolormesh(..., cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))`.
4. **Stippling** — Overlay significance: `ax.contourf(..., levels=[0, 0.05], hatches=['...'], colors='none')`.

### Quality Checklist
- [ ] Zero is EXACTLY white/neutral in the colorbar
- [ ] ROBUST limits: use 2nd–98th percentile, NOT raw min/max
- [ ] Warm = Red, Cool = Blue
- [ ] Precipitation anomalies: use `BrBG` instead of `RdBu_r`
- [ ] CARTOPY MANDATORY: coastlines + borders

### Common Pitfalls
- ⚠️ Without `TwoSlopeNorm`, skewed data makes 0 appear colored → misleading.
- ⚠️ NEVER use `data.min()`/`data.max()` for limits — one outlier cell makes the whole map unreadable.
- ⚠️ Always `transform=ccrs.PlateCarree()` with Cartopy.
""",

    "visualization_comparison": """
## Multi-Panel Comparison

### When to use
- Model A vs Model B vs Difference
- Historical vs SSP scenario vs Change
- Multi-variable side-by-side

### Workflow
1. **Grid** — `fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})`.
2. **Panels 1 & 2** — Absolute values with SHARED `vmin`/`vmax`.
3. **Panel 3** — Difference with diverging colormap centered at zero.
4. **Coastlines** — Add to ALL panels.

### Quality Checklist
- [ ] Panels 1 & 2 share EXACT same vmin/vmax
- [ ] Panel 3 has its own divergent colorbar centered at zero
- [ ] Titles clearly label each panel (model name, period)
- [ ] `transform=ccrs.PlateCarree()` on all plot calls

### Common Pitfalls
- ⚠️ Auto-scaled panels = impossible to compare visually. ALWAYS lock limits.
- ⚠️ Cartopy projection ONLY on map panels. Never on timeseries/histogram subplots.
""",

    "visualization_distribution": """
## Distribution Visualization

### When to use
- Histograms, PDFs, box plots
- Comparing distributions: historical vs future, model A vs model B

### Workflow
1. **Flatten** — `.values.flatten()`, drop NaNs.
2. **Shared bins** — `np.linspace(min, max, 50)`.
3. **Plot** — `ax.hist(data, bins=bins, alpha=0.5, density=True, label='Period')`.
4. **Median/mean markers** — Vertical lines with annotation.

### Quality Checklist
- [ ] `density=True` for comparable distributions of different sizes
- [ ] `alpha=0.5` for overlapping
- [ ] Legend when comparing
- [ ] Figure 10×6

### Common Pitfalls
- ⚠️ Raw counts (not density) skew comparison between different sample sizes.
- ⚠️ 30–50 bins usually optimal.

### Interpretation
- Rightward shift = warming. Wider = more variability = more extremes.
""",

    "visualization_dashboard": """
## Summary Dashboard

### When to use
- Comprehensive overview: map + time series + statistics in one figure
- Publication-ready model comparison summaries

### Workflow
1. **Layout** — `fig = plt.figure(figsize=(16, 10))` + `matplotlib.gridspec`.
2. **Top row** — Spatial map (use Cartopy projection).
3. **Bottom left** — Time series of regional/global mean.
4. **Bottom right** — Distribution histogram or box plot.

### Quality Checklist
- [ ] `plt.tight_layout()` or `constrained_layout=True`
- [ ] Panel labels (a, b, c) for reference
- [ ] Consistent color theme

### Common Pitfalls
- ⚠️ MIXED PROJECTION: Cartopy projections must ONLY be applied to map axes. Time series/histogram panels must NOT use `projection=ccrs.PlateCarree()`.
- ⚠️ Use `fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())` ONLY for map panels.
""",

    "visualization_contour": """
## Contour & Isobar Plots

### When to use
- Pressure maps with isobars
- Temperature isotherms
- Smoothly varying fields

### Workflow
1. **Define levels** — `levels = np.arange(990, 1040, 4)` for MSLP.
2. **Filled contour** — `ax.contourf(lon, lat, data, levels=levels, cmap=..., transform=ccrs.PlateCarree())`.
3. **Contour lines** — `cs = ax.contour(..., colors='black', linewidths=0.5)`.
4. **Labels** — `ax.clabel(cs, inline=True, fontsize=8)`.

### Quality Checklist
- [ ] 10–15 levels max
- [ ] CARTOPY MANDATORY
- [ ] `transform=ccrs.PlateCarree()` on all calls

### Common Pitfalls
- ⚠️ Too many levels → cluttered. Too few → no detail.
- ⚠️ `contourf` may fail on NaN-heavy ocean grids — mask or use `pcolormesh`.
""",

    # -------------------------------------------------------------------------
    # CRITICAL SAFETY GUIDES (from hallucination audit)
    # -------------------------------------------------------------------------
    "bias_correction_qdm": """
## Quantile Delta Mapping (QDM) Bias Correction

### CRITICAL WARNING: NON-STATIONARITY & GUARDRAIL HACKING
- ⚠️ NEVER compute the non-exceedance probability (`tau`) for a future value by sorting the *entire 100-year future array*. This assumes the future climate is stationary and completely erases the climate change delta.
- QDM must detrend the future series first, or chunk the future data into stationary windows (e.g., 30-year slices).
- ⚠️ NEVER use a scalar hack (e.g. `data -= bias`) to force historical baselines to match ERA5. If your baseline has a large residual error, your QDM algorithm logic is mathematically flawed. Fix the logic.
- ⚠️ Always verify the corrected output against ERA5 for the calibration period. If the bias-corrected historical does not match ERA5 within reasonable tolerance, the method has failed.

### Workflow
1. **Split future** into 30-year windows or detrend before computing quantiles.
2. **Map quantiles** from historical model → historical observed (ERA5).
3. **Apply delta** to future values preserving the change signal.
4. **Validate** by checking bias-corrected historical against ERA5.

### Quality Checklist
- [ ] Future quantiles NOT computed over the entire 2015–2100 period
- [ ] No scalar shifts applied to force baselines
- [ ] Bias-corrected historical verified against ERA5
- [ ] Monotonicity of quantile mapping preserved
""",

    "geospatial_gradients": """
## Computing Geospatial Gradients

### CRITICAL WARNING: NO IMAGE FILTERS
- ⚠️ NEVER use `scipy.ndimage.sobel`, `np.gradient`, `cv2.Sobel`, or `skimage` filters directly on lat/lon grids.
- Grid cells shrink as they approach the poles. Image filters calculate °C/pixel, creating massive artificial gradients at high latitudes.
- **Fix:** Calculate physical distances (dx, dy in meters) using the Earth's radius before computing gradients.

### Correct Approach
Use the pre-loaded `lonlat_gradient_magnitude()` helper or compute manually:
```
meters_per_deg_lat = np.pi * 6_371_000 / 180
meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(lat))
d_dx = da.differentiate('lon') / meters_per_deg_lon
d_dy = da.differentiate('lat') / meters_per_deg_lat
grad_mag = np.hypot(d_dx, d_dy)
```

### For Curvilinear/Unstructured Grids
- Either regrid to a regular grid FIRST, then compute gradients
- Or use native model metric fields (e.g., `dxC`, `dyC` for MOM6)
- NEVER apply index-space operations to raw ocean grids

### Quality Checklist
- [ ] Gradients computed in physical space (meters), not index space
- [ ] No image-processing libraries used on geophysical fields
- [ ] Curvilinear grids regridded before gradient computation
""",

    "spatial_statistics_advanced": """
## Advanced Spatial Statistics: Denominator Mismatch

### CRITICAL WARNING: MASK/WEIGHT ALIGNMENT
- ⚠️ If you mask data using `.where(mask)` or set values to NaN, you MUST apply the exact same mask to your `weights` array BEFORE calling `weighted(weights).mean()`.
- ⚠️ If you fail to mask the weights, xarray will correctly ignore NaNs in the numerator, but will sum the ENTIRE global area in the denominator. Your regional RMSE/means will be artificially tiny.

### Correct Approach
Use the pre-loaded `aligned_weighted_mean()` helper or:
```
weights_masked = weights.where(valid_mask)
result = (error * weights_masked).sum() / weights_masked.sum()
```

### DO NOT
```
# WRONG: weights cover entire globe, data covers only Texas
result = data.weighted(global_weights).mean(dim=['lat','lon'])
```

### Quality Checklist
- [ ] Weights masked identically to data
- [ ] Denominator verified > 0 after masking
- [ ] Domain extents identical for data and weights
- [ ] NaN fraction checked and reported
""",
}


# =============================================================================
# TOPIC CATALOG — one-line descriptions for agent routing
# =============================================================================

TOPIC_CATALOG = {
    # Data operations
    "load_data":            "Open CMIP6 Zarr stores from GCS, inspect coords, convert units, handle calendars",
    "spatial_subset":       "Slice by lat/lon region, handle curvilinear ocean grids, longitude wrapping",
    "temporal_subset":      "Time slicing, seasonal filtering, annual/seasonal aggregation with cftime",
    # Statistical analysis
    "anomalies":            "Compute departures from climatological baseline (per-model, multi-model)",
    "zscore":               "Standardized anomalies for cross-variable/cross-region comparison",
    "trend_analysis":       "Linear trends with significance testing, per-decade scaling",
    "eof_analysis":         "EOF/PCA for dominant spatial modes (ENSO, NAO, PDO-like patterns)",
    # Advanced analysis
    "correlation_analysis": "Spatial/temporal correlation maps with significance masking",
    "composite_analysis":   "Event composites (El Niño / La Niña) with bootstrap significance",
    "seasonal_decomposition": "Separate seasonal cycle from interannual variability",
    "spatial_statistics":   "Area-weighted global/regional means with latitude weighting",
    "climatology_normals":  "Compute 1981–2010 or 1850–1900 reference climatologies",
    # Climate indices & extremes
    "climate_indices":      "ENSO Niño3.4, NAO, PDO, AMO, AMOC from model output",
    "extremes":             "Threshold exceedance frequency from daily data, percentile-based",
    # CMIP6-specific
    "multi_model_comparison":  "Compare variable across models, regridding, ensemble mean ± spread",
    "scenario_comparison":     "Historical vs SSP fan plots with IPCC standard colors",
    "ensemble_analysis":       "Within-model ensemble spread, signal-to-noise ratio",
    "model_evaluation":        "Bias maps vs ERA5/observations, RMSE, pattern correlation",
    "cmip6_unit_conventions":  "Variable names, native units, conversion factors (K→°C, kg/m²/s→mm/day)",
    "warming_levels":          "Global warming level crossing year (1.5°C, 2°C, 3°C) with 20yr smoothing",
    # Visualization
    "visualization_spatial":      "Cartopy maps with proper colormaps, coastlines, colorbars",
    "visualization_timeseries":   "Time series with ensemble spread, model lines, grid, legends",
    "visualization_anomaly_map":  "Diverging maps centered at zero with TwoSlopeNorm, stippling",
    "visualization_comparison":   "Multi-panel side-by-side with shared/independent color scales",
    "visualization_distribution": "Histograms, PDFs, box plots for comparing distributions",
    "visualization_dashboard":    "Multi-panel dashboard: map + timeseries + distribution in one figure",
    "visualization_contour":      "Contour/isobar plots with labeled contour lines",
    # Safety guides
    "bias_correction_qdm":         "QDM non-stationarity warnings, guardrail hacking detection, validation protocol",
    "geospatial_gradients":        "Physical-space gradients — ban sobel/cv2, use Earth-radius scaling",
    "spatial_statistics_advanced": "Denominator mismatch, mask/weight alignment, weighted mean safety",
}

_ALL_TOPICS = list(TOPIC_CATALOG.keys())


# =============================================================================
# ARGUMENT SCHEMA
# =============================================================================

class AnalysisGuideArgs(BaseModel):
    """Arguments for CMIP6 analysis guide retrieval."""

    topics: List[Literal[
        # Data operations
        "load_data",
        "spatial_subset",
        "temporal_subset",
        # Statistical analysis
        "anomalies",
        "zscore",
        "trend_analysis",
        "eof_analysis",
        # Advanced analysis
        "correlation_analysis",
        "composite_analysis",
        "seasonal_decomposition",
        "spatial_statistics",
        "climatology_normals",
        # Climate indices & extremes
        "climate_indices",
        "extremes",
        # CMIP6-specific
        "multi_model_comparison",
        "scenario_comparison",
        "ensemble_analysis",
        "model_evaluation",
        "cmip6_unit_conventions",
        "warming_levels",
        # Visualization
        "visualization_spatial",
        "visualization_timeseries",
        "visualization_anomaly_map",
        "visualization_comparison",
        "visualization_distribution",
        "visualization_dashboard",
        "visualization_contour",
        # Safety guides
        "bias_correction_qdm",
        "geospatial_gradients",
        "spatial_statistics_advanced",
    ]] = Field(
        description=(
            "One or more analysis topics to retrieve guides for. "
            "Pass multiple topics to get combined guidance in a single call. "
            "Example: ['trend_analysis', 'visualization_spatial'] for a trend map task."
        )
    )


# =============================================================================
# TOOL FUNCTION
# =============================================================================

def get_analysis_guide(topics) -> str:
    """
    Get methodological guidance for CMIP6 climate data analysis.

    Accepts one or more topics and returns combined guidance.
    """
    # Normalize: accept a single string or a list
    if isinstance(topics, str):
        topics = [topics]

    sections = []
    unknown = []

    for topic in topics:
        guide = ANALYSIS_GUIDES.get(topic)
        if guide:
            sections.append(
                f"# CMIP6 Analysis Guide: {topic.replace('_', ' ').title()}\n{guide}"
            )
        else:
            unknown.append(topic)

    if not sections and unknown:
        catalog_lines = "\n".join(f"  • {k}: {v}" for k, v in TOPIC_CATALOG.items())
        return f"Unknown topic(s): {', '.join(unknown)}.\n\nAvailable topics:\n{catalog_lines}"

    result = "\n\n---\n\n".join(sections)

    if unknown:
        result += f"\n\n⚠️ Unknown topic(s) skipped: {', '.join(unknown)}"

    result += "\n\n---\nUse python_repl to implement this analysis with your CMIP6 data."
    return result


# =============================================================================
# TOOL DESCRIPTION — includes full topic catalog for agent routing
# =============================================================================

_catalog_text = "\n".join(f"    • {k}: {v}" for k, v in TOPIC_CATALOG.items())

_TOOL_DESCRIPTION = f"""Get methodological guidance for CMIP6 climate data analysis.

Pass ONE OR MORE topics to get combined workflow steps, quality checklists, and pitfall warnings.
For complex tasks, combine topics (e.g. ['trend_analysis', 'visualization_spatial'] for a trend map).

TOPIC CATALOG:
{_catalog_text}

CALL THIS BEFORE writing analysis or visualization code in python_repl."""


# =============================================================================
# TOOL DEFINITION
# =============================================================================

analysis_guide_tool = StructuredTool.from_function(
    func=get_analysis_guide,
    name="get_analysis_guide",
    description=_TOOL_DESCRIPTION,
    args_schema=AnalysisGuideArgs,
)
