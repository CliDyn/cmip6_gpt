from pydantic import BaseModel, Field, create_model
from typing import Optional, Literal, List, Dict
from src.config import Config
import json
import re
class CMIP6DownloadArgs(BaseModel):
    source_id: Optional[str] = Field(default=None, description="The model name in CMIP6")
    experiment_id: Optional[str] = Field(default=None, description="Experiment identifier")
    variable_id: Optional[str] = Field(default=None, description="Variable identifier")

    frequency: Optional[Literal[
        "1hr", "1hrCM", "1hrPt", "3hr", "3hrPt", "6hr", "6hrPt", "day", "dec", "fx",
        "mon", "monC", "monPt", "subhrPt", "yr", "yrPt"
    ]] = Field(
        default=None,
        description="Time frequency of the data with: high (sub-hourly to daily data: “subhrPt”, “1hr”, “1hrCM”, “1hrPt”, “3hr”, “3hrPt”, “6hr”, “6hrPt”, “day”), Medium (monthly data and related climatologies: “mon”, “monC”, “monPt”) and low (annual, decadal, or fixed data: “yr”, “yrPt”, “dec”, “fx”)  frequencies",
        enum_descriptions={
            "1hr": "Sampled hourly",
            "1hrCM": "Monthly-mean diurnal cycle resolving each day into 1-hour means",
            "1hrPt": "Sampled hourly, at specified time point within an hour",
            "3hr": "3 hourly mean samples",
            "3hrPt": "Sampled 3 hourly, at specified time point within the time period",
            "6hr": "6 hourly mean samples",
            "6hrPt": "Sampled 6 hourly, at specified time point within the time period",
            "day": "Daily mean samples",
            "dec": "Decadal mean samples",
            "fx": "Fixed (time invariant) field",
            "mon": "Monthly mean samples",
            "monC": "Monthly climatology computed from monthly mean samples",
            "monPt": "Sampled monthly, at specified time point within the time period",
            "subhrPt": "Sampled sub-hourly, at specified time point within an hour",
            "yr": "Annual mean samples",
            "yrPt": "Sampled yearly, at specified time point within the time period"
        }
    )
    nominal_resolution: Optional[Literal[
        "100 km", "250 km", "500 km", "50 km", "1x1 degree", "200 km",
        "25 km", "10000 km", "10 km", "2x2 degree"
    ]] = Field(default=None, description="Spatial resolution of the data inlcuding high (10km, 25km), medium (50km, 100km, 1x1 degree, 200km, 2x2 degree) and low (250km, 500km, 10000km) resolutions")
    variant_label: str = Field(default="r1i1p1f1", description="Variant label for the dataset : r<i>i<p>p<f>f, r1 to rX: Different realizations (often ranging from r1 to r10 or higher, depending on the ensemble size). i1 to iY: Different initialization procedures (usually i1 to i2 or i3). p1 to pZ: Different physics schemes (typically p1 to p2 or p3).f1 to fN: Different forcing versions (often f1 to f2 or f3).")
    sub_experiment_id: Optional[str] = Field(
        default=None,
        description="Sub-experiment identifier. Values range from 's1910' to 's2029', representing the year near the end of which the experiment was initialized. For example, 's1950' means the experiment was initialized near the end of 1950. Use 'none' if not applicable. This field is used in climate model experiments to specify different initialization times for ensemble runs."
    )
    activity_id: Optional[Literal[
        "AerChemMIP", "C4MIP", "CDRMIP", "CFMIP", "CMIP", "CORDEX", "DAMIP",
        "DCPP", "DynVarMIP", "FAFMIP", "GMMIP", "GeoMIP", "HighResMIP", "ISMIP6",
        "LS3MIP", "LUMIP", "OMIP", "PAMIP", "PMIP", "RFMIP", "SIMIP", "ScenarioMIP",
        "VIACSAB", "VolMIP"
    ]] = Field(default=None, description="Activity identifier for the CMIP6 project", enum_descriptions={
        "AerChemMIP": "Aerosols and Chemistry Model Intercomparison Project",
        "C4MIP": "Coupled Climate Carbon Cycle Model Intercomparison Project",
        "CDRMIP": "Carbon Dioxide Removal Model Intercomparison Project",
        "CFMIP": "Cloud Feedback Model Intercomparison Project",
        "CMIP": "CMIP DECK: 1pctCO2, abrupt4xCO2, amip, esm-piControl, esm-historical, historical, and piControl experiments",
        "CORDEX": "Coordinated Regional Climate Downscaling Experiment",
        "DAMIP": "Detection and Attribution Model Intercomparison Project",
        "DCPP": "Decadal Climate Prediction Project",
        "DynVarMIP": "Dynamics and Variability Model Intercomparison Project",
        "FAFMIP": "Flux-Anomaly-Forced Model Intercomparison Project",
        "GMMIP": "Global Monsoons Model Intercomparison Project",
        "GeoMIP": "Geoengineering Model Intercomparison Project",
        "HighResMIP": "High-Resolution Model Intercomparison Project",
        "ISMIP6": "Ice Sheet Model Intercomparison Project for CMIP6",
        "LS3MIP": "Land Surface, Snow and Soil Moisture",
        "LUMIP": "Land-Use Model Intercomparison Project",
        "OMIP": "Ocean Model Intercomparison Project",
        "PAMIP": "Polar Amplification Model Intercomparison Project",
        "PMIP": "Palaeoclimate Modelling Intercomparison Project",
        "RFMIP": "Radiative Forcing Model Intercomparison Project",
        "SIMIP": "Sea Ice Model Intercomparison Project",
        "ScenarioMIP": "Scenario Model Intercomparison Project",
        "VIACSAB": "Vulnerability, Impacts, Adaptation and Climate Services Advisory Board",
        "VolMIP": "Volcanic Forcings Model Intercomparison Project"
    })
    realm: Optional[Literal[
        "aerosol", "atmos", "atmosChem", "land", "landIce", "ocean", "ocnBgchem", "seaIce"
    ]] = Field(default=None, description="Realm of the climate system")
    institution_id: Optional[Literal["AER","AS-RCEC","AWI","BCC","CAMS", "CAS","CCCR-IITM","CCCma",
    "CMCC","CNRM-CERFACS","CSIRO","CSIRO-ARCCSS","CSIRO-COSIMA","DKRZ","DWD","E3SM-Project","EC-Earth-Consortium","ECMWF",
    "FIO-QLNM","HAMMOZ-Consortium","INM","IPSL","KIOST","LLNL","MESSy-Consortium","MIROC","MOHC",
    "MPI-M","MRI","NASA-GISS","NASA-GSFC","NCAR","NCC","NERC","NIMS-KMA","NIWA","NOAA-GFDL",
    "NTU","NUIST","PCMDI","PNNL-WACCEM","RTE-RRTMGP-Consortium","RUBISCO","SNU","THU",
    "UA","UCI","UCSB","UHH"
    ]] = Field(default=None, description="Institution identifier for the CMIP6 project", enum_descriptions={
        "AER": "Atmospheric and Environmental Research, USA",
        "AS-RCEC": "Academia Sinica, Taiwan",
        "AWI": "Alfred Wegener Institute, Germany",
        "BCC": "Beijing Climate Center, China",
        "CAMS": "Chinese Academy of Meteorological Sciences",
        "CAS": "Chinese Academy of Sciences",
        "CCCR-IITM": "Indian Institute of Tropical Meteorology",
        "CCCma": "Canadian Centre for Climate Modelling and Analysis",
        "CMCC": "Centro Euro-Mediterraneo sui Cambiamenti Climatici, Italy",
        "CNRM-CERFACS": "CNRM and CERFACS, France",
        "CSIRO": "CSIRO, Australia",
        "CSIRO-ARCCSS": "CSIRO and ARC Centre of Excellence, Australia",
        "CSIRO-COSIMA": "CSIRO and COSIMA, Australia",
        "DKRZ": "Deutsches Klimarechenzentrum, Germany",
        "DWD": "Deutscher Wetterdienst, Germany",
        "E3SM-Project": "DOE E3SM multi-lab consortium, USA",
        "EC-Earth-Consortium": "EC-Earth European consortium",
        "ECMWF": "European Centre for Medium-Range Weather Forecasts, UK",
        "FIO-QLNM": "First Institute of Oceanography, China",
        "HAMMOZ-Consortium": "HAMMOZ European consortium",
        "INM": "Institute for Numerical Mathematics, Russia",
        "IPSL": "Institut Pierre Simon Laplace, France",
        "KIOST": "Korea Institute of Ocean Science and Technology",
        "LLNL": "Lawrence Livermore National Laboratory, USA",
        "MESSy-Consortium": "MESSy/DLR consortium, Germany",
        "MIROC": "JAMSTEC/AORI/NIES/R-CCS consortium, Japan",
        "MOHC": "Met Office Hadley Centre, UK",
        "MPI-M": "Max Planck Institute for Meteorology, Germany",
        "MRI": "Meteorological Research Institute, Japan",
        "NASA-GISS": "NASA Goddard Institute for Space Studies, USA",
        "NASA-GSFC": "NASA Goddard Space Flight Center, USA",
        "NCAR": "National Center for Atmospheric Research, USA",
        "NCC": "NorESM Climate modeling Consortium, Norway",
        "NERC": "Natural Environment Research Council, UK",
        "NIMS-KMA": "National Institute of Meteorological Sciences, Korea",
        "NIWA": "National Institute of Water and Atmospheric Research, NZ",
        "NOAA-GFDL": "NOAA Geophysical Fluid Dynamics Laboratory, USA",
        "NTU": "National Taiwan University",
        "NUIST": "Nanjing University of Information Science, China",
        "PCMDI": "Program for Climate Model Diagnosis, LLNL, USA",
        "PNNL-WACCEM": "Pacific Northwest National Laboratory, USA",
        "RTE-RRTMGP-Consortium": "AER and University of Colorado, USA",
        "RUBISCO": "ORNL multi-lab consortium, USA",
        "SNU": "Seoul National University, Korea",
        "THU": "Tsinghua University, China",
        "UA": "University of Arizona, USA",
        "UCI": "University of California Irvine, USA",
        "UCSB": "University of California Santa Barbara, USA",
        "UHH": "Universität Hamburg, Germany"
    })

# Define the dynamic CMIP6DownloadArgs creation function
def create_dynamic_cmip6_args(
    relevant_facets: List[str],
    vector_search_results: Dict[str, List] = None,
    score_threshold: float = None,
    max_candidates: int = None
):
    """
    Dynamically creates a schema for CMIP6 download arguments based on relevant facets and vector search results.

    This function processes the relevant facets from a CMIP6 query and, if vector search results are available, 
    incorporates the top matches for specific facets (e.g., source_id, variable_id, experiment_id). 
    Low-relevance candidates are filtered out using a score threshold (ChromaDB L2 distance: lower = better).
    For each facet, it either uses default fields from the CMIP6 schema or dynamically generates options 
    based on search results, including descriptions.

    Each dynamic Literal includes an "UNMATCHED" escape value so the LLM can indicate that none of the
    RAG candidates match the user's intent, preventing forced hallucination.

    Args:
        relevant_facets (List[str]): List of facets relevant to the CMIP6 query.
        vector_search_results (Dict[str, List], optional): Vector search results containing relevant data for facets.
        score_threshold (float): Maximum L2 distance to accept a RAG candidate (default from config).
            Lower values are stricter. Tuned for gemini-embedding-001.
        max_candidates (int): Maximum number of candidates to include in the dynamic schema (default from config).

    Returns:
        DynamicCMIP6DownloadArgs: A dynamically generated model schema for CMIP6 download arguments.
    """
    MIN_CANDIDATES = 3  # Adaptive fallback: always keep at least this many

    # Resolve defaults from config
    if score_threshold is None:
        score_threshold = Config.get_rag_score_threshold()
    if max_candidates is None:
        max_candidates = Config.get_rag_max_candidates()

    print("\n--- CREATING DYNAMIC CMIP6 DOWNLOAD ARGS ---")
    print(f"Relevant facets: {relevant_facets}")
    print(f"Score threshold: {score_threshold}, Max candidates: {max_candidates}")
    print(f"Vector search results available: {bool(vector_search_results)}")
    print(f"Vector search results keys: {vector_search_results.keys() if vector_search_results else None}")

    dynamic_fields = {}
    for facet in relevant_facets:
        print(f"\nProcessing facet: {facet}")
        # Skip facets not defined in the base schema
        if facet not in CMIP6DownloadArgs.model_fields and facet not in ["source_id", "variable_id", "experiment_id"]:
            print(f"  ⚠ Skipping unknown facet: {facet}")
            continue
        if facet in ["source_id", "variable_id", "experiment_id"] and vector_search_results and facet in vector_search_results:
            # --- Score filtering: keep only relevant candidates ---
            raw_results = vector_search_results[facet]
            # Sort by score ascending (lower distance = more relevant)
            sorted_results = sorted(raw_results, key=lambda r: r.get('score', float('inf')))

            # Apply threshold filter
            filtered = [r for r in sorted_results if r.get('score', float('inf')) <= score_threshold]

            # Adaptive fallback: if too few pass threshold, take top MIN_CANDIDATES regardless
            if len(filtered) < MIN_CANDIDATES:
                filtered = sorted_results[:MIN_CANDIDATES]
                print(f"  Adaptive fallback: only {len([r for r in sorted_results if r.get('score', float('inf')) <= score_threshold])} "
                      f"passed threshold {score_threshold}, using top {MIN_CANDIDATES} instead")

            # Cap at max_candidates
            filtered = filtered[:max_candidates]

            print(f"  Score filtering for {facet}: {len(raw_results)} raw → {len(filtered)} after filter "
                  f"(threshold={score_threshold}, max={max_candidates})")
            if filtered:
                print(f"  Score range: {filtered[0].get('score', '?'):.4f} – {filtered[-1].get('score', '?'):.4f}")

            top_names = []
            descriptions = []
            for result in filtered:
                content = result['content']
                source = result['metadata']['source']
                score = result.get('score', None)
                name = source
                if name and name not in top_names:  # Deduplicate
                    top_names.append(name)
                    descriptions.append(re.sub(r'^.*?: ', '', content))
                    print(f"  ✓ {name} (score: {score:.4f})" if score is not None else f"  ✓ {name}")

            # Log rejected candidates for transparency
            rejected_count = len(raw_results) - len(filtered)
            if rejected_count > 0:
                print(f"  ✗ {rejected_count} candidates rejected (score > {score_threshold})")

            if top_names:
                # Add UNMATCHED escape hatch: allows LLM to indicate none of the
                # RAG candidates match the user's intent, instead of forcing a pick
                options = tuple(top_names) + ("UNMATCHED",)
                desc_dict = dict(zip(top_names, descriptions))
                desc_dict["UNMATCHED"] = "Select this if NONE of the above options match the user's query"
                dynamic_fields[facet] = (Optional[Literal[options]], Field(
                    default=None,
                    description=f"Top {len(top_names)} {facet} matches (score-filtered, threshold={score_threshold}). Select UNMATCHED if none fit.",
                    enum_descriptions=desc_dict
                ))
                print(f"\nDynamic field created for {facet}: {len(top_names)} options + UNMATCHED")
            else:
                print(f"No valid matches found for {facet}, using default field")
                if facet in CMIP6DownloadArgs.model_fields:
                    dynamic_fields[facet] = (
                        CMIP6DownloadArgs.model_fields[facet].annotation, CMIP6DownloadArgs.model_fields[facet])
        else:
            if facet in CMIP6DownloadArgs.model_fields:
                print(f"Using default field for {facet}")
                dynamic_fields[facet] = (
                    CMIP6DownloadArgs.model_fields[facet].annotation, CMIP6DownloadArgs.model_fields[facet])
            else:
                print(f"  ⚠ Skipping unknown facet: {facet}")

    DynamicCMIP6DownloadArgs = create_model("DynamicCMIP6DownloadArgs", **dynamic_fields)

    print("\nDynamic CMIP6DownloadArgs Schema:")
    print(json.dumps(DynamicCMIP6DownloadArgs.model_json_schema(), indent=2))

    print("--- END CREATING DYNAMIC CMIP6 DOWNLOAD ARGS ---")
    return DynamicCMIP6DownloadArgs