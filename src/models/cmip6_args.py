from langchain.pydantic_v1 import BaseModel, Field, create_model
from typing import Optional, Literal, List, Dict
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
        description="Time frequency of the data",
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
    ]] = Field(default=None, description="Spatial resolution of the data")
    variant_label: str = Field(default="r1i1p1f1", description="Variant label for the dataset")
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

# Define the dynamic CMIP6DownloadArgs creation function
def create_dynamic_cmip6_args(relevant_facets: List[str], vector_search_results: Dict[str, List] = None):
    """
    Dynamically creates a schema for CMIP6 download arguments based on relevant facets and vector search results.

    This function processes the relevant facets from a CMIP6 query and, if vector search results are available, 
    incorporates the top matches for specific facets (e.g., source_id, variable_id, experiment_id). 
    For each facet, it either uses default fields from the CMIP6 schema or dynamically generates options 
    based on search results, including descriptions.

    Args:
        relevant_facets (List[str]): List of facets relevant to the CMIP6 query.
        vector_search_results (Dict[str, List], optional): Vector search results containing relevant data for facets.

    Returns:
        DynamicCMIP6DownloadArgs: A dynamically generated model schema for CMIP6 download arguments.
    """
    print("\n--- CREATING DYNAMIC CMIP6 DOWNLOAD ARGS ---")
    print(f"Relevant facets: {relevant_facets}")
    print(f"Vector search results available: {bool(vector_search_results)}")
    print(f"Vector search results keys: {vector_search_results.keys() if vector_search_results else None}")

    dynamic_fields = {}
    for facet in relevant_facets:
        print(f"\nProcessing facet: {facet}")
        if facet in ["source_id", "variable_id", "experiment_id"] and vector_search_results and facet in vector_search_results:
            top_10 = []
            descriptions = []
            print(f"Vector search results for {facet}:")
            for result in vector_search_results[facet][:10]:
                print(f"  Raw result: {result}")
                content = result['content']
                source = result['metadata']['source']
                name = source
                if name:
                    top_10.append(name)
                    descriptions.append(re.sub(r'^.*?: ', '', content))

            print(f"Extracted top 10 for {facet}: {top_10}")
            print(f"Extracted descriptions for {facet}: {descriptions}")

            if top_10:
                dynamic_fields[facet] = (Optional[Literal[tuple(top_10)]], Field(
                    default=None,
                    description=f"Top 10 {facet} matches",
                    enum_descriptions=dict(zip(top_10, descriptions))
                ))
                print(f"\nDynamic field created for {facet}:")
                print(f"  Top 5 options: {top_10[:5]}")
            else:
                print(f"No valid matches found for {facet}, using default field")
                dynamic_fields[facet] = (
                    CMIP6DownloadArgs.__fields__[facet].outer_type_, CMIP6DownloadArgs.__fields__[facet].field_info)
        else:
            print(f"Using default field for {facet}")
            dynamic_fields[facet] = (
                CMIP6DownloadArgs.__fields__[facet].outer_type_, CMIP6DownloadArgs.__fields__[facet].field_info)

    DynamicCMIP6DownloadArgs = create_model("DynamicCMIP6DownloadArgs", **dynamic_fields)

    print("\nDynamic CMIP6DownloadArgs Schema:")
    print(json.dumps(DynamicCMIP6DownloadArgs.schema(), indent=2))

    print("--- END CREATING DYNAMIC CMIP6 DOWNLOAD ARGS ---")
    return DynamicCMIP6DownloadArgs