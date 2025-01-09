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
    institution_id: Optional[Literal["AER","AS-RCEC","AWI","BCC","CAMS", "CAS","CCCR-IITM","CCCma",
    "CMCC","CNRM-CERFACS","CSIRO","CSIRO-ARCCSS","CSIRO-COSIMA","DKRZ","DWD","E3SM-Project","EC-Earth-Consortium","ECMWF",
    "FIO-QLNM","HAMMOZ-Consortium","INM","IPSL","KIOST","LLNL","MESSy-Consortium","MIROC","MOHC",
    "MPI-M","MRI","NASA-GISS","NASA-GSFC","NCAR","NCC","NERC","NIMS-KMA","NIWA","NOAA-GFDL",
    "NTU","NUIST","PCMDI","PNNL-WACCEM","RTE-RRTMGP-Consortium","RUBISCO","SNU","THU",
    "UA","UCI","UCSB","UHH"
]] = Field(default=None, description="Institution identifier for the CMIP6 project", enum_descriptions={
        "AER":"Research and Climate Group, Atmospheric and Environmental Research, 131 Hartwell Avenue, Lexington, MA 02421, USA",
        "AS-RCEC":"Research Center for Environmental Changes, Academia Sinica, Nankang, Taipei 11529, Taiwan",
        "AWI":"Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research, Am Handelshafen 12, 27570 Bremerhaven, Germany",
        "BCC":"Beijing Climate Center, Beijing 100081, China",
        "CAMS":"Chinese Academy of Meteorological Sciences, Beijing 100081, China",
        "CAS":"Chinese Academy of Sciences, Beijing 100029, China",
        "CCCR-IITM":"Centre for Climate Change Research, Indian Institute of Tropical Meteorology Pune, Maharashtra 411 008, India",
        "CCCma":"Canadian Centre for Climate Modelling and Analysis, Environment and Climate Change Canada, Victoria, BC V8P 5C2, Canada",
        "CMCC":"Fondazione Centro Euro-Mediterraneo sui Cambiamenti Climatici, Lecce 73100, Italy",
        "CNRM-CERFACS":"CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France), CERFACS (Centre Europeen de Recherche et de Formation Avancee en Calcul Scientifique, Toulouse 31057, France)",
        "CSIRO":"Commonwealth Scientific and Industrial Research Organisation, Aspendale, Victoria 3195, Australia",
        "CSIRO-ARCCSS":"CSIRO (Commonwealth Scientific and Industrial Research Organisation, Aspendale, Victoria 3195, Australia), ARCCSS (Australian Research Council Centre of Excellence for Climate System Science). Mailing address: CSIRO, c/o Simon J. Marsland, 107-121 Station Street, Aspendale, Victoria 3195, Australia",
        "CSIRO-COSIMA":"CSIRO (Commonwealth Scientific and Industrial Research Organisation, Australia), COSIMA (Consortium for Ocean-Sea Ice Modelling in Australia). Mailing address: CSIRO, c/o Simon J. Marsland, 107-121 Station Street, Aspendale, Victoria 3195, Australia",
        "DKRZ":"Deutsches Klimarechenzentrum, Hamburg 20146, Germany",
        "DWD":"Deutscher Wetterdienst, Offenbach am Main 63067, Germany",
        "E3SM-Project":"LLNL (Lawrence Livermore National Laboratory, Livermore, CA 94550, USA); ANL (Argonne National Laboratory, Argonne, IL 60439, USA); BNL (Brookhaven National Laboratory, Upton, NY 11973, USA); LANL (Los Alamos National Laboratory, Los Alamos, NM 87545, USA); LBNL (Lawrence Berkeley National Laboratory, Berkeley, CA 94720, USA); ORNL (Oak Ridge National Laboratory, Oak Ridge, TN 37831, USA); PNNL (Pacific Northwest National Laboratory, Richland, WA 99352, USA); SNL (Sandia National Laboratories, Albuquerque, NM 87185, USA). Mailing address: LLNL Climate Program, c/o David C. Bader, Principal Investigator, L-103, 7000 East Avenue, Livermore, CA 94550, USA",
        "EC-Earth-Consortium":"AEMET, Spain; BSC, Spain; CNR-ISAC, Italy; DMI, Denmark; ENEA, Italy; FMI, Finland; Geomar, Germany; ICHEC, Ireland; ICTP, Italy; IDL, Portugal; IMAU, The Netherlands; IPMA, Portugal; KIT, Karlsruhe, Germany; KNMI, The Netherlands; Lund University, Sweden; Met Eireann, Ireland; NLeSC, The Netherlands; NTNU, Norway; Oxford University, UK; surfSARA, The Netherlands; SMHI, Sweden; Stockholm University, Sweden; Unite ASTR, Belgium; University College Dublin, Ireland; University of Bergen, Norway; University of Copenhagen, Denmark; University of Helsinki, Finland; University of Santiago de Compostela, Spain; Uppsala University, Sweden; Utrecht University, The Netherlands; Vrije Universiteit Amsterdam, the Netherlands; Wageningen University, The Netherlands. Mailing address: EC-Earth consortium, Rossby Center, Swedish Meteorological and Hydrological Institute/SMHI, SE-601 76 Norrkoping, Sweden",
        "ECMWF":"European Centre for Medium-Range Weather Forecasts, Reading RG2 9AX, UK",
        "FIO-QLNM":"FIO (First Institute of Oceanography, Ministry of Natural Resources, Qingdao 266061, China), QNLM (Qingdao National Laboratory for Marine Science and Technology, Qingdao 266237, China)",
        "HAMMOZ-Consortium":"ETH Zurich, Switzerland; Max Planck Institut fur Meteorologie, Germany; Forschungszentrum Julich, Germany; University of Oxford, UK; Finnish Meteorological Institute, Finland; Leibniz Institute for Tropospheric Research, Germany; Center for Climate Systems Modeling (C2SM) at ETH Zurich, Switzerland",
        "INM":"Institute for Numerical Mathematics, Russian Academy of Science, Moscow 119991, Russia",
        "IPSL":"Institut Pierre Simon Laplace, Paris 75252, France",
        "KIOST":"Korea Institute of Ocean Science and Technology, Busan 49111, Republic of Korea",
        "LLNL":"Lawrence Livermore National Laboratory, Livermore, CA 94550, USA. Mailing address: LLNL Climate Program, c/o Stephen A. Klein, Principal Investigator, L-103, 7000 East Avenue, Livermore, CA 94550, USA",
        "MESSy-Consortium":"The Modular Earth Submodel System (MESSy) Consortium, represented by the Institute for Physics of the Atmosphere, Deutsches Zentrum fur Luft- und Raumfahrt (DLR), Wessling, Bavaria 82234, Germany",
        "MIROC":"JAMSTEC (Japan Agency for Marine-Earth Science and Technology, Kanagawa 236-0001, Japan), AORI (Atmosphere and Ocean Research Institute, The University of Tokyo, Chiba 277-8564, Japan), NIES (National Institute for Environmental Studies, Ibaraki 305-8506, Japan), and R-CCS (RIKEN Center for Computational Science, Hyogo 650-0047, Japan)",
        "MOHC":"Met Office Hadley Centre, Fitzroy Road, Exeter, Devon, EX1 3PB, UK",
        "MPI-M":"Max Planck Institute for Meteorology, Hamburg 20146, Germany",
        "MRI":"Meteorological Research Institute, Tsukuba, Ibaraki 305-0052, Japan",
        "NASA-GISS":"Goddard Institute for Space Studies, New York, NY 10025, USA",
        "NASA-GSFC":"NASA Goddard Space Flight Center, Greenbelt, MD 20771, USA",
        "NCAR":"National Center for Atmospheric Research, Climate and Global Dynamics Laboratory, 1850 Table Mesa Drive, Boulder, CO 80305, USA",
        "NCC":"NorESM Climate modeling Consortium consisting of CICERO (Center for International Climate and Environmental Research, Oslo 0349), MET-Norway (Norwegian Meteorological Institute, Oslo 0313), NERSC (Nansen Environmental and Remote Sensing Center, Bergen 5006), NILU (Norwegian Institute for Air Research, Kjeller 2027), UiB (University of Bergen, Bergen 5007), UiO (University of Oslo, Oslo 0313) and UNI (Uni Research, Bergen 5008), Norway. Mailing address: NCC, c/o MET-Norway, Henrik Mohns plass 1, Oslo 0313, Norway",
        "NERC":"Natural Environment Research Council, STFC-RAL, Harwell, Oxford, OX11 0QX, UK",
        "NIMS-KMA":"National Institute of Meteorological Sciences/Korea Meteorological Administration, Climate Research Division, Seoho-bukro 33, Seogwipo-si, Jejudo 63568, Republic of Korea",
        "NIWA":"National Institute of Water and Atmospheric Research, Hataitai, Wellington 6021, New Zealand",
        "NOAA-GFDL":"National Oceanic and Atmospheric Administration, Geophysical Fluid Dynamics Laboratory, Princeton, NJ 08540, USA",
        "NTU":"National Taiwan University, Taipei 10650, Taiwan",
        "NUIST":"Nanjing University of Information Science and Technology, Nanjing, 210044, China",
        "PCMDI":"Program for Climate Model Diagnosis and Intercomparison, Lawrence Livermore National Laboratory, Livermore, CA 94550, USA",
        "PNNL-WACCEM":"PNNL (Pacific Northwest National Laboratory), Richland, WA 99352, USA",
        "RTE-RRTMGP-Consortium":"AER (Atmospheric and Environmental Research, Lexington, MA 02421, USA); UColorado (University of Colorado, Boulder, CO 80309, USA). Mailing address: AER c/o Eli Mlawer, 131 Hartwell Avenue, Lexington, MA 02421, USA",
        "RUBISCO":"ORNL (Oak Ridge National Laboratory, Oak Ridge, TN 37831, USA); ANL (Argonne National Laboratory, Argonne, IL 60439, USA); BNL (Brookhaven National Laboratory, Upton, NY 11973, USA); LANL (Los Alamos National Laboratory, Los Alamos, NM 87545); LBNL (Lawrence Berkeley National Laboratory, Berkeley, CA 94720, USA); NAU (Northern Arizona University, Flagstaff, AZ 86011, USA); NCAR (National Center for Atmospheric Research, Boulder, CO 80305, USA); UCI (University of California Irvine, Irvine, CA 92697, USA); UM (University of Michigan, Ann Arbor, MI 48109, USA). Mailing address: ORNL Climate Change Science Institute, c/o Forrest M. Hoffman, Laboratory Research Manager, Building 4500N Room F106, 1 Bethel Valley Road, Oak Ridge, TN 37831-6301, USA",
        "SNU":"Seoul National University, Seoul 08826, Republic of Korea",
        "THU":"Department of Earth System Science, Tsinghua University, Beijing 100084, China",
        "UA":"Department of Geosciences, University of Arizona, Tucson, AZ 85721, USA",
        "UCI":"Department of Earth System Science, University of California Irvine, Irvine, CA 92697, USA",
        "UCSB":"Bren School of Environmental Science and Management, University of California, Santa Barbara. Mailing address: c/o Samantha Stevenson, 2400 Bren Hall, University of California Santa Barbara, Santa Barbara, CA 93106, USA",
        "UHH":"Universitat Hamburg, Hamburg 20148, Germany"
    })

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