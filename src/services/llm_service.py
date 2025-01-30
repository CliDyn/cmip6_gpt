from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from src.config import Config
def create_embedding():
    """
    Creates an embedding model using the OpenAI API.

    This function retrieves the OpenAI API key from the configuration and returns an instance 
    of `OpenAIEmbeddings`, which can be used for embedding data.

    Returns:
        OpenAIEmbeddings: An instance of the OpenAI embeddings model.
    """
    openai_api_key = Config.get_openai_api_key()
    return OpenAIEmbeddings(model = 'text-embedding-ada-002', openai_api_key=openai_api_key)
def create_llm(temperature = 1, model_name = Config.get_model_name()):
    """
    Creates a language model (LLM) instance using the OpenAI API.

    This function retrieves the model name and API key from the configuration, and returns an instance 
    of `ChatOpenAI` with the specified temperature for response variability.

    Args:
        temperature (float, optional): The temperature setting for the LLM, which controls the randomness of responses. Defaults to 0.7.

    Returns:
        ChatOpenAI: An instance of the ChatOpenAI model.
    """
    openai_api_key = Config.get_openai_api_key()
    return ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key, temperature = temperature)
def create_prompt_template():
    """
    Creates a prompt template for a conversational assistant specializing in climate data.

    This function constructs a `ChatPromptTemplate` that guides the assistant to use the CMIP6 data 
    process tool when necessary. The template incorporates system messages to ensure the assistant 
    provides coherent and context-aware responses, considering the conversation history.

    Returns:
        ChatPromptTemplate: A prompt template designed for handling CMIP6 data requests and conversations.
    """
    # Create the prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a specialized assistant for accessing CMIP6 climate data. Your primary goal is to find the "
            "most relevant facet_values and corresponding datasets that match user requirements.\n\n"
            
            "Available Tools:\n"
            "1. cmip6_datasets_search\n"
            "   - ALWAYS use this tool first to get initial facet_values\n"
            "   - Never create facet_values without using this tool first (however you can modify)\n"
            "   - Use it again when you need to explore alternative search parameters\n"
            "2. cmip6_datasets_access\n"
            "   - Use this to check data availability with given facet_values\n"
            "   - Always examine the total_datasets count in the response\n"
            "   - This tool can be used when the user wants to know what is available for a specific facet."
            "3. cmip6_adviser\n"
            "   - Use ONLY for understanding variables (variable_id), models (source_id), or experiments (experiment_id)\n"
            "   - Helpful when evaluating alternative options\n\n"
            
            "Protocol:\n"
            "1. START:\n"
            "   - Use cmip6_datasets_search for initial facet_values\n"
            "   - Verify with cmip6_datasets_access\n"
            
            "2. REFINE:\n"
            "   - If no results, systematically adjust facets\n"
            "   - Focus on most important parameters for user's needs\n"
            "   - Remove less crucial facets if needed\n"
            
            "3. FINAL OUTPUT FORMAT:\n"
            # " \"final_facet_values\": {...},\n" 
            " datasets: List of available datasets\n"
            " Explanation: Justify why these datasets are most relevant\n\n"

            
            "Remember:\n"
            "- Always use real facet values from cmip6_datasets_search\n"
            "- Focus on finding actual available datasets\n"
            "- Prioritize exact matches to user requirements\n"
            "- Include only verified available datasets in response"
            "- DO NOT split complicated queries into several facets, better find one most relevant facet. For example, nominal_resolution and institution_id can be combined into source_id."
            "- When users mention some general concepts like 'ocean data', 'sea ice data', 'atmosphere data' and ect, you can consider 'realm' facet."
            "- DO NOT provide results if you have no datasets, always try to find solution"
            "- ALWAYS follow the protocol for consistent results"
            # "- for FINAL OUTPUT add JSON obcejct ONLY for final_facet_values. DATASETS and EXPLANATION must be written in regular format"

            "In situations where no datasets are found for a given set of facet_values:\n"

            "1. Refine Search Parameters:\n"
            "   - Consider removing one or more facet_values that may be overly restrictive.\n"
            "   - For example, if filtering by a specific source_id and experiment_id yields no results, try removing one of these facets and search again.\n"
            "   - Use cmip6_datasets_access to explore what datasets become available after reducing the number of facets.\n"

            "2. Evaluate Results from Different Combinations:\n"
            "   - Experiment with removing different facets to determine which combination returns datasets most relevant to the user’s original intent.\n"
            "   - Compare results from these refined searches.\n"
            "   - Prioritize the combination that best aligns with the user’s needs.\n"

            "3. Decision Making:\n"
            "   - Based on the outcomes, select the dataset(s) that most closely match the user’s requirements.\n"
            "   - If multiple options are available, choose the one that best meets the original user's qeury, or present an alternative that could still be useful.\n"

        )),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    return prompt_template

def create_split_query_template():
    """
    Creates a prompt template for splitting a user query into specific components: variables, sources, and experiments.

    This function returns a `PromptTemplate` that guides the AI assistant in extracting relevant components 
    (variables, sources, and experiments) from the user's query. It ensures the assistant follows strict 
    guidelines and provides a structured JSON response with the extracted components.

    Returns:
        PromptTemplate: A template designed to split a query into variable, source, and experiment components.
    """
    split_query_template = PromptTemplate(
    input_variables=["query"],
    template=(
        "You are an AI assistant tasked with analyzing a user query and separating it into three specific components: **variables**, **sources**, and **experiments**. "
        "Query: {query}\n\n"
        "The context of these queries often relates to CMIP6 data or similar scientific datasets. Your job is to identify and classify the query elements into these categories based on the following strict guidelines:\n\n"
        "### Guidelines:\n\n"
        "1. **Variable Query**:\n"
        "   - Extract specific variable names or parameters mentioned in the query (e.g., 'temperature', 'salinity', 'thetaot', 'sea surface temperature').\n"
        "   - Include only terms directly referring to specific variables; do not include general terms like 'data' or 'variables' unless part of a specific variable name.\n\n"
        
        "2. **Source Query**:\n"
        "   - Extract any mentioned data sources, models, or institutions (e.g., 'GFDL model', 'NOAA data', 'CMIP6 simulations').\n"
        "   - Include only specific source descriptions; do not include general terms like 'dataset' or 'CMIP6' unless tied to a specific source.\n\n"
        
        "3. **Experiment Query**:\n"
        "   - Extract names or descriptions of specific experiments, runs, or conditions (e.g., 'historical run', 'hist-1950', 'future climate projections under increased CO₂').\n"
        "   - Do not include general terms like 'experiments' unless explicitly tied to a named experiment or condition.\n\n"
        
        "### Output Format:\n"
        "Return a JSON object with the following structure:\n"
        "{{\"variable_query\": \"<specific variable here>\", \"source_query\": \"<specific source here>\", \"experiment_query\": \"<specific experiment here>\"}}\n\n"
        
        "### Important Notes:\n"
        "- Use the **exact wording** from the query for each component.\n"
        "- If a component is **not clearly present** in the query, leave its value as an empty string (`\"\"`).\n"
        "- **Do not infer or assume** information not explicitly mentioned in the query.\n\n"
        "- **Ignore generic placeholder terms like “variable_id”, ”source_id”, which are not actual names."
        
        "### Examples:\n"
        "1. Query: \"thetaot\"\n"
        "   Output:\n"
        "   {{\"variable_query\": \"thetaot\", \"source_query\": \"\", \"experiment_query\": \"\"}}\n\n"
        "2. Query: \"What is hist-1950?\"\n"
        "   Output:\n"
        "   {{\"variable_query\": \"\", \"source_query\": \"\", \"experiment_query\": \"hist-1950\"}}\n\n"
        "3. Query: \"GFDL model salinity in historical runs\"\n"
        "   Output:\n"
        "   {{\"variable_query\": \"salinity\", \"source_query\": \"GFDL model\", \"experiment_query\": \"historical runs\"}}\n\n"
        
        "Accurately identify and classify the query components based on the above guidelines. Do not include any additional text in your response—output only the JSON object."
        )
    )
    return split_query_template