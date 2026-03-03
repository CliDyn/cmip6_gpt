from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import Config


# ─── Singleton Embedding ────────────────────────────────────────────
_embedding_instance = None


def create_embedding():
    """
    Returns a singleton embedding instance.

    The embedding model and provider are read from config.yaml.
    Supports both OpenAI (text-embedding-ada-002, text-embedding-3-*)
    and Google (gemini-embedding-001) providers.
    Using a singleton avoids re-instantiating the model on every call.
    """
    global _embedding_instance
    if _embedding_instance is None:
        provider = Config.get_embedding_provider()
        model = Config.get_embedding_model()
        if provider == "google":
            import os
            _embedding_instance = GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=os.environ.get("GOOGLE_API_KEY"),
            )
        else:
            _embedding_instance = OpenAIEmbeddings(
                model=model,
                openai_api_key=Config.get_openai_api_key(),
            )
    return _embedding_instance
def create_llm(temperature = 1, model_name = None):
    """
    Creates a language model (LLM) instance.

    Routes to the correct provider:
    - gemini-2.5-*:  ChatVertexAI  (vertex_api_key → higher rate limits)
    - gemini-3.*+:   ChatGoogleGenerativeAI (GOOGLE_API_KEY — not yet on Vertex)
    - Others:        ChatOpenAI
    """
    import os
    if model_name is None:
        model_name = Config.get_model_name()
    provider = Config.infer_provider(model_name)
    if provider == "google":
        vertex_key = os.environ.get("vertex_api_key")
        google_key = os.environ.get("GOOGLE_API_KEY")
        # gemini-2.5-* → Vertex AI (higher TPM), gemini-3.*+ → Google AI
        use_vertex = vertex_key and model_name.startswith("gemini-2.")
        if use_vertex:
            from langchain_google_vertexai import ChatVertexAI  # lazy import
            return ChatVertexAI(
                model_name=model_name,
                api_key=vertex_key,
                temperature=temperature,
                convert_system_message_to_human=True,
            )
        else:
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=google_key,
                temperature=temperature,
                convert_system_message_to_human=True,
            )
    else:
        openai_api_key = Config.get_openai_api_key()
        return ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key, temperature=temperature)
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
            "You are PangaeaGPT, a CMIP6 climate data assistant.\n\n"

            "## ROUTING\n"
            "Route every user message to exactly ONE tool:\n"
            "• cmip6_adviser → user asks WHAT something IS (explain variable/model/experiment)\n"
            "• cmip6_datasets_search → user wants to FIND dataset IDs matching criteria\n"
            "• cmip6_datasets_access → user wants to CHECK availability or DOWNLOAD data\n"
            "• get_analysis_guide → CALL BEFORE any analysis/plotting to get best practices\n"
            "• python_repl → user wants analysis or visualization\n\n"

            "## SEARCH TOOL RULES\n"
            "Pass the user's NATURAL LANGUAGE — never CMIP6 codes.\n"
            "• variable_query: variable in user's words ('sea surface temperature', NOT 'tos')\n"
            "• source_query: model in user's words ('MPI model', NOT 'MPI-ESM1-2-HR')\n"
            "• experiment_query: experiment in user's words ('historical run', NOT 'historical')\n"
            "• frequency/realm/activity_id: only if explicitly mentioned\n"
            "Leave unmentioned args null. Prefer one strong arg over many weak ones.\n"
            "Category shortcuts: 'ocean data' → realm='ocean', NOT variable_query.\n"
            "Institution+resolution: 'high-res from AWI' → source_query, NOT nominal_resolution.\n\n"

            "## REFINEMENT PROTOCOL (0 results)\n"
            "1. Drop source_id → retry\n"
            "2. Still 0 → drop experiment_id → retry\n"
            "3. Still 0 → keep only variable_id + variant_label → retry\n"
            "Never return empty results without exhausting these steps.\n\n"

            "## ADVISER TOOL\n"
            "Include relevant_facets always. Add vector_search_fields only when asking about "
            "a specific variable_id, source_id, or experiment_id.\n\n"

            "## ACCESS TOOL\n"
            "Use facet_values returned by search. Always keep variant_label unless user overrides.\n"
            "Report: dataset count, model breakdown, ESGF link, Python download snippet.\n"
            "Mention 'Detailed information on datasets' and 'Python access from GCS' tabs.\n\n"

            "## ANALYSIS PROTOCOL\n"
            "1. Clarify objective → 2. Call get_analysis_guide for the relevant topic → "
            "3. Load data via access snippet → "
            "4. Compute in python_repl following the guide's quality checklist → "
            "5. Plot with correct colormaps, labels, and Cartopy coastlines; "
            "return figure paths ONLY from 'figures' in tool output.\n\n"

            "## FORMATTING\n"
            "Use inline code (`backticks`) for short identifiers like variable names, "
            "model names, units, and values (e.g. `tos`, `MPI-ESM1-2-HR`, `K`). "
            "Reserve fenced code blocks (```) ONLY for multi-line code, commands, or snippets. "
            "Never break a sentence across a code block — keep prose flowing.\n"
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
    