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
def create_llm(temperature = 0.7):
    """
    Creates a language model (LLM) instance using the OpenAI API.

    This function retrieves the model name and API key from the configuration, and returns an instance 
    of `ChatOpenAI` with the specified temperature for response variability.

    Args:
        temperature (float, optional): The temperature setting for the LLM, which controls the randomness of responses. Defaults to 0.7.

    Returns:
        ChatOpenAI: An instance of the ChatOpenAI model.
    """
    model_name = Config.get_model_name()
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
            "You are a knowledgeable assistant specializing in climate data and general conversations. "
            "Your primary tool is the CMIP6 data process tool, which you should use whenever the user requests specific CMIP6 data. "
            "Always consider the entire conversation history to understand the user's intent and provide coherent, context-aware responses. "
            "If the user modifies their request or adds additional constraints in follow-up messages, determine whether to use the tool again or provide a direct answer based on the updated context."
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
                                            "You are an AI assistant tasked with separating a user query into specific components. Analyze the given query and extract information about **variables**, **sources**, and **experiments** if present.\n\n"
                                            "Query: {query}\n\n"
                                            "Return a JSON object with 'variable_query', 'source_query', and 'experiment_query' keys. Follow these guidelines strictly:\n\n"
                                            "1. **variable_query**:\n"
                                            "   - Include any mentioned variables or parameters (e.g., 'temperature', 'salinity', 'sea surface temperature', 'precipitation anomalies').\n"
                                            "   - **Do not include generic terms like 'data' or 'variables' unless they are part of a specific variable description.**\n"
                                            "   - Only include content that specifically refers to variables.\n\n"
                                            "2. **source_query**:\n"
                                            "   - Include any mentioned data sources, models, or institutions (e.g., 'GFDL model', 'NOAA data', 'CMIP6 simulations').\n"
                                            "   - **Do not include generic terms like 'models' or 'data sources' or 'CMIP6 dataset' or 'dataset' or 'CMIP6' unless they are part of a specific source description.**\n"
                                            "   - Only include content that specifically refers to sources.\n\n"
                                            "3. **experiment_query**:\n"
                                            "   - Include ONLY specific experiment names, types, or detailed descriptions if present.\n"
                                            "   - This should refer to particular experimental conditions or setups, not general mentions of 'experiments'.\n"
                                            "   - Examples of valid entries: 'historical run', 'future climate projections', 'under increased CO₂ conditions'.\n"
                                            "   - **Do not include generic terms like 'experiments' or 'models' here.**\n\n"
                                            "4. **Use the exact wording from the query for each component.**\n\n"
                                            "5. **If a component is not clearly present in the query, leave its value as an empty string.**\n\n"
                                            "**IMPORTANT:**\n"
                                            "- The words 'data', 'variables', or 'models' alone do not constitute valid entries for 'variable_query' or 'source_query'.\n"
                                            "- Only include content in 'variable_query', 'source_query', or 'experiment_query' if it describes **specific** variables, sources, or experiments, respectively.\n"
                                            "- **Do not infer or assume information not explicitly mentioned in the query.**\n\n"
                                            "**ONLY RETURN THE JSON OBJECT, DO NOT ADD ANY OTHER TEXT.**"
                                            "Example output format: "
                                            "{{\"variable_query\": \"Sea surface temperature, SST\", \"source_query\": \"Alfred Wegener Institute, AWI\", \"experiment_query\": \"historical run\"}}"
                                        )
                                    )
    return split_query_template