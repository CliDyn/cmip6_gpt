import streamlit as st
from src.config import Config
from src.agents.cmip6_agent import create_cmip6_agent
from src.utils.chat_utils import display_chat_messages, handle_user_input
import os

try:
    os.environ['LANGCHAIN_TRACING_V2'] = st.secrets["LANGCHAIN"]["LANGCHAIN_TRACING_V2"]
    os.environ['LANGCHAIN_ENDPOINT'] = st.secrets["LANGCHAIN"]["LANGCHAIN_ENDPOINT"]
    os.environ['LANGCHAIN_API_KEY'] = st.secrets["LANGCHAIN"]["LANGCHAIN_API_KEY"]
except:
    print('No Langchain tracing')

def init_session_state():
    """Initialize all session state variables and preserve expanded models."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "opendap_states" not in st.session_state:
        st.session_state.opendap_states = {}
    if "expanded_models" not in st.session_state:
        st.session_state.expanded_models = set()  # Persist previous state
    if "pending_expanders" not in st.session_state:
        st.session_state.pending_expanders = []
@st.cache_resource # Add any parameters here if the agent creation depends on them e.g. _model_name
def get_cached_agent_executor():
    print("get_cached_agent_executor: Creating new agent executor instance...")
    # Config.get_model_name() will be used by create_cmip6_agent internally
    # If create_cmip6_agent needs the model_name directly, pass it:
    # agent_exec = create_cmip6_agent(model_name=Config.get_model_name())
    agent_exec = create_cmip6_agent()
    print(f"get_cached_agent_executor: Agent executor created. ID: {id(agent_exec)}")
    # You can also print the ID of the REPL tool's instance here for debugging
    for tool in agent_exec.tools:
        if hasattr(tool, 'name') and tool.name == "python_repl" and hasattr(tool, 'func') and hasattr(tool.func, '__self__'):
            print(f"REPL instance ID in cached agent: {id(tool.func.__self__)}")
            break
    return agent_exec

def run_app():
    """
    Runs the CMIP6 GPT application, setting up the UI and handling user interaction.
    """
    # Initialize session state
    init_session_state()
    
    openai_api_key = st.secrets["openai"]["api_key"]
    Config.set_openai_api_key(openai_api_key)
    st.set_page_config(page_title="CMIP6 GPT", page_icon="🤖", layout="wide")
    st.markdown("# CMIP-6 GPT")

    # Sidebar for model selection
    with st.sidebar:
        st.title("Configuration")
        model_name = st.selectbox(
            "Select Model", 
            ["gpt-4o","gpt-4o-mini", "gpt-3.5-turbo", "o1-preview"], 
            key="model_name"
        )
        if model_name != Config.get_model_name():
            Config.set_model_name(model_name)
            # If the agent's LLM needs to be updated or the agent recreated based on model name,
            # you might need to clear the cache for get_cached_agent_executor.
            # For simple LLM change within the same agent structure, this might not be needed
            # if the agent internally picks up the new model from Config.
            # If create_cmip6_agent itself needs model_name to build different tools,
            # then caching needs to be aware of model_name.
            # A simple way to force re-creation on model change:
            st.cache_resource.clear() # Clears all resource caches
            print(f"Model changed to {model_name}. Cleared agent cache.")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.opendap_states = {}
            st.session_state.expanded_models = set()
            st.cache_resource.clear()
            print("Chat history cleared. Agent cache (and REPL state) also cleared.")
            st.rerun()

    Config.set_model_name(model_name)
    agent_executor = get_cached_agent_executor()
    display_chat_messages()
    handle_user_input(agent_executor)