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
def run_app():
    """
    Runs the CMIP6 GPT application, setting up the UI and handling user interaction.
    """
    openai_api_key = st.secrets["openai"]["api_key"]
    Config.set_openai_api_key(openai_api_key)
    st.set_page_config(page_title="CMIP6 GPT", page_icon="🤖", layout="wide")
    st.markdown("# CMIP-6 GPT")
    # Sidebar for model selection
    with st.sidebar:
        st.title("Configuration")
        model_name = st.selectbox("Select Model", ["gpt-4o","gpt-4o-mini", "gpt-3.5-turbo", "o1-preview"], key="model_name")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            # st.experimental_rerun()

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    Config.set_model_name(model_name)
# Create the agent
    agent_executor = create_cmip6_agent()
    display_chat_messages()
    handle_user_input(agent_executor)