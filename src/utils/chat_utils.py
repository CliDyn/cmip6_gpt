import streamlit as st
from typing import List, Optional, Dict, Any
import json
def display_chat_messages():
    """
    Displays chat messages from the session state in a chat-like format.

    This function iterates through the chat messages stored in the session state and displays each one 
    according to its role (e.g., 'user', 'assistant') in a markdown format.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(agent_executor):
    """
    Handles the user's input and generates a response from the AI assistant.

    This function takes user input through the chat interface, appends it to the session history, and 
    streams the AI's response using the provided `agent_executor`. The response is displayed in real time, 
    and if the response contains a CMIP6 data request, it parses and displays both the summary and full result.

    Args:
        agent_executor: The agent responsible for processing the user's query and generating a response.
    """
    if user_input := st.chat_input("What would you like to know about climate data or CMIP6?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                for chunk in agent_executor.stream({"input": user_input, "chat_history": st.session_state.messages}):
                    if isinstance(chunk, dict) and "output" in chunk:
                        full_response += chunk["output"]
                        message_placeholder.markdown(full_response + "▌")

                # Parse the full_response if it's a JSON string
                try:
                    response_dict = json.loads(full_response)
                    if isinstance(response_dict, dict) and "summary" in response_dict and "full_result" in response_dict:
                        message_placeholder.markdown(response_dict["summary"])
                        with st.expander("Full CMIP6 Data Request Result", expanded=False):
                            st.json(json.loads(response_dict["full_result"]))
                    else:
                        message_placeholder.markdown(full_response)
                except json.JSONDecodeError:
                    message_placeholder.markdown(full_response)

            except Exception as e:
                error_message = f"An error occurred: {str(e)}\n\nPlease try rephrasing your query or contact support if the issue persists."
                st.error(error_message)
                full_response = error_message

        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

def format_chat_history(chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Formats the chat history into a readable string for use in prompts or logs.

    This function takes the chat history (or retrieves it from the session state if not provided) and 
    formats it by labeling each message with either 'User' or 'Assistant', followed by the content of 
    the message.

    Args:
        chat_history (Optional[List[Dict[str, str]]]): A list of dictionaries representing the chat history. 
        If not provided, it retrieves the messages from session state.

    Returns:
        str: A formatted string representing the chat history, with each message labeled by role.
    """
    if chat_history is None:
        chat_history = st.session_state.get('messages', [])
    formatted_history = ""
    for message in chat_history:
        role = "User" if message["role"] == "user" else "Assistant"
        content = message["content"]
        formatted_history += f"{role}: {content}\n"
    return formatted_history


def display_debug_info(title, content):
    """
    Displays debugging information in an expandable section.

    This function creates an expandable section in the UI, titled with the given `title`, and displays 
    the provided `content` as a JSON object for debugging purposes.

    Args:
        title (str): The title for the expandable section.
        content (Any): The content to display inside the expandable section, formatted as JSON.
    """
    with st.expander(f"{title}", expanded=False):
        st.json(content)