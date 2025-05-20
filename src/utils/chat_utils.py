import streamlit as st
from typing import List, Optional, Dict, Any
import pandas as pd
import os
import re
import json
import requests


def parse_image_markdown(text: str) -> (str, Optional[str]):
    """Extract image path from markdown and return cleaned text and absolute path."""
    match = re.search(r"!\[.*?\]\((.*?)\)", text)
    if match:
        path = match.group(1)
        full_path = os.path.join(os.getcwd(), "temp_figures", os.path.basename(path))
        clean_text = re.sub(r"!\[.*?\]\(.*?\)", "", text).strip()
        return clean_text, full_path
    return text, None
def display_chat_messages():
    """
    Displays chat messages from the session state in a chat-like format.

    This function iterates through the chat messages stored in the session state and displays each one 
    according to its role (e.g., 'user', 'assistant') in a markdown format.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                try:
                    clean, img = parse_image_markdown(message["content"])
                    st.markdown(clean)
                    if img:
                        st.image(img, width=750)
                except Exception:
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
                clean, img = parse_image_markdown(full_response)
                message_placeholder.markdown(clean)
                if img:
                    st.image(img, width=750)
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
        if content == None:
            content = 'python_repl: ' + message['tool_calls']
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
def display_debug_info_final(title, content, download_opendap = False):
    """
    Displays debugging information in an expandable table format and collects
    all OpenDAP links across models.
    
    Args:
        title (str): The title for the expandable section
        content (str): The JSON content to display
        
    Returns:
        list: All collected OpenDAP links with model information when title is "Final Facet Values"
    """
    data = json.loads(content)
    all_model_links = []
    
    with st.expander(f"{title}", expanded=False):
        st.write(f"Total datasets found: {data['hit_count']}")
        
        # Initialize session state for OpenDAP links if not exists
        if 'opendap_links' not in st.session_state:
            st.session_state.opendap_links = {}
            
        for model_name, model_data in data['models'].items():
            st.write(f"### {model_name}")
            
            # Create tabs for each model
            tab_titles = ["Model Information"]
            if download_opendap:
                tab_titles.append("OpenDAP Links")
            tabs = st.tabs(tab_titles)
            with tabs[0]: 
                # Create DataFrame for the model's parameters
                model_rows = []
                model_rows.append({
                    'Parameter': 'Total Datasets',
                    'Values': str(model_data['dataset_count']),
                    'Details': ''
                })
                
                for param, values in model_data.items():
                    if param != 'dataset_count':
                        value_str = ', '.join([f"{k}: {v}" for k, v in values.items()])
                        total_count = sum(values.values())
                        model_rows.append({
                            'Parameter': param,
                            'Values': f"Total: {total_count}",
                            'Details': value_str
                        })
                
                df = pd.DataFrame(model_rows)
                st.dataframe(
                    df,
                    column_config={
                        "Parameter": st.column_config.Column("Parameter", width="medium"),
                        "Values": st.column_config.Column("Count", width="small"),
                        "Details": st.column_config.Column("Detailed Breakdown", width="large")
                    },
                    hide_index=True
                )
            if download_opendap and len(tabs) > 1:
                with tabs[1]:  # OpenDAP Links tab
                    # RESET SESSION STATE FOR THIS MODEL (FOR TESTING)
                    # Uncomment to force refresh data when testing
                    # if model_name in st.session_state.opendap_links:
                    #     del st.session_state.opendap_links[model_name]
                
                    if model_name not in st.session_state.opendap_links:
                        # Extract all parameters with multiple values
                        multi_value_parameters = {}
                        parameter_values = {}
                        
                        # Default member_id
                        parameter_values['member_id'] = ['r1i1p1f1']
                        
                        # Extract all parameter values and track those with multiple values
                        for param, values in model_data.items():
                            if param != 'dataset_count' and values:
                                value_list = list(values.keys())
                                parameter_values[param] = value_list
                                if len(value_list) > 1:
                                    multi_value_parameters[param] = value_list
                        
                        # Get all combinations of parameter values
                        param_combinations = []
                        
                        # Helper function to recursively generate all combinations
                        def generate_combinations(params, current_index, current_combo):
                            if current_index == len(params):
                                param_combinations.append(current_combo.copy())
                                return
                            
                            param_name = list(params.keys())[current_index]
                            for value in params[param_name]:
                                current_combo[param_name] = value
                                generate_combinations(params, current_index + 1, current_combo)
                        
                        generate_combinations(parameter_values, 0, {})
                        
                        # Preferred nodes in order of priority
                        preferred_nodes = ["aims3.llnl.gov", "esgf-data1.llnl.gov", "esgf-data2.llnl.gov"]
                        
                        # Dictionary to organize links by unique ID (combination of filename and parameters)
                        # This ensures we don't overwrite different parameter combinations with same filename
                        all_results = []
                        
                        # Fetch OpenDAP links for each parameter combination
                        with st.spinner(f"Fetching OpenDAP links for {model_name}..."):
                            for combination_idx, combination in enumerate(param_combinations):
                                try:
                                    # Call esgf_search with the current combination of parameters
                                    all_links = esgf_search(**combination)
                                    
                                    # Process each link
                                    for link in all_links:
                                        try:
                                            # Extract filename (last part of URL)
                                            filename = link.split('/')[-1]
                                            
                                            # Extract node from URL
                                            url_parts = link.split('/')
                                            node = url_parts[2] if len(url_parts) > 3 else "unknown"
                                            
                                            # Create a unique ID that includes relevant parameter values
                                            # This ensures we don't overwrite results with different parameters
                                            param_id_parts = []
                                            for param in multi_value_parameters:
                                                if param in combination:
                                                    param_id_parts.append(f"{param}={combination[param]}")
                                            
                                            # Include parameter values in the result
                                            result_entry = {
                                                'filename': filename,
                                                'node': node,
                                                'url': link,
                                                'params': combination.copy()  # Store all parameters
                                            }
                                            
                                            # Add to results array
                                            all_results.append(result_entry)
                                            
                                        except Exception as e:
                                            st.warning(f"Error processing link {link}: {e}")
                                    
                                except Exception as e:
                                    st.error(f"Error fetching OpenDAP links for combination {combination}: {e}")
                        
                        # Apply node preferences - group by unique combination of filename and parameters
                        final_results = {}
                        
                        # Group results by their unique parameter combination + filename
                        for result in all_results:
                            # Create a key that uniquely identifies this result
                            key_parts = [result['filename']]
                            for param in multi_value_parameters:
                                if param in result['params']:
                                    key_parts.append(f"{param}={result['params'][param]}")
                            
                            unique_key = "|".join(key_parts)
                            
                            if unique_key not in final_results:
                                final_results[unique_key] = []
                            
                            final_results[unique_key].append(result)
                        
                        # For each unique result, select the preferred node
                        best_results = []
                        for unique_key, result_group in final_results.items():
                            # First try to find preferred nodes
                            selected_result = None
                            
                            for preferred_node in preferred_nodes:
                                for result in result_group:
                                    if result['node'] == preferred_node:
                                        selected_result = result
                                        break
                                if selected_result:
                                    break
                            
                            # If no preferred node found, use the first one
                            if not selected_result and result_group:
                                selected_result = result_group[0]
                                
                            if selected_result:
                                best_results.append(selected_result)
                        
                        # Store links with their parameters in session state
                        st.session_state.opendap_links[model_name] = best_results
                    
                    # Display OpenDAP links as a DataFrame with dynamic parameter columns
                    if st.session_state.opendap_links.get(model_name):
                        links_data = st.session_state.opendap_links[model_name]
                        
                        # Check if number of links exceeds limit
                        if len(links_data) > 500:
                            st.warning(f"⚠️ This model has {len(links_data)} OpenDAP links, which exceeds the 500 link limit. Please access the ESGF server directly to download this data.")
                            
                            # Show only first 10 links as preview
                            display_links = links_data[:10]
                            st.write("Showing first 10 links as preview:")
                        else:
                            display_links = links_data
                            st.write(f"Total unique OpenDAP links found: {len(links_data)}")
                        
                        # Collect all links for the final return value
                        for link_data in links_data:
                            all_model_links.append({
                                "model": model_name,
                                "node": link_data['node'],
                                "filename": link_data['filename'],
                                "url": link_data['url'],
                                **link_data['params']  # Include all parameters
                            })
                        
                        # Get parameters with more than one unique value across all results
                        param_to_values = {}
                        for link_data in display_links:
                            for param, value in link_data['params'].items():
                                if param not in param_to_values:
                                    param_to_values[param] = set()
                                param_to_values[param].add(value)
                        
                        # Identify parameters with multiple values
                        dynamic_columns = [param for param, values in param_to_values.items() 
                                        if len(values) > 1]
                        
                        # Extract and display node and filename information with dynamic parameter columns
                        link_info = []
                        for i, link_data in enumerate(display_links):
                            # Extract date range (everything from last underscore to before .nc)
                            filename = link_data['filename']
                            date_range = "unknown"
                            if '_' in filename and filename.endswith('.nc'):
                                parts = filename.split('_')
                                if len(parts) > 1:
                                    # Get the last part before .nc extension
                                    date_part = parts[-1].replace('.nc', '')
                                    if '-' in date_part:  # Make sure it looks like a date range
                                        date_range = date_part
                            
                            # Create entry with model name and base columns
                            entry = {
                                "ID": i+1,
                                "Model": model_name,
                                "Data Node": link_data['node'],
                                "Date Range": date_range
                            }
                            
                            # Add dynamic parameter columns
                            for param in dynamic_columns:
                                entry[param] = link_data['params'].get(param, "")
                            
                            # Add remaining standard columns
                            entry.update({
                                "Filename": filename,
                                "OpenDAP URL": link_data['url']
                            })
                            
                            link_info.append(entry)
                        
                        # Create DataFrame with dynamic columns
                        opendap_df = pd.DataFrame(link_info)
                        
                        # Configure column order and properties
                        column_config = {
                            "ID": st.column_config.Column("ID", width="small"),
                            "Model": st.column_config.Column("Model", width="medium"),
                            "Data Node": st.column_config.Column("Data Node", width="medium"),
                            "Date Range": st.column_config.Column("Date Range", width="medium")
                        }
                        
                        # Add config for dynamic parameter columns
                        for param in dynamic_columns:
                            column_config[param] = st.column_config.Column(param, width="medium")
                        
                        # Add remaining standard column config
                        column_config.update({
                            "Filename": st.column_config.Column("Filename", width="large"),
                            "OpenDAP URL": st.column_config.TextColumn("OpenDAP URL", width="large")
                        })
                        
                        # Display DataFrame with dynamic columns
                        st.dataframe(
                            opendap_df,
                            column_config=column_config,
                            hide_index=True
                        )
                        
                    else:
                        st.write("No OpenDAP links available for this model.")
                        # Return all collected links
                    return pd.DataFrame(all_model_links)
            
            st.write("---")
    # Return empty list for other titles
    return all_model_links

def display_opendap_links(df: pd.DataFrame) -> None:
    """
    Display a pandas DataFrame containing unique OpenDAP links in Streamlit.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data to display
        link_column (str): Name of the column containing the links (default: 'Link')
        
    Returns:
        None
    """
    link_column = 'url'
    # Remove duplicate links
    if link_column in df.columns:
        initial_count = len(df)
        df = df.drop_duplicates(subset=[link_column], keep='first')
        unique_count = len(df)
    else:
        st.warning(f"Column '{link_column}' not found in DataFrame")
        initial_count = len(df)
        unique_count = initial_count

    # Add a title
    st.subheader("OpenDAP Links: ")
    
    # Display the DataFrame
    st.dataframe(
        df,
        use_container_width=True,
        height=400,
    )
    
    # Display stats
    st.write(f"Total number of links: {unique_count}")
def display_python_code(query):
    code = f"""
    import pandas as pd
    import xarray as xr
    # Load the metadata CSV
    df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
    # Filter the dataframe
    df_spec = df.query("{query}")
    # get the path to a specific zarr store (the first one from the dataframe above)
    zstore = df_spec.zstore.values[-1]
    # Open the first dataset
    ds = xr.open_zarr(zstore, consolidated=True, storage_options={{'token':'anon'}})"""
    with st.expander("Python access from Google Cloude Storage", expanded=False):
        st.header("CMIP6 Data Access Code") 
        st.write("This code loads climate model data from Google Cloud Storage using Zarr format.")
        st.code(code, language='python')
    return code


# Author: Unknown
# I got the original version from a word document published by ESGF
# https://docs.google.com/document/d/1pxz1Kd3JHfFp8vR2JCVBfApbsHmbUQQstifhGNdc6U0/edit?usp=sharing

# API AT: https://github.com/ESGF/esgf.github.io/wiki/ESGF_Search_REST_API#results-pagination

def esgf_search(server="https://esgf-node.llnl.gov/esg-search/search",
                files_type="OPENDAP", local_node=True, project="CMIP6",
                verbose=False, format="application%2Fsolr%2Bjson",
                use_csrf=False, **search):
    client = requests.session()
    payload = search
    payload["project"] = project
    payload["type"]= "File"
    if local_node:
        payload["distrib"] = "false"
    if use_csrf:
        client.get(server)
        if 'csrftoken' in client.cookies:
            # Django 1.6 and up
            csrftoken = client.cookies['csrftoken']
        else:
            # older versions
            csrftoken = client.cookies['csrf']
        payload["csrfmiddlewaretoken"] = csrftoken

    payload["format"] = format

    offset = 0
    numFound = 10000
    all_files = []
    files_type = files_type.upper()
    while offset < numFound:
        payload["offset"] = offset
        url_keys = [] 
        for k in payload:
            url_keys += ["{}={}".format(k, payload[k])]

        url = "{}/?{}".format(server, "&".join(url_keys))
        print(url)
        r = client.get(url)
        r.raise_for_status()
        resp = r.json()["response"]
        numFound = int(resp["numFound"])
        resp = resp["docs"]
        offset += len(resp)
        for d in resp:
            if verbose:
                for k in d:
                    print("{}: {}".format(k,d[k]))
            url = d["url"]
            for f in d["url"]:
                sp = f.split("|")
                if sp[-1] == files_type:
                    all_files.append(sp[0].split(".html")[0])
    return sorted(all_files)