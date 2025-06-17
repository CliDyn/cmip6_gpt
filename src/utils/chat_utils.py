import streamlit as st
from typing import List, Optional, Dict, Any
import pandas as pd
import os
import re
import json
import requests
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class StreamlitPlotHandler:
    """Helper class to handle plot display in Streamlit"""
    
    @staticmethod
    def display_plots_from_result(result: Dict[str, Any]) -> bool:
        """
        Display plots from REPL result immediately in Streamlit.
        Returns True if plots were displayed, False otherwise.
        """
        if not result:
            return False
            
        plot_data = result.get("plot_data", [])
        if not plot_data:
            return False
            
        plots_displayed = False
        for i, plot_data_item in enumerate(plot_data):
            try:
                # Handle both raw bytes and base64-encoded strings
                if isinstance(plot_data_item, str):
                    plot_bytes = base64.b64decode(plot_data_item)
                else:
                    plot_bytes = plot_data_item
                
                # Create a container for the plot
                with st.container():
                    # Display the plot with better formatting
                    st.image(
                        plot_bytes, 
                        use_container_width=True, 
                        caption=f"Plot {i+1}"
                    )
                    plots_displayed = True
            except Exception as e:
                st.error(f"Error displaying plot {i+1}: {str(e)}")
            
        return plots_displayed
    
    @staticmethod
    def display_plots_from_files(file_paths: List[str]) -> bool:
        """
        Display plots from file paths.
        Returns True if plots were displayed, False otherwise.
        """
        if not file_paths:
            return False
            
        plots_displayed = False
        for i, path in enumerate(file_paths):
            try:
                if os.path.exists(path):
                    # Display each plot in its own container
                    with st.container():
                        st.image(
                            path, 
                            use_container_width=True, 
                            caption=f"Plot {i+1} - {os.path.basename(path)}"
                        )
                        plots_displayed = True
                else:
                    st.warning(f"Plot file not found: {path}")
            except Exception as e:
                st.error(f"Error displaying plot {i+1} from file {path}: {str(e)}")
                
        return plots_displayed

class OptimizedStreamlitPlotHandler:
    """
    Optimized plot handler for handling path-based plot storage.
    This reduces token usage by storing plot paths instead of base64 data.
    """
    
    @staticmethod
    def display_plots_from_paths(file_paths: List[str]) -> bool:
        """
        Display plots from file paths with optimized handling.
        Returns True if plots were displayed, False otherwise.
        """
        if not file_paths:
            return False
            
        plots_displayed = False
        for i, path in enumerate(file_paths):
            try:
                if os.path.exists(path):
                    # Display each plot in its own container
                    with st.container():
                        st.image(
                            path, 
                            use_container_width=True, 
                            caption=f"Generated Plot {i+1}"
                        )
                        plots_displayed = True
                else:
                    st.warning(f"Plot file not found: {path}")
            except Exception as e:
                st.error(f"Error displaying plot {i+1} from file {path}: {str(e)}")
                
        return plots_displayed
    
    @staticmethod
    def cleanup_temp_plots(file_paths: List[str]) -> None:
        """
        Clean up temporary plot files to save disk space.
        """
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                st.warning(f"Could not remove temporary file {path}: {str(e)}")

def parse_image_markdown(text: str) -> tuple[str, Optional[str]]:
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
    Enhanced version of display_chat_messages with optimized plot handling.
    Supports both legacy plot_data and new figure_paths approach.
    """
    try:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                elif message["role"] == "tool":
                    # Handle tool messages (from Python REPL)
                    content = message.get("content", "")
                    if content.strip():
                        st.code(content, language="text")
                    
                    # Handle figure paths from tool messages
                    if "figure_paths" in message and message["figure_paths"]:
                        try:
                            OptimizedStreamlitPlotHandler.display_plots_from_paths(message["figure_paths"])
                        except Exception as e:
                            st.error(f"Error displaying plots from tool message: {str(e)}")
                    
                    # Display errors if present
                    if message.get("error"):
                        st.error(message["error"])
                        
                else:
                    # Handle assistant messages
                    content = message.get("content", "")
                    
                    # Display text content if it exists
                    if content and content.strip():
                        st.markdown(content)
                    
                    # Handle legacy figures (backward compatibility)
                    if "figures" in message and message["figures"]:
                        try:
                            StreamlitPlotHandler.display_plots_from_files(message["figures"])
                        except Exception as e:
                            st.error(f"Error displaying legacy figures: {str(e)}")
                    
                    # Handle new optimized figure paths
                    if "figure_paths" in message and message["figure_paths"]:
                        try:
                            OptimizedStreamlitPlotHandler.display_plots_from_paths(message["figure_paths"])
                        except Exception as e:
                            st.error(f"Error displaying figure paths: {str(e)}")
                    
                    # Handle legacy plot data (backward compatibility)
                    if "plot_data" in message and message["plot_data"]:
                        try:
                            StreamlitPlotHandler.display_plots_from_result({"plot_data": message["plot_data"]})
                        except Exception as e:
                            st.error(f"Error displaying legacy plot data: {str(e)}")
                    
                    # Handle expanders with better error handling
                    for expander in message.get("expanders", []):
                        try:
                            if expander.get("type") == "dataset_info":
                                links_df = display_debug_info_final(
                                    "Detailed information on datasets",
                                    expander["detailed_summary"],
                                    expander.get("download_opendap", False),
                                )
                                if expander.get("download_opendap") and not links_df.empty:
                                    display_opendap_links(links_df)
                                display_python_code(expander["query_for_python_code"])
                            elif expander.get("type") == "debug_info":
                                display_debug_info(expander["title"], expander["content"], store=False)
                        except Exception as e:
                            st.error(f"Error displaying expander: {str(e)}")
                            
    except Exception as e:
        st.error(f"Error displaying chat messages: {str(e)}")

def handle_user_input(agent_executor):
    """
    Handles the user's input and generates a response from the AI assistant.
    Updated to work with the optimized agent that uses path-based plot storage.
    """
    if user_input := st.chat_input("What would you like to know about climate data or CMIP6?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            collected_plot_data = []
            collected_figures: list[str] = []
            collected_figure_paths: list[str] = []
            
            try:
                for chunk in agent_executor.stream(
                    {"input": user_input, "chat_history": st.session_state.messages}
                ):
                    # Handle streamed text
                    if isinstance(chunk, dict) and "output" in chunk:
                        full_response += chunk["output"]
                        message_placeholder.markdown(full_response + "▌")

                    # Handle legacy plot data (backward compatibility)
                    if isinstance(chunk, dict) and "plot_data" in chunk:
                        collected_plot_data.extend(chunk["plot_data"])
                        StreamlitPlotHandler.display_plots_from_result({"plot_data": chunk["plot_data"]})

                    # Handle legacy file-based figures (backward compatibility)
                    if isinstance(chunk, dict) and "figures" in chunk and chunk["figures"]:
                        collected_figures.extend(chunk["figures"])
                        StreamlitPlotHandler.display_plots_from_files(chunk["figures"])
                    
                    # Handle new optimized figure paths
                    if isinstance(chunk, dict) and "figure_paths" in chunk and chunk["figure_paths"]:
                        collected_figure_paths.extend(chunk["figure_paths"])
                        OptimizedStreamlitPlotHandler.display_plots_from_paths(chunk["figure_paths"])

                # Clear the typing indicator
                message_placeholder.markdown(full_response)

            except Exception as e:
                error_message = f"An error occurred: {str(e)}\n\nPlease try rephrasing your query or contact support if the issue persists."
                st.error(error_message)
                full_response = error_message
                message_placeholder.markdown(full_response)

            # Add AI response to chat history with all possible plot formats
            expanders = st.session_state.get("pending_expanders", [])
            message_data = {
                "role": "assistant",
                "content": full_response,
                "expanders": expanders,
            }
            
            # Add plot data in appropriate format
            if collected_plot_data:
                message_data["plot_data"] = collected_plot_data
            if collected_figures:
                message_data["figures"] = collected_figures
            if collected_figure_paths:
                message_data["figure_paths"] = collected_figure_paths
                
            st.session_state.messages.append(message_data)
            st.session_state.pending_expanders = []

def format_chat_history(chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Formats the chat history into a readable string for use in prompts or logs.
    Updated to handle tool messages and figure paths.
    """
    if chat_history is None:
        chat_history = st.session_state.get('messages', [])
    formatted_history = ""
    
    for message in chat_history:
        role = message["role"]
        if role == "user":
            role_label = "User"
        elif role == "assistant":
            role_label = "Assistant"
        elif role == "tool":
            role_label = "Tool"
        else:
            role_label = role.capitalize()
            
        content = message.get("content", "")
        if content is None:
            if "tool_calls" in message:
                content = f'Tool calls: {message["tool_calls"]}'
            elif message.get("role") == "tool":
                content = "Tool execution result"
            else:
                content = "No content"
                
        formatted_history += f"{role_label}: {content}\n"
        
        # Add information about plots if present
        if message.get("figure_paths"):
            formatted_history += f"  [Generated {len(message['figure_paths'])} plot(s)]\n"
        elif message.get("figures"):
            formatted_history += f"  [Generated {len(message['figures'])} figure(s)]\n"
        elif message.get("plot_data"):
            formatted_history += f"  [Generated {len(message['plot_data'])} plot(s)]\n"
            
    return formatted_history

def display_debug_info(title, content, store: bool = True):
    """
    Displays debugging information in an expandable section.
    """
    with st.expander(f"{title}", expanded=False):
        st.json(content)
    if store and "pending_expanders" in st.session_state:
        st.session_state.pending_expanders.append(
            {
                "type": "debug_info",
                "title": title,
                "content": content,
            }
        )

def display_debug_info_final(title, content, download_opendap=False):
    """
    Displays debugging information in an expandable table format and collects
    all OpenDAP links across models.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON content: {e}")
        return pd.DataFrame()
        
    all_model_links = []
    
    with st.expander(f"{title}", expanded=False):
        st.write(f"Total datasets found: {data.get('hit_count', 0)}")
        
        # Initialize session state for OpenDAP links if not exists
        if 'opendap_links' not in st.session_state:
            st.session_state.opendap_links = {}
            
        for model_name, model_data in data.get('models', {}).items():
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
                    'Values': str(model_data.get('dataset_count', 0)),
                    'Details': ''
                })
                
                for param, values in model_data.items():
                    if param != 'dataset_count' and isinstance(values, dict):
                        value_str = ', '.join([f"{k}: {v}" for k, v in values.items()])
                        total_count = sum(values.values()) if values else 0
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
                    if model_name not in st.session_state.opendap_links:
                        # Extract all parameters with multiple values
                        multi_value_parameters = {}
                        parameter_values = {}
                        
                        # Default member_id
                        parameter_values['member_id'] = ['r1i1p1f1']
                        
                        # Extract all parameter values and track those with multiple values
                        for param, values in model_data.items():
                            if param != 'dataset_count' and values and isinstance(values, dict):
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
                        
                        if parameter_values:
                            generate_combinations(parameter_values, 0, {})
                        
                        # Preferred nodes in order of priority
                        preferred_nodes = ["aims3.llnl.gov", "esgf-data1.llnl.gov", "esgf-data2.llnl.gov"]
                        
                        # Dictionary to organize links by unique ID
                        all_results = []
                        
                        # Fetch OpenDAP links for each parameter combination
                        if param_combinations:
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
            
            st.write("---")
    
    # Return DataFrame of all collected links
    return pd.DataFrame(all_model_links)

def display_opendap_links(df: pd.DataFrame) -> None:
    """
    Display a pandas DataFrame containing unique OpenDAP links in Streamlit.
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
    """Display Python code for accessing CMIP6 data from Google Cloud Storage."""
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
    
    with st.expander("Python access from Google Cloud Storage", expanded=False):
        st.header("CMIP6 Data Access Code") 
        st.write("This code loads climate model data from Google Cloud Storage using Zarr format.")
        st.code(code, language='python')
    return code

# ESGF Search Function
# Author: Unknown
# Original version from ESGF documentation
# https://docs.google.com/document/d/1pxz1Kd3JHfFp8vR2JCVBfApbsHmbUQQstifhGNdc6U0/edit?usp=sharing
# API AT: https://github.com/ESGF/esgf.github.io/wiki/ESGF_Search_REST_API#results-pagination

def esgf_search(server="https://esgf-node.llnl.gov/esg-search/search",
                files_type="OPENDAP", local_node=True, project="CMIP6",
                verbose=False, format="application%2Fsolr%2Bjson",
                use_csrf=False, **search):
    """
    Search for CMIP6 data files using the ESGF API.
    
    Args:
        server: ESGF search server URL
        files_type: Type of files to search for (default: "OPENDAP")
        local_node: Whether to search only local node (default: True)
        project: Project name (default: "CMIP6")
        verbose: Whether to print verbose output (default: False)
        format: Response format (default: "application%2Fsolr%2Bjson")
        use_csrf: Whether to use CSRF token (default: False)
        **search: Additional search parameters
        
    Returns:
        List of sorted file URLs
    """
    client = requests.session()
    payload = search
    payload["project"] = project
    payload["type"] = "File"
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
        if verbose:
            print(url)
            
        try:
            r = client.get(url)
            r.raise_for_status()
            resp = r.json()["response"]
            numFound = int(resp["numFound"])
            resp = resp["docs"]
            offset += len(resp)
            
            for d in resp:
                if verbose:
                    for k in d:
                        print("{}: {}".format(k, d[k]))
                url_list = d.get("url", [])
                for f in url_list:
                    sp = f.split("|")
                    if len(sp) > 1 and sp[-1] == files_type:
                        all_files.append(sp[0].split(".html")[0])
                        
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data from ESGF: {e}")
            break
        except (KeyError, json.JSONDecodeError) as e:
            st.error(f"Error parsing response from ESGF: {e}")
            break
            
    return sorted(all_files)