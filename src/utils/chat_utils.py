"""
Chat utilities — Streamlit-free.
Pure data functions for formatting, parsing, and ESGF search.
All rendering is handled by the React frontend.
"""
from typing import List, Optional, Dict, Any
import os
import re
import json
import requests


def parse_image_markdown(text: str) -> tuple[str, Optional[str]]:
    """Extract image path from markdown and return cleaned text and absolute path."""
    match = re.search(r"!\[.*?\]\((.*?)\)", text)
    if match:
        path = match.group(1)
        full_path = os.path.join(os.getcwd(), "temp_figures", os.path.basename(path))
        clean_text = re.sub(r"!\[.*?\]\(.*?\)", "", text).strip()
        return clean_text, full_path
    return text, None


def format_chat_history(chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Formats chat history into a readable string for use in prompts or logs.
    """
    if chat_history is None:
        chat_history = []
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

        if message.get("figure_paths"):
            formatted_history += f"  [Generated {len(message['figure_paths'])} plot(s)]\n"
        elif message.get("figures"):
            formatted_history += f"  [Generated {len(message['figures'])} figure(s)]\n"
        elif message.get("plot_data"):
            formatted_history += f"  [Generated {len(message['plot_data'])} plot(s)]\n"

    return formatted_history


def parse_dataset_info(content: str) -> Dict[str, Any]:
    """
    Parse detailed dataset summary JSON into a structured dict for the frontend.
    Replaces the old display_debug_info_final Streamlit function.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Failed to parse dataset info", "raw": content}

    result = {
        "hit_count": data.get("hit_count", 0),
        "models": {}
    }

    for model_name, model_data in data.get("models", {}).items():
        model_info = {
            "dataset_count": model_data.get("dataset_count", 0),
            "parameters": {}
        }
        for param, values in model_data.items():
            if param != "dataset_count" and isinstance(values, dict):
                model_info["parameters"][param] = {
                    "total": sum(values.values()) if values else 0,
                    "breakdown": values
                }
        result["models"][model_name] = model_info

    return result


def generate_python_code(query: str) -> str:
    """Generate Python code for accessing CMIP6 data from Google Cloud Storage."""
    return f"""import pandas as pd
import xarray as xr
# Load the metadata CSV
df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
# Filter the dataframe
df_spec = df.query("{query}")
# get the path to a specific zarr store (the first one from the dataframe above)
zstore = df_spec.zstore.values[-1]
# Open the first dataset
ds = xr.open_zarr(zstore, consolidated=True, storage_options={{'token':'anon'}})"""


# ESGF Search Function
def esgf_search(server="https://esgf-node.llnl.gov/esg-search/search",
                files_type="OPENDAP", local_node=True, project="CMIP6",
                verbose=False, format="application%2Fsolr%2Bjson",
                use_csrf=False, **search):
    """
    Search for CMIP6 data files using the ESGF API.
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
            csrftoken = client.cookies['csrftoken']
        else:
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
            print(f"Error fetching data from ESGF: {e}")
            break
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing response from ESGF: {e}")
            break

    return sorted(all_files)