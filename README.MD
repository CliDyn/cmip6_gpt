# CMIP6 GPT

**CMIP6 GPT** is an AI-powered assistant designed to help users access CMIP6 climate data through natural language conversations. Built using OpenAI's GPT models, LangChain, and Streamlit.

## Features

- **Natural Language Interface**: Interact with CMIP6 data using conversational queries.
- **Dynamic Data Retrieval**: Leverages vector search and language models to fetch relevant climate data.


### Setup
1. Create a virtual environment and install the required packages:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Configure API Keys. Create a .streamlit/secrets.toml file in the project root directory with the following content:
```toml
# .streamlit/secrets.toml
[openai]
api_key = "your-openai-api-key"

[LANGCHAIN]
# Add any LangChain-specific configurations here if necessary
LANGCHAIN_TRACING_V2 = 'true'
LANGCHAIN_ENDPOINT = 'https://api.smith.langchain.com'
LANGCHAIN_API_KEY = 'your-langhcain-api-key'
```

### Usage
Run the Streamlit application:
```bash
streamlit run main.py
```
Open your web browser and navigate to http://localhost:8501 to access the application.
