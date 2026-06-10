import os
import yaml
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load .env at import time
load_dotenv()


def _load_config_yaml() -> dict:
    """Load config.yaml from project root."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


_CONFIG_DATA = _load_config_yaml()


class Config:
    model_name: str = _CONFIG_DATA.get("llm", {}).get("default_model", "gemini-3.1-pro-preview-vertex")
    # Per-request RAG knobs (updated by server before each request)
    rag_chunks_per_search: int = 10
    rag_num_searches: int = 5
    # Reviewer models (set per-request by server, separate for each reviewer)
    # ⚠️ INDEPENDENCE NOTE: Using the same model for both reviewers provides
    # zero epistemic independence — they share identical biases, blind spots,
    # and hallucination patterns. For genuine defense-in-depth, use different
    # model families (e.g., gemini + claude, or gemini + gpt).
    # The current defaults use the same model with different role prompts,
    # which provides some diversity but NOT true independence.
    reviewer_model_1: str = "gemini-3.1-pro-preview"
    reviewer_model_2: str = "gemini-3.1-pro-preview"
    reviewers_enabled: bool = True
    REVIEWER_MODELS = ["gemini-3.1-pro-preview", "claude-opus-4-8", "gpt-5.5"]

    @classmethod
    def set_model_name(cls, model_name: str):
        cls.model_name = model_name

    @classmethod
    def get_model_name(cls) -> str:
        return cls.model_name

    @classmethod
    def get_openai_api_key(cls) -> str:
        return os.environ.get("OPENAI_API_KEY", "")

    @classmethod
    def get_retrievers_dir(cls) -> str:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        retrievers_dir = os.path.join(project_root, 'retrievers')
        return retrievers_dir

    # --- Config accessors ---

    @classmethod
    def get_rag_config(cls) -> dict:
        return _CONFIG_DATA.get("rag", {})

    @classmethod
    def get_retriever_paths(cls) -> Dict[str, str]:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        paths = _CONFIG_DATA.get("rag", {}).get("retrievers", {
            "variable_id": "retrievers/chroma_langchain_db",
            "source_id": "retrievers/chroma_langchain_db_sources_new1",
            "experiment_id": "retrievers/chroma_langchain_db_exp_new",
        })
        return {k: os.path.join(project_root, v) for k, v in paths.items()}

    @classmethod
    def get_rag_top_k(cls) -> int:
        return _CONFIG_DATA.get("rag", {}).get("top_k", 20)

    @classmethod
    def get_rag_score_threshold(cls) -> float:
        return _CONFIG_DATA.get("rag", {}).get("score_threshold", 0.5)

    @classmethod
    def get_rag_max_candidates(cls) -> int:
        return _CONFIG_DATA.get("rag", {}).get("max_candidates", 10)

    @classmethod
    def get_reranking_config(cls) -> dict:
        return _CONFIG_DATA.get("rag", {}).get("reranking", {"enabled": True, "query_match_boost": 0.3})

    @classmethod
    def get_qdrant_config(cls) -> dict:
        return _CONFIG_DATA.get("rag", {}).get("qdrant", {
            "url": "http://localhost:6333",
            "collections": {
                "variable_id": "cmip6_variables",
                "source_id": "cmip6_sources",
                "experiment_id": "cmip6_experiments",
                "literature": "cmip6_papers",
            }
        })

    @classmethod
    def get_esgf_config(cls) -> dict:
        return _CONFIG_DATA.get("esgf", {
            "search_url": "https://esgf-node.llnl.gov/esg-search/search",
            "web_frontend_url": "https://aims2.llnl.gov/search?",
        })

    @classmethod
    def get_agent_max_iterations(cls) -> int:
        return _CONFIG_DATA.get("agent", {}).get("max_iterations", 30)

    @classmethod
    def get_available_models(cls) -> list:
        return _CONFIG_DATA.get("llm", {}).get("available_models", [
            "gemini-3.1-pro-preview-vertex", "gemini-3.1-pro-preview",
            "gpt-5.2", "gpt-4o", "gpt-4.1", "gpt-4.1-nano", "gpt-4o-mini",
            "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash",
        ])

    @staticmethod
    def infer_provider(model_name: str) -> str:
        """Return 'google' if the model name starts with 'gemini', else 'openai'."""
        return "google" if model_name.startswith("gemini") else "openai"

    @classmethod
    def get_embedding_model(cls) -> str:
        return _CONFIG_DATA.get("llm", {}).get("embedding_model", "text-embedding-3-small")

    @classmethod
    def get_embedding_provider(cls) -> str:
        return _CONFIG_DATA.get("llm", {}).get("embedding_provider", "openai")

    # --- Compression / context-window accessors ---

    @classmethod
    def get_compression_config(cls) -> dict:
        return _CONFIG_DATA.get("compression", {})

    @classmethod
    def get_model_context_limit(cls, model_name: Optional[str] = None) -> int:
        """Return the input-token capacity for a model. Falls back via prefix match
        and finally a conservative 128k default."""
        limits: Dict[str, int] = _CONFIG_DATA.get("model_context_limits", {}) or {}
        name = model_name or cls.model_name
        if name in limits:
            return int(limits[name])
        for key, val in limits.items():
            if key == "default":
                continue
            if name.startswith(key):
                return int(val)
        return int(limits.get("default", 128000))

    # --- Blackboard accessors ---

    @classmethod
    def get_blackboard_config(cls) -> dict:
        return _CONFIG_DATA.get("blackboard", {})

    @classmethod
    def blackboard_enabled(cls) -> bool:
        return bool(_CONFIG_DATA.get("blackboard", {}).get("enabled", True))