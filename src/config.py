import os
class Config:
    model_name: str = "gpt-4o"  # Default model
    @classmethod
    def set_model_name(cls, model_name: str):
        cls.model_name = model_name

    @classmethod
    def set_openai_api_key(cls, api_key: str):
        cls.openai_api_key = api_key

    @classmethod
    def get_model_name(cls) -> str:
        return cls.model_name

    @classmethod
    def get_openai_api_key(cls) -> str:
        return cls.openai_api_key
    @classmethod
    def get_retrievers_dir(cls) -> str:
        # Get the absolute path to the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        retrievers_dir = os.path.join(project_root, 'retrievers')
        return retrievers_dir
