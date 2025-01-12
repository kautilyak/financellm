import os
from dotenv import load_dotenv


class Config:
    def __init__(self):
        load_dotenv()

        self.alpha_vantage_key = self._get_required_env("ALPHA_VANTAGE_API_KEY")

        # Vector store settings
        self.vector_store_path = os.getenv("VECTOR_STORE_PATH", "../data/")

        # Model settings
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "llama3.2")
        self.llm_model = os.getenv("LLM_MODEL", "llama3.2")

    def _get_required_env(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Missing required environment variable: {key}")
        return value