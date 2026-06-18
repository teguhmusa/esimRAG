"""
config/settings.py
Central configuration using Pydantic Settings.
All values can be overriden via environment variables or .env file.
"""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Project paths ──────────────────────────────────────────────
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = Path(__file__).parent.parent / "data"
    logs_dir: Path = Path(__file__).parent.parent / "logs"

    # ── LLM ───────────────────────────────────────────────────────
    llm_provider: str = Field(default="anthropic", env="LLM_PROVIDER")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    llm_model: str = Field(default="claude-opus-4-8", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1024, env="LLM_MAX_TOKENS")

    # ── Embedding ─────────────────────────────────────────────────
    embedding_model: str = Field(default="BAAI/bge-large-en-v1.5", env="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", env="EMBEDDING_DEVICE")

    # ── Vector DB ─────────────────────────────────────────────────
    vector_db_type: str = Field(default="chroma", env="VECTOR_DB_TYPE")
    chroma_persist_dir: str = Field(default="./data/chroma_db", env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(default="sgp22_knowledge_base", env="CHROMA_COLLECTION_NAME")

    # ── Retrieval ─────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=5, env="RETRIEVAL_TOP_K")
    retrieval_score_threshold: float = Field(default=0.3, env="RETRIEVAL_SCORE_THRESHOLD")

    # ── Chunking ──────────────────────────────────────────────────
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, env="CHUNK_OVERLAP")

    # ── Logging ───────────────────────────────────────────────────
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/esim_rag.log", env="LOG_FILE")

    class Config:
        env_file = ".env.txt"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton — import this class in all module.
settings = Settings()

# Auto-create direktori yang dibutuhkan
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.logs_dir.mkdir(parents=True, exist_ok=True)
(settings.data_dir / "raw").mkdir(exist_ok=True)