import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_PATH = DATA_DIR / "knowledge"
LOG_PATH = DATA_DIR / "logger"
HISTORY_PATH = DATA_DIR / "history"

for directory in (KNOWLEDGE_PATH, LOG_PATH, HISTORY_PATH):
    directory.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    HISTORY_PATH: str = str(HISTORY_PATH)
    CHROMA_DB_PATH: str = str(DATA_DIR / "chroma_db")
    MD5_PATH: str = str(DATA_DIR / "md5_bge_small_zh_v1_5.txt")
    KNOWLEDGE_PATH: Path = KNOWLEDGE_PATH
    LOG_PATH: str = str(LOG_PATH)

    CHOOSE_CHAT_MODEL: str = "deepseek"
    CHAT_MODEL_NAME: str = "deepseek-chat"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_API_KEY: str = ""
    MS_MODEL_NAME: str = "deepseek-chat"
    MS_BASE_URL: str = "https://api.deepseek.com/v1"
    MS_API_KEY: str = ""
    HF_EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-zh-v1.5"

    JWT_SECRET_KEY: str = ""
    JWT_EXPIRE_HOURS: int = 24
    JWT_ALGORITHM: str = "HS256"
    CORS_ALLOWED_ORIGINS: str = "http://127.0.0.1:5173,http://localhost:5173"

    REDIS_URL: str = "redis://127.0.0.1:6379/0"
    MEMORY_TTL_SECONDS: int = 1800
    MEMORY_WINDOW_TURNS: int = 10
    DEFAULT_TENANT_ID: str = "default"
    RATE_LIMIT_PER_MINUTE: int = 120
    DEFAULT_CHAT_TIMEOUT_MS: int = 8000
    INTENT_HIGH_THRESHOLD: float = 0.75
    INTENT_LOW_THRESHOLD: float = 0.4
    INTENT_KEYWORDS_JSON: str = ""
    INTENT_EXAMPLES_JSON: str = ""

    MAX_CONTEXT: int = 4000
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 30
    SEPARATORS: list[str] = ["\n\n", "。", ".", "?", "？", "!", " ", ""]
    TOP_K: int = 10
    USE_BM25: bool = True
    RERANK: bool = False
    RERANK_TOP_K: int = 5
    RERANK_MODEL: str = "BAAI/bge-reranker-base"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


config = Settings()
