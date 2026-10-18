from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

from app.config import config


def chat_model_is_configured() -> bool:
    if config.CHOOSE_CHAT_MODEL == "deepseek":
        return bool(config.DEEPSEEK_API_KEY)
    if config.CHOOSE_CHAT_MODEL == "openai":
        return bool(config.MS_API_KEY)
    raise RuntimeError(f"不支持的聊天模型配置: {config.CHOOSE_CHAT_MODEL}")


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    if config.CHOOSE_CHAT_MODEL == "deepseek":
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置")
        return ChatDeepSeek(
            model=config.CHAT_MODEL_NAME,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            temperature=0,
        )
    if config.CHOOSE_CHAT_MODEL == "openai":
        if not config.MS_API_KEY:
            raise RuntimeError("MS_API_KEY 未配置")
        return ChatOpenAI(
            model=config.MS_MODEL_NAME,
            api_key=config.MS_API_KEY,
            base_url=config.MS_BASE_URL,
            temperature=0,
        )
    raise RuntimeError(f"不支持的聊天模型配置: {config.CHOOSE_CHAT_MODEL}")


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    return HuggingFaceBgeEmbeddings(
        model_name=config.HF_EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_instruction="为这个句子生成表示以用于检索相关文章：",
    )


@lru_cache(maxsize=1)
def chroma_db() -> Chroma:
    return Chroma(
        collection_name="lab-agent-bge-small-zh-v1-5",
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_DB_PATH,
    )
