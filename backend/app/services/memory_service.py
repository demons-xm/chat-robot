import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from app.config import config

logger = logging.getLogger("app")
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-memory")
_local_lock = Lock()
_local_working_memory: dict[str, tuple[float, list[dict[str, str]]]] = {}
_local_session_users: dict[tuple[str, str], tuple[float, str]] = {}
_redis_client = None
_redis_lock = Lock()
_memory_collection = None
_chroma_lock = Lock()


def _memory_key(tenant_id: str, user_id: str, session_id: str) -> str:
    return f"agent:work:{tenant_id}:{user_id}:{session_id}"


def _session_user_key(tenant_id: str, session_id: str) -> str:
    return f"agent:session-user:{tenant_id}:{session_id}"


def _get_redis():
    global _redis_client
    if _redis_client is None:
        with _redis_lock:
            if _redis_client is None:
                import redis

                _redis_client = redis.Redis.from_url(
                    config.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                )
    return _redis_client


def _get_memory_collection():
    global _memory_collection
    if _memory_collection is None:
        with _chroma_lock:
            if _memory_collection is None:
                from langchain_chroma import Chroma

                from app.agent.factory import get_embeddings

                _memory_collection = Chroma(
                    collection_name="agent_long_term_memory",
                    embedding_function=get_embeddings(),
                    persist_directory=config.CHROMA_DB_PATH,
                )
    return _memory_collection


def _read_working(key: str) -> list[dict[str, str]]:
    try:
        value = _get_redis().get(key)
        if value:
            return json.loads(value)
    except Exception as exc:
        logger.warning("Redis工作记忆读取失败，使用进程内短期缓存: %s", exc)
    now = time.monotonic()
    with _local_lock:
        cached = _local_working_memory.get(key)
        if cached and cached[0] > now:
            return list(cached[1])
        _local_working_memory.pop(key, None)
    return []


def _read_long_term(tenant_id: str, user_id: str, session_id: str) -> tuple[list[str], dict[str, Any]]:
    try:
        collection = _get_memory_collection()
        where = {"$and": [{"tenant_id": tenant_id}, {"user_id": user_id}]}
        episodes = collection.similarity_search(
            f"会话 {session_id} 的历史摘要与关键事件",
            k=5,
            filter={"$and": [*where["$and"], {"memory_type": "episodic"}]},
        )
        profiles = collection.similarity_search(
            "用户画像与偏好",
            k=5,
            filter={"$and": [*where["$and"], {"memory_type": "profile"}]},
        )
        profile: dict[str, Any] = {}
        for doc in profiles:
            try:
                profile.update(json.loads(doc.metadata.get("value", "{}")))
            except (ValueError, TypeError):
                logger.warning("忽略格式异常的用户画像记忆")
        return [doc.page_content for doc in episodes], profile
    except Exception as exc:
        logger.warning("Chroma长期记忆读取失败，降级为工作记忆: %s", exc)
        return [], {}


def _read_memory(tenant_id: str, user_id: str, session_id: str) -> dict[str, Any]:
    key = _memory_key(tenant_id, user_id, session_id)
    work = _read_working(key)
    episodes, profile = _read_long_term(tenant_id, user_id, session_id)
    return {
        "workingMemory": work,
        "episodicMemory": episodes,
        "userProfile": profile,
    }


async def get_memory(tenant_id: str, user_id: str, session_id: str) -> dict[str, Any]:
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor, _read_memory, tenant_id, user_id, session_id
    )


async def get_session_context(tenant_id: str, session_id: str) -> list[dict[str, str]]:
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor,
        _read_session_context,
        tenant_id,
        session_id,
    )


def _read_session_context(tenant_id: str, session_id: str) -> list[dict[str, str]]:
    user_id = None
    try:
        user_id = _get_redis().get(_session_user_key(tenant_id, session_id))
    except Exception as exc:
        logger.warning("Redis会话索引读取失败，检查进程内索引: %s", exc)
    if user_id is None:
        with _local_lock:
            cached = _local_session_users.get((tenant_id, session_id))
            if cached and cached[0] > time.monotonic():
                user_id = cached[1]
            else:
                _local_session_users.pop((tenant_id, session_id), None)
    if user_id is None:
        return []
    return _read_working(_memory_key(tenant_id, user_id, session_id))


def _save_memory(payload: dict[str, Any]) -> None:
    tenant_id = payload["tenant_id"]
    user_id = payload["user_id"]
    session_id = payload["session_id"]
    key = _memory_key(tenant_id, user_id, session_id)
    history = _read_working(key)
    history.extend(
        [
            {"role": "user", "content": payload["query"]},
            {"role": "assistant", "content": payload["answer"]},
        ]
    )
    history = history[-config.MEMORY_WINDOW_TURNS * 2 :]
    saved = False
    try:
        _get_redis().setex(key, config.MEMORY_TTL_SECONDS, json.dumps(history, ensure_ascii=False))
        _get_redis().setex(
            _session_user_key(tenant_id, session_id),
            config.MEMORY_TTL_SECONDS,
            user_id,
        )
        saved = True
    except Exception as exc:
        logger.warning("Redis工作记忆写入失败，降级为进程内短期缓存: %s", exc)
        with _local_lock:
            _local_session_users[(tenant_id, session_id)] = (
                time.monotonic() + config.MEMORY_TTL_SECONDS,
                user_id,
            )
    if not saved:
        with _local_lock:
            _local_working_memory[key] = (
                time.monotonic() + config.MEMORY_TTL_SECONDS,
                history,
            )

    try:
        collection = _get_memory_collection()
        timestamp = datetime.now(timezone.utc).isoformat()
        summary = payload["session_summary"].strip()
        if summary:
            collection.add_texts(
                [summary],
                metadatas=[
                    {
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "session_id": session_id,
                        "memory_type": "episodic",
                        "created_at": timestamp,
                    }
                ],
            )
        features = payload.get("user_features") or {}
        if features:
            feature_dict = features if isinstance(features, dict) else {"tags": features}
            collection.add_texts(
                [json.dumps(feature_dict, ensure_ascii=False)],
                metadatas=[
                    {
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "session_id": session_id,
                        "memory_type": "profile",
                        "value": json.dumps(feature_dict, ensure_ascii=False),
                        "created_at": timestamp,
                    }
                ],
            )
    except Exception as exc:
        logger.warning("Chroma长期记忆写入失败，已保留工作记忆: %s", exc)


def enqueue_memory_save(payload: dict[str, Any]) -> None:
    future = _executor.submit(_save_memory, payload)

    def report_failure(completed):
        exception = completed.exception()
        if exception:
            logger.error("异步记忆写入任务失败: %s", exception, exc_info=exception)

    future.add_done_callback(report_failure)
