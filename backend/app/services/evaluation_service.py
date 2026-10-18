import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.factory import chat_model_is_configured, get_chat_model
from app.config import BASE_DIR

logger = logging.getLogger("app")
EVALUATION_DB = Path(BASE_DIR) / "data" / "agent_evaluations.sqlite3"


def initialize_evaluation_store() -> None:
    EVALUATION_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(EVALUATION_DB, timeout=10) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                reference TEXT,
                scores_json TEXT NOT NULL,
                average_score REAL NOT NULL,
                judge_reason TEXT NOT NULL,
                quality_tag TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def _quality_tag(average: float) -> str:
    if average >= 4:
        return "优质"
    if average >= 2.5:
        return "一般"
    return "不合格"


def evaluate_sample(query: str, answer: str, reference: str | None = None) -> dict[str, Any]:
    if not chat_model_is_configured():
        raise RuntimeError("LLM-as-Judge不可用：聊天模型API密钥未配置")
    prompt = (
        "你是客服问答质量评测员。按0到5分评估相关性 relevance、准确性 accuracy、"
        "完整性 completeness、有用性 usefulness。准确性需以参考文本为准；参考文本为空时"
        "仅判断回答是否有明显事实臆测。只输出JSON，格式为"
        '{"score":{"relevance":0,"accuracy":0,"completeness":0,"usefulness":0},'
        '"judgeReason":"简短中文理由"}。\n'
        f"用户问题：{query}\n客服回答：{answer}\n参考文本：{reference or '未提供'}"
    )
    response = get_chat_model().invoke([SystemMessage(content="严格按指定JSON格式返回。"), HumanMessage(content=prompt)])
    try:
        result = json.loads(str(response.content))
        raw_scores = result["score"]
        scores = {
            name: float(raw_scores[name])
            for name in ("relevance", "accuracy", "completeness", "usefulness")
        }
        if any(not 0 <= score <= 5 for score in scores.values()):
            raise ValueError("评分超出0到5范围")
        reason = str(result["judgeReason"]).strip()
        if not reason:
            raise ValueError("评测理由为空")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.error("LLM-as-Judge返回格式无效: %s", exc)
        raise RuntimeError("LLM-as-Judge返回格式无效") from exc

    average = round(sum(scores.values()) / len(scores), 2)
    evaluation_id = uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(EVALUATION_DB, timeout=10) as connection:
        connection.execute(
            """
            INSERT INTO evaluations
            (id, query, answer, reference, scores_json, average_score, judge_reason, quality_tag, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evaluation_id,
                query,
                answer,
                reference,
                json.dumps(scores, ensure_ascii=False),
                average,
                reason,
                _quality_tag(average),
                created_at,
            ),
        )
    result = {
        "evaluationId": evaluation_id,
        "score": scores,
        "avgScore": average,
        "judgeReason": reason,
        "qualityTag": _quality_tag(average),
    }
    logger.info(
        "llm_judge",
        extra={"avg_score": average, "quality_tag": result["qualityTag"]},
    )
    return result


def evaluate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for index, sample in enumerate(samples):
        try:
            result = evaluate_sample(
                query=sample["query"],
                answer=sample["answer"],
                reference=sample.get("reference"),
            )
            result["sampleId"] = sample.get("sample_id") or str(index + 1)
            results.append(result)
        except Exception as exc:
            logger.exception("批量回归评测样本失败 (index=%s)", index, exc_info=exc)
            results.append(
                {
                    "sampleId": sample.get("sample_id") or str(index + 1),
                    "error": "样本评测失败",
                }
            )
    completed = [item for item in results if "avgScore" in item]
    averages = [item["avgScore"] for item in completed]
    return {
        "total": len(samples),
        "completed": len(completed),
        "failed": len(samples) - len(completed),
        "avgScore": round(sum(averages) / len(averages), 2) if averages else None,
        "results": results,
    }


def log_async_evaluation_failure(exc: Exception) -> None:
    logger.warning("异步LLM评测失败，不影响问答结果: %s", exc, exc_info=exc)
