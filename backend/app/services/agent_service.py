import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from threading import Lock
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.factory import chat_model_is_configured, get_chat_model, get_embeddings
from app.agent.rag.fusion import reciprocal_rank_fusion
from app.config import config

logger = logging.getLogger("app")

DEFAULT_INTENT_KEYWORDS = {
    "转人工": ("转人工", "人工客服", "人工服务", "找客服", "真人"),
    "投诉": ("投诉", "举报", "差劲", "不满意", "欺骗", "态度差"),
    "建议": ("建议", "希望增加", "改进", "优化", "意见反馈"),
    "故障报修": ("故障", "报修", "无法使用", "不能用", "坏了", "错误代码"),
    "订单咨询": ("订单", "物流", "发货", "快递", "退款", "退货", "订单号"),
    "配送咨询": ("配送", "快递公司", "承运商", "配送方式", "配送范围", "配送时效"),
    "业务咨询": ("如何", "怎么", "规则", "流程", "政策", "办理", "产品", "服务"),
    "身份咨询": ("你是谁", "你叫什么", "你是什么助手", "你能做什么"),
}
DEFAULT_INTENT_EXAMPLES = {
    "业务咨询": "请介绍业务办理流程和产品服务规则",
    "投诉": "我要投诉，商品质量很差，对服务不满意",
    "建议": "我有一个改进建议，希望增加新的功能",
    "转人工": "请帮我转人工客服，我要和真人沟通",
    "闲聊": "你好，今天天气不错，我们聊聊天吧",
    "故障报修": "设备坏了无法使用，请帮我报修处理故障",
    "订单咨询": "请查询我的订单什么时候发货，物流到哪里了",
    "配送咨询": "我们支持哪些快递公司和配送方式",
    "身份咨询": "你是谁，你能做什么",
}
_HUMAN_REQUEST_NEGATION = re.compile(
    r"(?:不想|不需要|不用|不要|无需|暂时不|先不).{0,5}"
    r"(?:转人工|转接人工|找人工|联系人工|人工客服|真人客服)",
    re.IGNORECASE,
)
_EXPLICIT_HUMAN_REQUEST = re.compile(
    r"(?:转接?|切换到|接通)(?:一下|到)?(?:人工|真人)(?:客服|服务)?"
    r"|(?:找|联系|接)(?:一下|一位|一个)?(?:人工|真人)(?:客服|服务)?"
    r"|(?:我想|我要|麻烦|帮我|给我).{0,5}"
    r"(?:转人工|转接人工|找人工|联系人工|人工客服|真人客服|找客服|联系客服)"
    r"|(?:人工客服|真人客服).{0,4}(?:帮我|接待|联系|转接)",
    re.IGNORECASE,
)
_SPECIAL_INTENT_PATTERNS = (
    (
        "配送咨询",
        re.compile(
            r"(?:你们|你|平台|店里)?\s*(?:都)?\s*"
            r"(?:支持|有哪些|有哪几种|哪几家|哪些|什么|可以使用|可以选择|能用|合作)"
            r".{0,12}(?:发货方式|配送方式|快递公司|快递|物流公司|承运商)"
            r"|(?:发货方式|配送方式|快递公司|快递|物流公司|承运商)"
            r".{0,12}(?:有哪些|哪几家|哪些|支持什么|支持哪些|可以选择)"
            r"|(?:你们|平台|店里).{0,8}(?:怎么发货|用什么快递|发什么快递)",
            re.IGNORECASE,
        ),
        "配送方式或承运商支持查询",
    ),
    (
        "订单咨询",
        re.compile(
            r"(?:物流状态|订单状态|查物流|查询物流|物流到哪|快递到哪|包裹到哪|"
            r"订单.{0,4}(?:状态|物流|到哪)|"
            r"(?:订单号|快递单号|运单号).{0,8}(?:不记得|记不得|忘了|不知道|没有)|"
            r"(?:不记得|记不得|忘了|不知道|没有).{0,8}(?:订单号|快递单号|运单号))",
            re.IGNORECASE,
        ),
        "个人订单物流状态查询",
    ),
    (
        "身份咨询",
        re.compile(
            r"(?:你|您)(?:是谁|叫什么(?:名字)?|是什么(?:助手|机器人|智能体)?"
            r"|能做什么|可以做什么)[啊呢吗呀？?。！!]*$",
            re.IGNORECASE,
        ),
        "智能客服身份咨询",
    ),
)
_vector_store = None
_vector_store_lock = Lock()
_NEGATION_MARKERS = ("不记得", "记不得", "不确定", "不知道", "不能", "无法", "没有", "不想", "不需要", "不用", "不要", "未")
_DELIVERY_STATUS_QUERY = re.compile(
    r"(?:物流状态|订单状态|查物流|查询物流|物流到哪|快递到哪|包裹到哪|"
    r"订单.{0,4}(?:状态|物流|到哪)|到哪里了|到哪了|发到哪)",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _intent_configuration() -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    rules = dict(DEFAULT_INTENT_KEYWORDS)
    examples = dict(DEFAULT_INTENT_EXAMPLES)
    try:
        if config.INTENT_KEYWORDS_JSON:
            configured_rules = json.loads(config.INTENT_KEYWORDS_JSON)
            if not isinstance(configured_rules, dict):
                raise ValueError("INTENT_KEYWORDS_JSON必须是JSON对象")
            for label, keywords in configured_rules.items():
                if not isinstance(keywords, list) or not all(isinstance(item, str) for item in keywords):
                    raise ValueError(f"意图“{label}”的关键词必须是字符串数组")
                rules[str(label)] = tuple(keywords)
        if config.INTENT_EXAMPLES_JSON:
            configured_examples = json.loads(config.INTENT_EXAMPLES_JSON)
            if not isinstance(configured_examples, dict) or not all(
                isinstance(label, str) and isinstance(example, str)
                for label, example in configured_examples.items()
            ):
                raise ValueError("INTENT_EXAMPLES_JSON必须是字符串到字符串的JSON对象")
            examples.update(configured_examples)
    except ValueError as exc:
        raise RuntimeError(f"意图识别配置无效: {exc}") from exc
    for label, patterns in rules.items():
        try:
            for pattern in patterns:
                re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise RuntimeError(f"意图“{label}”包含无效关键词正则: {exc}") from exc
    for label in rules:
        examples.setdefault(label, label)
    return {label: tuple(keywords) for label, keywords in rules.items()}, examples


def preprocess_query(query: str, extra_entity: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", query)
    cleaned = re.sub(r"[^\w\s\u4e00-\u9fff.,!?，。！？;；:：'\"“”‘’()（）\-_#]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()[:2000]
    for informal, normalized in (("咋", "怎么"), ("啥", "什么"), ("发货没", "是否发货")):
        cleaned = cleaned.replace(informal, normalized)
    entities: dict[str, Any] = {}
    phone_match = re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", cleaned)
    order_match = re.search(
        r"\b(?:OD|ORD|ORDER)[-_]?[A-Z0-9]{6,}\b"
        + ("" if phone_match else r"|\b\d{10,20}\b"),
        cleaned,
        re.I,
    )
    if order_match:
        entities["orderNo"] = order_match.group(0)
    if phone_match:
        entities["phone"] = phone_match.group(0)
    product_match = re.search(r"\b(?:SKU|SPU|P)[-_]?[A-Z0-9]{4,}\b", cleaned, re.I)
    if product_match:
        entities["productId"] = product_match.group(0)
    entities.update(extra_entity or {})
    return cleaned, entities


def _llm_intent(query: str, context: list[dict[str, str]] | None = None) -> tuple[str | None, float, str]:
    if not chat_model_is_configured():
        return None, 0.0, ""
    _, examples = _intent_configuration()
    labels = list(examples)
    prompt = (
        "你是客服意图分类器。以当前用户问题为主要依据，从以下标签中选择一个："
        + "、".join(labels)
        + '。近期会话仅用于理解当前问题中的省略和指代，不得把历史意图直接当作当前意图。'
        '只有当前问题明确请求转人工时才选择“转人工”。'
        '严格输出JSON：{"intentLabel":"标签","confidence":0到1之间的数字,"subIntent":"子意图"}。'
    )
    try:
        response = get_chat_model().invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=(
                        f"近期会话上下文：{json.dumps(context or [], ensure_ascii=False)}\n"
                        f"当前问题：{query}"
                    )
                ),
            ]
        )
        data = json.loads(str(response.content))
        label = data.get("intentLabel")
        confidence = float(data.get("confidence", 0))
        if label not in labels or not 0 <= confidence <= 1:
            return None, 0.0, ""
        return label, confidence, str(data.get("subIntent", ""))
    except (ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        import logging

        logging.getLogger("app").warning("LLM意图分类不可用，降级至规则与向量: %s", exc)
        return None, 0.0, ""
    except Exception as exc:
        import logging

        logging.getLogger("app").warning("LLM意图分类调用失败，降级至规则与向量: %s", exc)
        return None, 0.0, ""


@lru_cache(maxsize=1)
def _intent_prototypes() -> dict[str, list[float]]:
    _, examples = _intent_configuration()
    embeddings = get_embeddings()
    labels = list(examples)
    vectors = embeddings.embed_documents([examples[label] for label in labels])
    return {label: vector for label, vector in zip(labels, vectors)}


def _embedding_scores(query: str) -> dict[str, float]:
    vectors = _intent_prototypes()
    query_vector = get_embeddings().embed_query(query)
    scores = {}
    for label, vector in vectors.items():
        dot = sum(a * b for a, b in zip(query_vector, vector))
        norm_query = math.sqrt(sum(value * value for value in query_vector))
        norm_vector = math.sqrt(sum(value * value for value in vector))
        similarity = dot / (norm_query * norm_vector) if norm_query and norm_vector else 0.0
        scores[label] = max(0.0, min(1.0, (similarity - 0.35) / 0.65))
    return scores


def _is_explicit_human_request(query: str) -> bool:
    if _HUMAN_REQUEST_NEGATION.search(query):
        return False
    return bool(_EXPLICIT_HUMAN_REQUEST.search(query))


def _deterministic_intent(query: str) -> dict[str, Any] | None:
    if _is_explicit_human_request(query):
        return {
            "intentLabel": "转人工",
            "confidence": 1.0,
            "subIntent": "用户明确请求人工服务",
            "hitKeywords": ["明确请求人工"],
            "scoreDetail": {
                "llmFewShotScore": 0.0,
                "embeddingScore": 0.0,
                "regexKeywordScore": 1.0,
            },
            "needConfirm": False,
            "transferHuman": True,
        }
    for label, pattern, sub_intent in _SPECIAL_INTENT_PATTERNS:
        match = pattern.search(query)
        if match:
            return {
                "intentLabel": label,
                "confidence": 0.95,
                "subIntent": sub_intent,
                "hitKeywords": [match.group(0)],
                "scoreDetail": {
                    "llmFewShotScore": 0.0,
                    "embeddingScore": 0.0,
                    "regexKeywordScore": 0.95,
                },
                "needConfirm": False,
                "transferHuman": False,
            }
    return None


def classify_intent(query: str, context: list[dict[str, str]] | None = None) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise ValueError("query不能为空")
    deterministic = _deterministic_intent(query)
    if deterministic:
        return deterministic
    keyword_rules, examples = _intent_configuration()

    keyword_hits = {
        label: [pattern for pattern in patterns if re.search(pattern, query, re.IGNORECASE)]
        for label, patterns in keyword_rules.items()
    }
    keyword_hits = {label: words for label, words in keyword_hits.items() if words}
    # Mentioning a human agent is not the same as explicitly asking to be transferred.
    keyword_hits.pop("转人工", None)
    keyword_label = max(keyword_hits, key=lambda label: len(keyword_hits[label])) if keyword_hits else None
    keyword_scores = {label: (0.85 if label == keyword_label else 0.0) for label in examples}

    llm_label, llm_confidence, sub_intent = _llm_intent(query, context)
    llm_scores = {label: (llm_confidence if label == llm_label else 0.0) for label in examples}

    embedding_scores = {label: 0.0 for label in examples}
    try:
        embedding_scores = _embedding_scores(query)
    except Exception as exc:
        import logging

        logging.getLogger("app").warning("意图Embedding不可用，降级至LLM与关键词: %s", exc)

    conversational = len(query) > 35 or any(word in query for word in ("啊", "呢", "呀", "嘛", "吧"))
    weights = (0.5, 0.3, 0.2) if conversational else (0.25, 0.25, 0.5)
    available = [bool(llm_label), bool(embedding_scores and any(embedding_scores.values())), True]
    active_weights = [weight if enabled else 0.0 for weight, enabled in zip(weights, available)]
    total_weight = sum(active_weights) or 1.0
    scores = {}
    for label in examples:
        scores[label] = (
            active_weights[0] * llm_scores[label]
            + active_weights[1] * embedding_scores[label]
            + active_weights[2] * keyword_scores[label]
        ) / total_weight

    label = max(scores, key=scores.get)
    confidence = round(scores[label], 4)
    if label == "转人工":
        non_human_scores = {key: value for key, value in scores.items() if key != "转人工"}
        if non_human_scores:
            label = max(non_human_scores, key=non_human_scores.get)
            confidence = round(scores[label], 4)
    if not keyword_hits and not llm_label and max(embedding_scores.values(), default=0) < 0.15:
        label = "闲聊"
        confidence = 0.3
    need_confirm = config.INTENT_LOW_THRESHOLD <= confidence < config.INTENT_HIGH_THRESHOLD
    return {
        "intentLabel": label,
        "confidence": confidence,
        "subIntent": sub_intent or (keyword_hits.get(label) or [""])[0],
        "hitKeywords": keyword_hits.get(label, []),
        "scoreDetail": {
            "llmFewShotScore": round(llm_scores.get(label, 0.0), 4),
            "embeddingScore": round(embedding_scores.get(label, 0.0), 4),
            "regexKeywordScore": round(keyword_scores.get(label, 0.0), 4),
        },
        "needConfirm": need_confirm,
        "transferHuman": False,
    }


def _context_text(context: list[dict[str, str]]) -> str:
    return "\n".join(
        str(item.get("content", ""))
        for item in context[-10:]
        if item.get("role") in {"user", "assistant", "system"}
    )


def _infer_query_type(
    query: str,
    intent_label: str,
    context: list[dict[str, str]],
) -> str:
    if intent_label in {"配送咨询", "身份咨询"}:
        return "general_knowledge"
    if _DELIVERY_STATUS_QUERY.search(query):
        return "personal_order_status"
    if re.search(r"(?:订单号|快递单号|运单号).{0,8}(?:不记得|记不得|忘了|不知道|没有)", query):
        return "personal_order_status"
    if re.search(r"(?:不记得|记不得|忘了|不知道|没有).{0,8}(?:订单号|快递单号|运单号)", query):
        previous = _context_text(context)
        if _DELIVERY_STATUS_QUERY.search(previous) or re.search(
            r"(?:查|查询|查看).{0,6}(?:订单|物流|快递)|订单.{0,6}(?:状态|物流)",
            previous,
        ):
            return "personal_order_status"
    if re.search(r"(?:我的|我这个|本人).{0,8}(?:订单|快递|包裹).{0,8}(?:状态|物流|到哪|发货)", query):
        return "personal_order_status"
    return "general_knowledge"


def _parse_model_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型未返回JSON对象")
    result = json.loads(text[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("模型响应不是JSON对象")
    return result


def _llm_understand_query(
    query: str,
    context: list[dict[str, str]],
    entities: dict[str, Any],
) -> dict[str, Any] | None:
    if not chat_model_is_configured():
        return None
    _, examples = _intent_configuration()
    labels = list(examples)
    prompt = (
        "你是智能客服的对话理解器。请以当前用户原话为准，结合上下文解释省略和指代，"
        "不要把上下文中已解决的问题当成当前问题。不要编造用户没有表达的实体、需求或事实。"
        "将当前问题整理成适合知识库检索的独立查询 standaloneQuery；改写只能补足上下文，"
        "不能改变用户目标、否定、实体、时间或查询范围。"
        "intentLabel必须从给定标签中选择；queryType只能是general_knowledge、"
        "personal_order_status或other。requiredSlots和missingSlots为字符串数组；"
        "entities为JSON对象。只返回JSON对象，不要Markdown。"
        f"\n意图标签：{json.dumps(labels, ensure_ascii=False)}"
        '\n格式：{"intentLabel":"标签","confidence":0.0,"subIntent":"子意图",'
        '"standaloneQuery":"独立查询","queryType":"general_knowledge",'
        '"requiredSlots":[],"missingSlots":[],"entities":{}}'
    )
    try:
        response = get_chat_model().invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=(
                        f"对话上下文：{json.dumps(context[-10:], ensure_ascii=False)}\n"
                        f"已提取实体：{json.dumps(entities, ensure_ascii=False)}\n"
                        f"当前用户原话：{query}"
                    )
                ),
            ]
        )
        result = _parse_model_json(str(response.content))
        label = result.get("intentLabel")
        confidence = float(result.get("confidence", 0))
        standalone_query = result.get("standaloneQuery")
        query_type = result.get("queryType")
        required_slots = result.get("requiredSlots", [])
        missing_slots = result.get("missingSlots", [])
        model_entities = result.get("entities", {})
        if (
            label not in labels
            or not 0 <= confidence <= 1
            or not isinstance(standalone_query, str)
            or not standalone_query.strip()
            or query_type not in {"general_knowledge", "personal_order_status", "other"}
            or not isinstance(required_slots, list)
            or not all(isinstance(slot, str) for slot in required_slots)
            or not isinstance(missing_slots, list)
            or not all(isinstance(slot, str) for slot in missing_slots)
            or not isinstance(model_entities, dict)
        ):
            raise ValueError("模型对话理解结果字段无效")
        return {
            "intentLabel": label,
            "confidence": confidence,
            "subIntent": str(result.get("subIntent", "")),
            "standaloneQuery": standalone_query.strip()[:2000],
            "queryType": query_type,
            "requiredSlots": required_slots[:10],
            "missingSlots": missing_slots[:10],
            "entities": model_entities,
        }
    except Exception as exc:
        logger.warning("LLM对话理解失败，回退到规则和原始问题: %s", exc)
        return None


def _rewrite_preserves_explicit_facts(
    query: str,
    rewritten_query: str,
    entities: dict[str, Any],
) -> bool:
    if not rewritten_query.strip():
        return False
    original_negated = any(marker in query for marker in _NEGATION_MARKERS)
    rewritten_negated = any(marker in rewritten_query for marker in _NEGATION_MARKERS)
    if original_negated != rewritten_negated:
        return False
    original_intent = _deterministic_intent(query)
    rewritten_intent = _deterministic_intent(rewritten_query)
    if (
        original_intent
        and original_intent["intentLabel"] == "配送咨询"
        and _DELIVERY_STATUS_QUERY.search(rewritten_query)
    ):
        return False
    if (
        _DELIVERY_STATUS_QUERY.search(query)
        and rewritten_intent
        and rewritten_intent["intentLabel"] == "配送咨询"
    ):
        return False
    for value in entities.values():
        if isinstance(value, str) and value and value.casefold() not in rewritten_query.casefold():
            return False
    return True


def _verify_query_rewrite(
    query: str,
    rewritten_query: str,
    context: list[dict[str, str]],
    entities: dict[str, Any],
) -> tuple[bool, str]:
    if query.strip() == rewritten_query.strip():
        return True, "改写与原问题一致"
    if not _rewrite_preserves_explicit_facts(query, rewritten_query, entities):
        return False, "改写丢失或改变了原问题中的否定或实体"
    if not chat_model_is_configured():
        return False, "没有可用模型验证改写"
    prompt = (
        "你是严格的客服查询改写审核器。比较原始问题、上下文和改写问题。"
        "判断改写是否仍要求解决同一件事，并保留否定、用户已知/未知信息、实体、"
        "时间、对象、范围和不确定性。根据上下文补全省略可以接受；新增未被上下文支持的"
        "事实、把一般政策问题变成个人订单查询、把个人状态查询变成一般政策问题均不接受。"
        '只返回JSON：{"consistent":true,"reason":"简短原因"}。'
    )
    try:
        response = get_chat_model().invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=(
                        f"上下文：{json.dumps(context[-10:], ensure_ascii=False)}\n"
                        f"实体：{json.dumps(entities, ensure_ascii=False)}\n"
                        f"原始问题：{query}\n改写问题：{rewritten_query}"
                    )
                ),
            ]
        )
        result = _parse_model_json(str(response.content))
        consistent = result.get("consistent")
        reason = result.get("reason", "")
        if not isinstance(consistent, bool) or not isinstance(reason, str):
            raise ValueError("语义校验响应字段无效")
        return consistent, reason[:300]
    except Exception as exc:
        logger.warning("查询改写语义校验失败，使用原始问题: %s", exc)
        return False, "语义校验不可用，已回退原始问题"


def understand_query(
    query: str,
    context: list[dict[str, str]] | None = None,
    entities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify the current turn, resolve context, and safely prepare its retrieval query."""
    context = context or []
    entities = entities or {}
    deterministic = _deterministic_intent(query)
    model_result = None if deterministic else _llm_understand_query(query, context, entities)
    if (
        model_result
        and model_result["intentLabel"] == "转人工"
        and not _is_explicit_human_request(query)
    ):
        model_result = None
    if model_result:
        intent_info = {
            "intentLabel": model_result["intentLabel"],
            "confidence": round(model_result["confidence"], 4),
            "subIntent": model_result["subIntent"],
            "hitKeywords": [],
            "scoreDetail": {
                "llmFewShotScore": round(model_result["confidence"], 4),
                "embeddingScore": 0.0,
                "regexKeywordScore": 0.0,
            },
            "needConfirm": (
                config.INTENT_LOW_THRESHOLD
                <= model_result["confidence"]
                < config.INTENT_HIGH_THRESHOLD
            ),
            "transferHuman": model_result["intentLabel"] == "转人工",
        }
        rewritten_query = model_result["standaloneQuery"]
        query_type = model_result["queryType"]
        required_slots = model_result["requiredSlots"]
        missing_slots = model_result["missingSlots"]
        model_entities = model_result["entities"]
    else:
        intent_info = classify_intent(query, context)
        rewritten_query = query
        query_type = "other"
        required_slots = []
        missing_slots = []
        model_entities = {}

    if deterministic:
        intent_info = deterministic
        rewritten_query = (
            model_result["standaloneQuery"] if model_result else query
        )
    if intent_info["intentLabel"] in {"配送咨询", "身份咨询"}:
        query_type = "general_knowledge"
    else:
        query_type = _infer_query_type(query, intent_info["intentLabel"], context)
        context_mentions_orders = bool(
            re.search(r"(?:订单|物流|快递|包裹)", _context_text(context))
        )
        if (
            query_type == "general_knowledge"
            and model_result
            and model_result["queryType"] == "personal_order_status"
            and (intent_info["intentLabel"] == "订单咨询" or context_mentions_orders)
        ):
            query_type = "personal_order_status"
    if query_type == "personal_order_status":
        intent_info = {
            **intent_info,
            "intentLabel": "订单咨询",
            "subIntent": "个人订单物流状态查询",
        }

    merged_entities = {**model_entities, **entities}
    if query_type == "personal_order_status":
        required_slots = list(dict.fromkeys([*required_slots, "orderNo"]))
        if not merged_entities.get("orderNo"):
            missing_slots = list(dict.fromkeys([*missing_slots, "orderNo"]))
        else:
            missing_slots = [slot for slot in missing_slots if slot != "orderNo"]

    is_consistent, verification_reason = _verify_query_rewrite(
        query,
        rewritten_query,
        context,
        merged_entities,
    )
    if not is_consistent:
        rewritten_query = query
    return {
        "intentInfo": intent_info,
        "originalQuery": query,
        "rewrittenQuery": rewritten_query,
        "rewriteConsistent": is_consistent,
        "rewriteVerification": verification_reason,
        "queryType": query_type,
        "entities": merged_entities,
        "requiredSlots": required_slots,
        "missingSlots": missing_slots,
    }


def route_agent(intent_label: str, entities: dict[str, Any], query: str) -> dict[str, Any]:
    if intent_label == "转人工":
        target = "HumanTransferAgent"
        reason = "用户明确请求人工服务"
    elif intent_label == "配送咨询":
        target = "BusinessConsultAgent"
        reason = "用户咨询配送方式或承运商支持范围"
    elif intent_label == "身份咨询":
        target = "GeneralQAAgent"
        reason = "用户询问智能客服身份或能力"
    elif entities.get("orderNo"):
        target = "OrderConsultAgent"
        reason = "命中订单号实体，优先路由至订单Agent"
    elif entities.get("phone") or entities.get("productId"):
        target = "BusinessConsultAgent"
        reason = "命中业务实体，路由至咨询Agent"
    elif intent_label == "订单咨询":
        target = "OrderConsultAgent"
        reason = "意图为订单咨询，路由至订单Agent"
    elif intent_label == "投诉":
        target = "ComplaintAgent"
        reason = "意图为投诉，路由至投诉Agent"
    elif intent_label == "故障报修":
        target = "FaultRepairAgent"
        reason = "意图为故障报修，路由至故障处理Agent"
    elif intent_label == "建议":
        target = "SuggestionAgent"
        reason = "意图为建议，路由至建议受理Agent"
    elif intent_label not in {
        "业务咨询", "投诉", "建议", "转人工", "闲聊", "故障报修",
        "订单咨询", "配送咨询", "身份咨询",
    } and any(word in query for word in ("订单", "物流", "发货", "退款")):
        target = "OrderConsultAgent"
        reason = "意图未明确，关键词命中订单场景，路由至订单Agent"
    elif intent_label not in {
        "业务咨询", "投诉", "建议", "转人工", "闲聊", "故障报修",
        "订单咨询", "配送咨询", "身份咨询",
    } and any(word in query for word in ("投诉", "不满意", "举报")):
        target = "ComplaintAgent"
        reason = "意图未明确，关键词命中投诉场景，路由至投诉Agent"
    elif intent_label == "闲聊":
        target = "GeneralQAAgent"
        reason = "闲聊由通用问答Agent处理"
    else:
        target = "BusinessConsultAgent" if intent_label == "业务咨询" else "GeneralQAAgent"
        reason = f"按意图“{intent_label}”路由至{target}"
    return {"targetAgent": target, "routeReason": reason, "fallback": target == "GeneralQAAgent"}


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:
                from app.agent.vector_store.vector_store import VectorStore

                _vector_store = VectorStore()
    return _vector_store


def _rewrite_queries(query: str) -> list[str]:
    if not chat_model_is_configured():
        return [query]
    try:
        result = get_chat_model().invoke(
            [
                SystemMessage(content="把客服问题改写成最多3条适合知识库检索的短查询，每行一条，不要编号或解释。"),
                HumanMessage(content=query),
            ]
        )
        candidates = [line.strip(" -*\t") for line in str(result.content).splitlines() if line.strip()]
        unique = list(dict.fromkeys([query, *candidates]))
        return unique[:3]
    except Exception as exc:
        import logging

        logging.getLogger("app").warning("Query改写失败，使用原始Query: %s", exc)
        return [query]


def retrieve_documents(
    query: str,
    tenant_id: str,
    top_k: int = 3,
    enable_query_rewrite: bool = True,
    query_variants: list[str] | None = None,
) -> dict[str, Any]:
    if query_variants is not None:
        queries = list(
            dict.fromkeys(
                [query, *(item.strip() for item in query_variants if item.strip())]
            )
        )[:3]
    else:
        queries = _rewrite_queries(query) if enable_query_rewrite else [query]
    vector_store = _get_vector_store()
    raw_docs: dict[str, Any] = {}
    rankings: list[list[str]] = []

    def search_vectors(rewritten: str):
        return vector_store.vector_store.similarity_search(
            rewritten,
            k=max(top_k * 3, config.TOP_K),
            filter={"tenant_id": tenant_id},
        )

    def search_keywords(rewritten: str):
        return vector_store.search_bm25()(rewritten)

    with ThreadPoolExecutor(max_workers=max(2, len(queries) * 2)) as executor:
        pending = [
            (
                rewritten,
                executor.submit(search_vectors, rewritten),
                executor.submit(search_keywords, rewritten),
            )
            for rewritten in queries
        ]
        retrieval_results = []
        for rewritten, vector_future, keyword_future in pending:
            try:
                vector_docs = vector_future.result()
            except Exception as exc:
                logger.warning("租户向量检索失败，保留关键词检索结果: %s", exc)
                vector_docs = []
            try:
                keyword_docs = keyword_future.result()
            except Exception as exc:
                logger.warning("BM25检索失败，保留向量检索结果: %s", exc)
                keyword_docs = []
            retrieval_results.append((vector_docs, keyword_docs))

    for vector_docs, keyword_docs in retrieval_results:
        keyword_docs = [doc for doc in keyword_docs if doc.metadata.get("tenant_id") == tenant_id]
        for result_set in (vector_docs, keyword_docs):
            ranking = []
            for rank, doc in enumerate(result_set, start=1):
                if doc.metadata.get("tenant_id") != tenant_id:
                    continue
                doc_id = str(
                    doc.metadata.get(
                        "doc_id",
                        f"{doc.metadata.get('path', '')}#{doc.metadata.get('chunk', rank)}",
                    )
                )
                ranking.append(doc_id)
                raw_docs[doc_id] = {
                    "docId": doc_id,
                    "content": doc.page_content,
                }
            if ranking:
                rankings.append(ranking)

    ranked_ids = reciprocal_rank_fusion(rankings, top_k=len(raw_docs))
    fused_scores = {}
    for doc_id in ranked_ids:
        fused_scores[doc_id] = sum(
            1 / (60 + source.index(doc_id) + 1)
            for source in rankings
            if doc_id in source
        )
    max_fused_score = max(fused_scores.values(), default=1.0)
    for doc_id in ranked_ids:
        raw_docs[doc_id]["score"] = round(fused_scores[doc_id] / max_fused_score, 4)
    ranked_docs = [raw_docs[doc_id] for doc_id in ranked_ids]
    raw_retrieve_list = [dict(item) for item in ranked_docs]
    if chat_model_is_configured() and ranked_docs:
        try:
            candidates = [
                {"docId": item["docId"], "content": item["content"][:1200]}
                for item in ranked_docs[: max(top_k * 3, top_k)]
            ]
            response = get_chat_model().invoke(
                [
                    SystemMessage(
                        content="根据查询和候选知识片段的相关性排序。只输出按从高到低排列的docId JSON数组。"
                    ),
                    HumanMessage(
                        content=json.dumps(
                            {"query": query, "documents": candidates},
                            ensure_ascii=False,
                        )
                    ),
                ]
            )
            ordered_ids = json.loads(str(response.content))
            if not isinstance(ordered_ids, list):
                raise ValueError("重排结果不是数组")
            valid_ids = {item["docId"] for item in ranked_docs}
            ordered_ids = [doc_id for doc_id in ordered_ids if doc_id in valid_ids]
            order = list(dict.fromkeys([*ordered_ids, *[item["docId"] for item in ranked_docs]]))
            by_id = {item["docId"]: item for item in ranked_docs}
            ranked_docs = [by_id[doc_id] for doc_id in order]
        except Exception as exc:
            logger.warning("LLM文档重排失败，使用融合排序: %s", exc)
    if config.RERANK and ranked_docs:
        from langchain_core.documents import Document

        scores_by_id = {item["docId"]: item["score"] for item in ranked_docs}
        docs = [
            Document(page_content=item["content"], metadata={"doc_id": item["docId"], "tenant_id": tenant_id})
            for item in ranked_docs
        ]
        try:
            reranked = vector_store.reranker.rerank(query, docs, top_k)
            ranked_docs = [
                {
                    "docId": doc.metadata["doc_id"],
                    "content": doc.page_content,
                    "score": scores_by_id[doc.metadata["doc_id"]],
                }
                for doc in reranked
            ]
        except Exception as exc:
            logger.warning("RAG重排失败，使用融合排序: %s", exc)
    return {
        "rewriteQueries": queries,
        "rawRetrieveList": raw_retrieve_list[: max(top_k * 3, top_k)],
        "rerankDocs": ranked_docs[:top_k],
        "mrr": None,
    }


def generate_answer(
    query: str,
    history: list[dict[str, str]],
    reference_docs: list[dict[str, Any]],
    agent_name: str = "GeneralQAAgent",
    entities: dict[str, Any] | None = None,
) -> str:
    context = "\n\n".join(doc["content"] for doc in reference_docs)
    prompt = (
        f"你是智能客服Agent中的{agent_name}，请按对应业务场景自然、友好地处理用户请求。"
        "结合上下文理解用户当前目标，不要机械重复已经问过的问题。"
        "回答必须以参考资料为事实依据；没有相关资料时明确说明未找到，"
        "不得编造规则、订单状态或承诺。若资料不足但可以给出部分答案，先回答已知部分，"
        "再只提出一个必要且具体的澄清问题。历史对话仅用于理解上下文，不得覆盖当前问题。\n"
        f"参考资料：\n{context or '（未检索到相关知识）'}\n"
        f"历史对话：\n{json.dumps(history, ensure_ascii=False)}\n"
        f"业务实体：\n{json.dumps(entities or {}, ensure_ascii=False)}\n"
        f"用户问题：{query}\n请用简洁、礼貌的中文回答。"
    )
    try:
        response = get_chat_model().invoke([HumanMessage(content=prompt)])
    except RuntimeError as exc:
        raise RuntimeError("LLM服务未配置或不可用") from exc
    answer = str(response.content).strip()
    if not answer:
        raise RuntimeError("LLM返回了空回答")
    return answer


def make_session_summary(query: str, answer: str) -> str:
    return f"用户：{query[:500]}；客服：{answer[:1000]}"[:1500]
