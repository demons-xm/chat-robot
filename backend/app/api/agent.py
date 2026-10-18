import asyncio
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import require_token
from app.common.response import ApiResponse
from app.schemas.platform import AgentRouteRequest, ChatRequest
from app.services.agent_service import (
    generate_answer,
    make_session_summary,
    preprocess_query,
    retrieve_documents,
    route_agent,
    understand_query,
)
from app.services.evaluation_service import evaluate_sample, log_async_evaluation_failure
from app.services.memory_service import enqueue_memory_save, get_memory
from app.services.rate_limit_service import enforce_tenant_rate_limit

logger = logging.getLogger("app")
router = APIRouter(prefix="/agent", tags=["智能客服Agent"])


@router.post("/chat", summary="智能客服统一问答入口")
async def agent_chat(
    req: ChatRequest,
    request: Request,
    _claims: dict = Depends(require_token),
):
    token_user_id = _claims.get("userId", _claims.get("user_id", _claims.get("sub")))
    if token_user_id is None or str(token_user_id) != req.user_id:
        raise HTTPException(status_code=401, detail="Token用户与请求userId不匹配")
    enforce_tenant_rate_limit(request, req.tenant_id)
    started = time.perf_counter()

    async def process():
        query, entities = preprocess_query(req.query, req.extra_entity)
        if not query:
            raise HTTPException(status_code=400, detail="query清洗后不能为空")
        memory = await get_memory(req.tenant_id, req.user_id, req.session_id)
        intent_context = list(memory["workingMemory"][-6:])
        if memory["episodicMemory"]:
            intent_context.insert(
                0,
                {
                    "role": "system",
                    "content": "历史会话摘要（仅用于理解当前问题）：" + "\n".join(memory["episodicMemory"][-3:]),
                },
            )
        query_understanding = await run_in_threadpool(
            understand_query, query, intent_context, entities
        )
        intent_info = query_understanding["intentInfo"]
        intent_info["queryType"] = query_understanding["queryType"]
        intent_info["rewriteConsistent"] = query_understanding["rewriteConsistent"]
        understood_entities = query_understanding["entities"]
        routing = route_agent(intent_info["intentLabel"], understood_entities, query)
        history = list(memory["workingMemory"])
        if memory["episodicMemory"]:
            history.insert(
                0,
                {
                    "role": "system",
                    "content": "历史会话摘要（仅用于上下文理解）：" + "\n".join(memory["episodicMemory"]),
                },
            )
        if memory["userProfile"]:
            history.insert(
                0,
                {"role": "system", "content": "用户画像：" + str(memory["userProfile"])},
            )

        should_transfer = intent_info["transferHuman"]
        is_live_order_query = query_understanding["queryType"] == "personal_order_status"
        should_confirm = (
            intent_info["needConfirm"]
            and not should_transfer
            and not is_live_order_query
        )
        if should_transfer:
            answer = "我已记录您的需求，建议为您转接人工客服进一步处理。"
        elif is_live_order_query:
            answer = (
                "理解，忘了订单号也没关系。不过目前我无法直接连接订单或物流系统查询实时状态，"
                "也不能仅凭手机号或姓名确认具体包裹。您可以在下单平台的“我的订单”中查看物流，"
                "或联系该平台官方客服；如果您想了解一般配送方式或配送范围，我也可以直接为您介绍。"
            )
        elif should_confirm:
            if intent_info["intentLabel"] == "业务咨询":
                answer = "为了更准确地回答，您想了解哪项产品、服务或办理流程？"
            else:
                answer = "我还不太确定您的问题。可以再补充一下具体想了解的内容吗？"
        else:
            original_query = query_understanding["originalQuery"]
            standalone_query = query_understanding["rewrittenQuery"]
            rag = await run_in_threadpool(
                retrieve_documents,
                standalone_query,
                req.tenant_id,
                3,
                False,
                [original_query] if standalone_query != original_query else None,
            )
            answer_history = history
            if standalone_query != original_query:
                answer_history = [
                    *history,
                    {
                        "role": "system",
                        "content": f"用户当前原话：{original_query}；经语义校验的上下文独立查询：{standalone_query}",
                    },
                ]
            answer = await run_in_threadpool(
                generate_answer,
                standalone_query,
                answer_history,
                rag["rerankDocs"],
                routing["targetAgent"],
                understood_entities,
            )
        if should_transfer or should_confirm or is_live_order_query:
            rag = {"rerankDocs": []}
        summary = make_session_summary(query, answer)
        enqueue_memory_save(
            {
                "tenant_id": req.tenant_id,
                "user_id": req.user_id,
                "session_id": req.session_id,
                "query": query,
                "answer": answer,
                "session_summary": summary,
                "user_features": memory["userProfile"],
            }
        )
        evaluation_task = asyncio.create_task(
            asyncio.to_thread(evaluate_sample, query, answer, "\n".join(
                doc["content"] for doc in rag["rerankDocs"]
            ) or None)
        )
        evaluation_task.add_done_callback(
            lambda completed: log_async_evaluation_failure(completed.exception())
            if not completed.cancelled() and completed.exception()
            else None
        )
        logger.info(
            "agent_chat",
            extra={
                "tenant_id": req.tenant_id,
                "session_id": req.session_id,
                "intent": intent_info["intentLabel"],
                "confidence": intent_info["confidence"],
                "agent": routing["targetAgent"],
                "reference_count": len(rag["rerankDocs"]),
                "query_type": query_understanding["queryType"],
                "rewrite_consistent": query_understanding["rewriteConsistent"],
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return {
            "answer": answer,
            "intentInfo": intent_info,
            "agentName": routing["targetAgent"],
            "referenceDocs": rag["rerankDocs"],
            "queryUnderstanding": {
                "originalQuery": query_understanding["originalQuery"],
                "rewrittenQuery": query_understanding["rewrittenQuery"],
                "rewriteConsistent": query_understanding["rewriteConsistent"],
                "queryType": query_understanding["queryType"],
                "requiredSlots": query_understanding["requiredSlots"],
                "missingSlots": query_understanding["missingSlots"],
            },
            "transferHuman": should_transfer,
            "transferReason": (
                "用户明确请求转人工" if should_transfer else ""
            ),
            "sessionSummary": summary,
        }

    try:
        data = await asyncio.wait_for(process(), timeout=req.timeout / 1000)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="问答请求超时") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ApiResponse.success(data=data)


@router.post("/router", summary="多Agent路由决策")
async def agent_router(req: AgentRouteRequest, request: Request):
    enforce_tenant_rate_limit(request, req.tenant_id)
    result = route_agent(req.intent_label, req.entity, req.query)
    logger.info(
        "agent_route",
        extra={"tenant_id": req.tenant_id, "agent": result["targetAgent"]},
    )
    return ApiResponse.success(data=result)
