import logging

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.common.response import ApiResponse
from app.schemas.platform import IntentRequest
from app.services.agent_service import classify_intent, preprocess_query
from app.services.memory_service import get_session_context
from app.services.rate_limit_service import enforce_tenant_rate_limit

logger = logging.getLogger("app")
router = APIRouter(prefix="/intent", tags=["意图识别"])


@router.post("/classify", summary="客服意图识别")
async def intent_classify(req: IntentRequest, request: Request):
    enforce_tenant_rate_limit(request, req.tenant_id)
    query, _ = preprocess_query(req.query)
    if not query:
        raise HTTPException(status_code=400, detail="query清洗后不能为空")
    context = (
        await get_session_context(req.tenant_id, req.session_id)
        if req.session_id
        else []
    )
    result = await run_in_threadpool(classify_intent, query, context[-6:])
    logger.info(
        "intent_classify",
        extra={
            "tenant_id": req.tenant_id,
            "session_id": req.session_id,
            "intent": result["intentLabel"],
            "confidence": result["confidence"],
        },
    )
    return ApiResponse.success(data=result)
