import logging

from fastapi import APIRouter, Request

from app.common.response import ApiResponse
from app.schemas.platform import MemoryKeyRequest, MemorySaveRequest
from app.services.memory_service import enqueue_memory_save, get_memory
from app.services.rate_limit_service import enforce_tenant_rate_limit

logger = logging.getLogger("app")
router = APIRouter(prefix="/memory", tags=["三级记忆"])


@router.post("/get", summary="读取会话记忆")
async def memory_get(req: MemoryKeyRequest, request: Request):
    enforce_tenant_rate_limit(request, req.tenant_id)
    memory = await get_memory(req.tenant_id, req.user_id, req.session_id)
    logger.info(
        "memory_op",
        extra={"operation": "read", "tenant_id": req.tenant_id, "user_id": req.user_id},
    )
    return ApiResponse.success(data=memory)


@router.post("/save", summary="异步写入会话记忆")
async def memory_save(req: MemorySaveRequest, request: Request):
    enforce_tenant_rate_limit(request, req.tenant_id)
    enqueue_memory_save(
        {
            "tenant_id": req.tenant_id,
            "user_id": req.user_id,
            "session_id": req.session_id,
            "query": req.query,
            "answer": req.answer,
            "session_summary": req.session_summary,
            "user_features": req.user_features,
        }
    )
    logger.info(
        "memory_op",
        extra={"operation": "enqueue_save", "tenant_id": req.tenant_id, "user_id": req.user_id},
    )
    return ApiResponse.success(message="记忆写入任务已入队", data={"queued": True})
