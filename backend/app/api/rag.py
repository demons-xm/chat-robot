import logging

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.common.response import ApiResponse
from app.schemas.platform import RagRetrieveRequest
from app.services.agent_service import preprocess_query, retrieve_documents
from app.services.rate_limit_service import enforce_tenant_rate_limit

logger = logging.getLogger("app")
router = APIRouter(prefix="/rag", tags=["RAG检索"])


@router.post("/retrieve", summary="租户隔离的知识库检索")
async def rag_retrieve(req: RagRetrieveRequest, request: Request):
    enforce_tenant_rate_limit(request, req.tenant_id)
    query, _ = preprocess_query(req.query)
    if not query:
        raise HTTPException(status_code=400, detail="query清洗后不能为空")
    result = await run_in_threadpool(
        retrieve_documents,
        query,
        req.tenant_id,
        req.top_k,
        req.enable_query_rewrite,
    )
    logger.info(
        "rag_retrieve",
        extra={
            "tenant_id": req.tenant_id,
            "retrieved_count": len(result["rawRetrieveList"]),
            "mrr": result["mrr"],
        },
    )
    return ApiResponse.success(data=result)
