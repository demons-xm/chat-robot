import logging

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app.common.response import ApiResponse
from app.schemas.platform import BatchJudgeRequest, JudgeRequest
from app.services.evaluation_service import evaluate_batch, evaluate_sample

logger = logging.getLogger("app")
router = APIRouter(prefix="/eval", tags=["LLM自动评测"])


@router.post("/llmjudge", summary="LLM-as-Judge问答质量评测")
async def llm_judge(req: JudgeRequest):
    try:
        result = await run_in_threadpool(
            evaluate_sample, req.query, req.answer, req.reference
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    logger.info(
        "llm_judge",
        extra={"avg_score": result["avgScore"], "quality_tag": result["qualityTag"]},
    )
    return ApiResponse.success(data=result)


@router.post("/batch", summary="批量回归评测")
async def batch_evaluate(req: BatchJudgeRequest):
    result = await run_in_threadpool(
        evaluate_batch,
        [sample.model_dump() for sample in req.samples],
    )
    return ApiResponse.success(data=result)
