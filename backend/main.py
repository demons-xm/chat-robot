import asyncio
import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api import api
from app.config import config
from app.common.exceptions import (
    BusinessException,
    business_exception_handler,
    global_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from app.common.response import ApiResponse
from app.services.evaluation_service import initialize_evaluation_store
from app.services.rate_limit_service import allow_request


from app.services.agent_service import _get_vector_store

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    initialize_evaluation_store()
    try:
        await asyncio.to_thread(_get_vector_store)
    except Exception:
        logger.exception("知识库初始化失败，问答服务将以降级模式启动")
    yield


app = FastAPI(title="智能客服Agent（LLM问答平台）", version="1.0.0", lifespan=lifespan)
app.include_router(api)
app.add_exception_handler(BusinessException, business_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        client_ip = request.client.host if request.client else "unknown"
        if not allow_request(f"ip:{client_ip}"):
            response = JSONResponse(
                status_code=429,
                content=ApiResponse.error(code=429, message="请求过于频繁，请稍后重试").model_dump(by_alias=True),
            )
            response.headers["X-Request-ID"] = uuid4().hex
            return response
    response = await call_next(request)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in config.CORS_ALLOWED_ORIGINS.split(",")
        if origin.strip()
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["健康检查"])
async def root():
    return ApiResponse.success(message="智能客服Agent服务运行中")
