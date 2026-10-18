from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

from app.common.response import ApiResponse

logger = logging.getLogger("app")


class BusinessException(Exception):
    """自定义业务异常"""
    def __init__(self, message: str, code: int = 500):
        super().__init__(message)
        self.message = message
        self.code = code


async def business_exception_handler(request: Request, exc: BusinessException):
    status_code = exc.code if 400 <= exc.code <= 599 else 500
    return JSONResponse(
        status_code=status_code,
        content=ApiResponse.error(code=exc.code, message=exc.message).model_dump(by_alias=True)
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """
    http 异常处理器，处理 fastapi 原生 HTTPException
    HTTP 协议状态码，放在响应头，浏览器 / 网关能读到
    HTTP 状态码（404、401、403 这类）
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=ApiResponse.error(code=exc.status_code, message=str(exc.detail)).model_dump(by_alias=True)
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    参数（validation）异常处理器，处理 fastapi.exceptions 的 RequestValidationError
    把异常里的code、message传给Response.error
    """
    return JSONResponse(
        status_code=400,
        content=ApiResponse.error(code=400, message="请求参数错误").model_dump(by_alias=True)
    )


async def global_exception_handler(request: Request, exc: Exception):
    """
    全局异常处理器，处理服务器内部错误，原生 Exception
    把异常里的code、message传给Response.error
    """
    logger.exception("未处理请求异常", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content=ApiResponse.error(code=500, message="服务器内部错误").model_dump(by_alias=True)
    )


if __name__ == '__main__':
    exce = BusinessException(message="111")
    print(exce.message, exce.code)
