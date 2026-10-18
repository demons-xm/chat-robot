from pydantic import BaseModel, ConfigDict, Field
from typing import Any
from uuid import uuid4


class ApiResponse(BaseModel):
    """统一封装的返回格式"""
    code: int
    message: str
    data: Any = None
    request_id: str = Field(default_factory=lambda: uuid4().hex, alias="requestId")

    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def success(cls, message: str = "请求成功", data: Any = None):
        return cls(code=200, message=message, data=data)

    @classmethod
    def error(cls, code: int = 500, message: str = "请求失败"):
        return cls(code=code, message=message)
