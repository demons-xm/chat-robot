from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import config


def to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class ChatRequest(ApiModel):
    session_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=10000)
    tenant_id: str = Field(min_length=1, max_length=128)
    channel: str | None = Field(default=None, max_length=32)
    extra_entity: dict[str, Any] = Field(default_factory=dict)
    timeout: int = Field(default=config.DEFAULT_CHAT_TIMEOUT_MS, ge=100, le=120000)

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query不能为空")
        return value


class IntentRequest(ApiModel):
    query: str = Field(min_length=1, max_length=10000)
    tenant_id: str = Field(min_length=1, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)


class RagRetrieveRequest(ApiModel):
    query: str = Field(min_length=1, max_length=10000)
    tenant_id: str = Field(min_length=1, max_length=128)
    top_k: int = Field(default=3, ge=1, le=20)
    enable_query_rewrite: bool = True


class AgentRouteRequest(ApiModel):
    intent_label: str = Field(min_length=1, max_length=128)
    entity: dict[str, Any] = Field(default_factory=dict)
    tenant_id: str = Field(min_length=1, max_length=128)
    query: str = Field(default="", max_length=10000)


class MemoryKeyRequest(ApiModel):
    session_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)


class MemorySaveRequest(MemoryKeyRequest):
    query: str = Field(min_length=1, max_length=10000)
    answer: str = Field(min_length=1, max_length=20000)
    session_summary: str = Field(default="", max_length=4000)
    user_features: dict[str, Any] | list[str] = Field(default_factory=dict)


class JudgeRequest(ApiModel):
    query: str = Field(min_length=1, max_length=10000)
    answer: str = Field(min_length=1, max_length=20000)
    reference: str | None = Field(default=None, max_length=20000)


class JudgeSample(JudgeRequest):
    sample_id: str | None = Field(default=None, max_length=128)


class BatchJudgeRequest(ApiModel):
    samples: list[JudgeSample] = Field(min_length=1, max_length=100)
