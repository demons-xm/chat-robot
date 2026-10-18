from fastapi import APIRouter

from app.api.agent import router as agent_router
from app.api.evaluation import router as evaluation_router
from app.api.intent import router as intent_router
from app.api.memory import router as memory_router
from app.api.rag import router as rag_router


api = APIRouter(prefix="/api/v1")
api.include_router(agent_router)
api.include_router(intent_router)
api.include_router(rag_router)
api.include_router(memory_router)
api.include_router(evaluation_router)
