# -*- coding: utf-8 -*-
"""bge-reranker 重排：CrossEncoder 对 (query, doc) 打分后重排。
懒加载 + 自动降级：未安装 sentence-transformers 或模型不可用时返回原顺序，不影响主链路。
"""
import os
from langchain_core.documents import Document

from app.config import config
from app.utils.logger import logger


class Reranker:
    def __init__(self) -> None:
        self._model = None

    @property
    def ready(self) -> bool:
        if not config.RERANK:
            return False
        if self._model is not None:
            return True
        try:
            # 国内网络默认走 HF 镜像，避免 huggingface.co 连接超时挂起
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from sentence_transformers import CrossEncoder
            try:
                # 优先本地缓存加载（离线可用）；未缓存时走镜像下载
                self._model = CrossEncoder(
                    config.RERANK_MODEL, max_length=512, local_files_only=True)
            except Exception:
                self._model = CrossEncoder(config.RERANK_MODEL, max_length=512)
            return True
        except Exception as exc:
            logger.warning("重排模型加载失败，自动跳过重排: %s", exc)
            return False

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        """打分重排，返回 top_k；任何异常都回退为原顺序。"""
        if not docs:
            return docs
        if not self.ready:
            return docs[:top_k]
        try:
            pairs = [(query, d.page_content) for d in docs]
            scores = self._model.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        except Exception as exc:
            logger.warning("重排打分失败，保留融合顺序: %s", exc)
            return docs[:top_k]
        return [d for d, _ in ranked[:top_k]]
