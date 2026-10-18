import os
from pathlib import Path
import time
from langchain_core.documents import Document

from app.config import config
from app.agent.factory import chroma_db
from app.agent.vector_store.md5 import FileMD5Manager
from app.agent.vector_store.file_loader import FileLoader
from app.utils.logger import logger
from app.agent.vector_store.lexical_bm25 import BM25Index
from app.agent.vector_store.reranker import Reranker


class VectorStore:
    """
    混合索引：Chroma向量库 + BM25词检索
    【本类：纯知识库业务，不含线程锁、并发控制】
    约定：
    self._documents 只支持尾部extend追加；运行时不删除、不修改已有元素，删除需重启服务清理脏数据
    BM25.search返回self._documents的下标，检索按下标读取
    USE_BM25=False 时不构建BM25索引
    """
    def __init__(self):
        self.knowledge_base_path = config.KNOWLEDGE_PATH
        # 实例化FileLoader
        self.file_loader = FileLoader()
        try:
            self.vector_store = chroma_db()
        except Exception as exc:
            self.vector_store = None
            logger.warning(f"ChromaDB初始化失败，RAG将退化为BM25检索：{exc}")
        self.bm25 = BM25Index()
        self.reranker = Reranker()
        self.md5_manager = FileMD5Manager()

        self._documents: list[Document] = []
        self.chunk_count = 0

        # 初始化加载全部知识库
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """递归加载知识库目录中的 PDF、Markdown 和 TXT 文件，服务启动时调用"""
        logger.info("开始批量加载全部知识库文件...")
        supported_suffixes = {".pdf", ".md", ".markdown", ".txt"}
        for file_path in config.KNOWLEDGE_PATH.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in supported_suffixes:
                continue
            try:
                self._add_knowledge(file_path)
            except Exception as e:
                logger.info(f"文件 {file_path.name} 批量加载失败：{str(e)}")
                continue
        logger.info(f"知识库批量加载完成，总chunk数：{self.chunk_count}")

    def add_file(self, file_path: Path):
        """后台上传新增单文件入库（业务逻辑）"""
        if not file_path.is_file():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        self._add_knowledge(file_path)

    def _add_knowledge(self, file_path: Path):
        """内部单文件处理：解析、切分、追加文档、BM25、Chroma、MD5判断"""
        # 调用FileLoader, docs: list[Document]
        docs = self.file_loader.load(file_path)
        if not docs:
            logger.warning(f"{file_path.name} 解析后无文档chunk，跳过")
            return

        self._documents.extend(docs)

        if config.USE_BM25:
            self.bm25.add_documents([d.page_content for d in docs])
            logger.info(f"{file_path.name} 已加载到BM25索引")

        with open(file_path, "rb") as f:
            file_md5 = self.md5_manager.get_file_md5(f)

        if self.md5_manager.exists_md5(md5_str=file_md5):
            try:
                existing = self.vector_store.get(ids=[doc.metadata["doc_id"] for doc in docs])
                metadatas = existing.get("metadatas") or []
                if len(metadatas) != len(docs) or any(
                    not isinstance(metadata, dict)
                    or metadata.get("tenant_id") != doc.metadata["tenant_id"]
                    for metadata, doc in zip(metadatas, docs)
                ):
                    self.vector_store.add_documents(
                        docs,
                        ids=[doc.metadata["doc_id"] for doc in docs],
                    )
                    logger.info(f"{file_path.name} 已更新向量库租户元数据")
                else:
                    logger.info(f"{file_path.name} MD5已存在，跳过Chroma入库")
            except Exception as exc:
                logger.warning(f"{file_path.name} 元数据检查/更新失败：{exc}")
            return

        doc_ids = [d.metadata["doc_id"] for d in docs]
        self.vector_store.add_documents(docs, ids=doc_ids)
        self.chunk_count += len(docs)
        self.md5_manager.save_md5(file_md5)
        logger.info(f"{file_path.name} 入库Chroma完成，已记录MD5")

    def search_retriever(self):
        """纯向量检索"""
        return self.vector_store.as_retriever(search_kwargs={"k": config.TOP_K})

    def search_bm25(self):
        """BM25词检索"""
        def bm25_retriever(query: str):
            if not config.USE_BM25:
                return []
            idx_list = self.bm25.search(query=query, top_k=config.TOP_K)
            doc_len = len(self._documents)
            return [self._documents[idx] for idx in idx_list if 0 <= idx < doc_len]
        return bm25_retriever

    def search_reranker(self, query: str, docs: list[Document], detail: dict = None,
                        top_k: int = config.RERANK_TOP_K) -> list[Document]:
        if config.RERANK:
            t0 = time.perf_counter()
            docs = self.reranker.rerank(query, docs, top_k)
            detail["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return docs[:top_k]

    def delete_knowledge(self, file_name: str) -> bool:
        """删除知识库文件业务逻辑（不含锁）"""
        abs_file_path = os.path.join(self.knowledge_base_path, file_name)
        abs_file_path = os.path.normpath(abs_file_path)

        if not os.path.exists(abs_file_path):
            return False

        try:
            with open(abs_file_path, "rb") as f:
                target_md5 = FileMD5Manager.get_file_md5(f)
        except Exception as e:
            logger.warning(f"读取文件计算MD5失败：{e}")
            return False

        # 删除Chroma向量
        try:
            self.vector_store.delete(where={"path": abs_file_path})
        except Exception as e:
            logger.warning(f"Chroma删除向量失败：{e}")
            return False

        # 删除本地磁盘文件
        try:
            os.remove(abs_file_path)
        except Exception as e:
            logger.warning(f"删除本地文件失败：{e}")
            return False

        # 删除md5记录
        ok = self.md5_manager.remove_md5(target_md5)
        if ok:
            logger.info(f"MD5记录 {target_md5} 已移除")
        else:
            logger.warning(f"MD5记录 {target_md5} 在记录文件中未找到")

        logger.warning("文件已删除。运行时无法清理BM25索引与self._documents，存在脏数据；需重启服务重建知识库")
        return True


if __name__ == "__main__":
    vs = VectorStore()
    vs.delete_knowledge("2026年版《正德员工手册》-人资服务项目组(1)(3).pdf")  # 替换为实际文件名