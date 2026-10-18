# -*- coding: utf-8 -*-
"""
FileLoader：文件解析加载类
输入：pathlib.Path 文件路径
输出：list[Document]
支持：pdf / md / markdown / txt
延迟导入pypdf，缺失库只对应格式不可用
"""
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

from app.config import config


class FileLoader:
    def __init__(self):
        # 初始化文本切分器、markdown标题切分器，参数外部传入（从config）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=config.SEPARATORS,
            length_function=len,
        )
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")],
            strip_headers=False,
        )
        self.max_section_len = max(config.CHUNK_SIZE * 2, config.CHUNK_SIZE + 200)

    def load(self, file_path: Path) -> list[Document]:
        """
        对外主方法：传入Path对象，解析文件并切分，返回Document列表
        """
        raw_text = self._parse_file(file_path)
        suffix = file_path.suffix.lower()

        if suffix in (".md", ".markdown"):
            chunks = self._split_markdown(raw_text)
        else:
            # pdf/txt 走普通字符分割
            texts = self.text_splitter.split_text(raw_text)
            chunks = [Document(page_content=t) for t in texts]

        # 增加metadata，带上文件路径、chunk序号、doc_id
        result_docs = []
        relative_parts = file_path.resolve().relative_to(config.KNOWLEDGE_PATH.resolve()).parts
        tenant_id = relative_parts[0] if len(relative_parts) > 1 else config.DEFAULT_TENANT_ID
        for idx, chunk in enumerate(chunks):
            meta = {
                "path": str(file_path),
                "chunk": idx,
                "doc_id": f"{str(file_path)}#{idx}",
                "tenant_id": tenant_id,
            }
            # 合并原有chunk自带metadata（markdown切分会自带header信息）
            merged_meta = {**chunk.metadata, **meta}
            result_docs.append(Document(page_content=chunk.page_content, metadata=merged_meta))
        return result_docs

    def _parse_file(self, path: Path) -> str:
        """文件解析为纯文本"""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._parse_pdf(path)
        if suffix in (".md", ".txt", ".markdown"):
            return path.read_text(encoding="utf-8", errors="ignore")
        raise ValueError(f"暂不支持的文件类型: {suffix}（支持 pdf/md/txt）")

    @staticmethod
    def _parse_pdf(path: Path) -> str:
        """延迟导入pypdf解析pdf"""
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(p.strip() for p in pages if p.strip())

    def _split_markdown(self, text: str) -> list[Document]:
        """markdown按标题切分，过长section再递归文本分割"""
        sections = self.markdown_splitter.split_text(text)
        docs = []
        for section in sections:
            if len(section.page_content) <= self.max_section_len:
                docs.append(section)
            else:
                docs.extend(self.text_splitter.split_documents([section]))
        return docs


if __name__ == '__main__':
    fl = FileLoader()
    f_path = Path(r"D:\agent\AI-Robot-FastAPI-LangChain-RAG-CrewAI-Agent--main\data\knowledge_base.md")

    doc = fl.load(f_path)

    print(doc)
