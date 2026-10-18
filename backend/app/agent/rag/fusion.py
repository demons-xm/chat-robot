from langchain_core.documents import Document


def reciprocal_rank_fusion(rankings: list[list], k: int = 60, top_k: int = 10) -> list:
    """rankings: 多路文档标识排序列表（越靠前排名越高）。返回融合后的文档标识。"""
    scores: dict = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:top_k]


def hybrid_method(vector_docs: list[Document], bm25_docs: list[Document]):
    fused_ids = reciprocal_rank_fusion(
        [[d.metadata.get("doc_id") for d in vector_docs],
         [d.metadata.get("doc_id") for d in bm25_docs]],
        top_k=len(vector_docs)
    )
    doc_map = {d.metadata.get("doc_id"): d for d in vector_docs + bm25_docs}

    return [doc_map[did] for did in fused_ids if did in doc_map]
