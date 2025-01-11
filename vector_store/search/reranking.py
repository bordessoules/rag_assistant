# reranking.py
from typing import List
from langchain_core.documents import Document

class CrossEncoder:
    """Simple cross-encoder reranking implementation"""
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """For now, just return the original documents"""
        return documents