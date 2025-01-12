from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document

class VectorStoreRepository(ABC):
    @abstractmethod
    def search(self, query: str, k: int) -> List[Document]:
        pass
        
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        pass
    @abstractmethod
    def get_retriever(self, search_k: Optional[int] = None):
        """Get retriever with search parameters"""
        pass

    @staticmethod
    @abstractmethod
    def reset() -> None:
        pass

class VectorStoreError(Exception):
    """Base exception for vector store operations"""
    pass

class ValidationError(VectorStoreError):
    """Raised when document validation fails"""
    pass