from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class VectorStoreRepository(ABC):
    @abstractmethod
    def search(self, query: str, k: int) -> List[Document]:
        pass
        
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass

class VectorStoreError(Exception):
    """Base exception for vector store operations"""
    pass

class ValidationError(VectorStoreError):
    """Raised when document validation fails"""
    pass