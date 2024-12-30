from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from langchain_core.documents import Document

class VectorStoreRepository(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 4) -> List[Document]:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass
