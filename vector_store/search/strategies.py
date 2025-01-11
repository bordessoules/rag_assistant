from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class SearchStrategy(ABC):
    @abstractmethod
    def execute(self, vector_store, query: str, k: int) -> List[Document]:
        """Execute search strategy"""
        pass
        
    def preprocess_query(self, query: str) -> str:
        """Common query preprocessing"""
        return query.strip().lower()
        
    def postprocess_results(self, results: List[Document]) -> List[Document]:
        """Common result postprocessing"""
        return [doc for doc in results if doc.page_content.strip()]

class SimilaritySearch(SearchStrategy):
    def execute(self, vector_store, query: str, k: int) -> List[Document]:
        query = self.preprocess_query(query)
        results = vector_store.similarity_search(query, k=k)
        return self.postprocess_results(results)

class HybridSearch(SearchStrategy):
    def execute(self, vector_store, query: str, k: int) -> List[Document]:
        query = self.preprocess_query(query)
        
        # Get vector results
        vector_results = vector_store.similarity_search(query, k=k*2)
        
        # Keyword matching
        keyword_matches = set()
        query_terms = query.split()
        
        for doc in vector_results:
            doc_content = doc.page_content.lower()
            if any(term in doc_content for term in query_terms):
                keyword_matches.add(doc)
                
        # Combine results with priority to keyword matches
        hybrid_results = list(keyword_matches) + [
            doc for doc in vector_results if doc not in keyword_matches
        ]
        
        return self.postprocess_results(hybrid_results[:k])
