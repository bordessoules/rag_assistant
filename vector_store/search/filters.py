from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

class ScoreThresholdFilter:
    """Filter for document relevance scores"""
    
    def __init__(self, threshold: float, k: int):
        self.threshold = threshold
        self.k = k
        
    def create_filtered_retriever(self, base_retriever: BaseRetriever) -> BaseRetriever:
        """Wraps a retriever with score threshold filtering"""
        
        class ScoreFilteredRetriever(BaseRetriever):
            def __init__(self, filter_instance, base):
                self.filter = filter_instance
                self.base_retriever = base
                
            def _get_relevant_documents(
                self, 
                query: str, 
                *, 
                run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
                docs_and_scores = self.base_retriever.get_relevant_documents(
                    query, 
                    include_scores=True
                )
                
                filtered = [
                    doc for doc, score in docs_and_scores 
                    if score >= self.filter.threshold
                ]
                return filtered[:self.filter.k]
                
        return ScoreFilteredRetriever(self, base_retriever)
