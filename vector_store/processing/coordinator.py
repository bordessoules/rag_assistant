# vector_store/coordinator.py
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from vector_store.repository.chroma import ChromaRepository
from vector_store.search.pipeline import RetrievalPipeline
from config import settings

logger = logging.getLogger(__name__)

class SearchCoordinator:
    """Coordinates search operations between different components"""
    
    def __init__(self, repository: ChromaRepository):
        self.repository = repository
        self.pipeline = RetrievalPipeline()
    
    def coordinate_search(self, 
                         query: str, 
                         search_config: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Coordinate search process"""
        try:
            # Use default config if none provided
            config = search_config or {
                'k': settings.SEARCH_K,
                'strategy': 'hybrid'
            }
            
            # Get initial results through pipeline
            initial_results = self.pipeline.execute(
                query=query,
                vector_store=self.repository.vector_store,
                **config
            )
            
            return initial_results
            
        except Exception as e:
            logger.error(f"Search coordination failed: {str(e)}")
            raise

    def process_batch_queries(self, 
                            queries: List[str],
                            search_config: Optional[Dict[str, Any]] = None) -> Dict[str, List[Document]]:
        """Process multiple queries in batch"""
        results = {}
        for query in queries:
            try:
                results[query] = self.coordinate_search(query, search_config)
            except Exception as e:
                logger.error(f"Batch query failed for '{query}': {str(e)}")
                results[query] = []
        return results