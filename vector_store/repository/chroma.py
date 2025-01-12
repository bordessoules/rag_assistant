from typing import Dict, List, Optional, Type
from langchain_chroma import Chroma
import chromadb
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import shutil
import os
import logging
from vector_store.repository.base import VectorStoreRepository
from vector_store.search.expansion import QueryExpander
from vector_store.search.reranking import CrossEncoder
from vector_store.search.strategies import HybridSearch, SearchStrategy, SimilaritySearch

logger = logging.getLogger(__name__)

class ChromaRepository(VectorStoreRepository):
    
    @staticmethod
    def reset() -> None:
        """Static method to clean up storage without instantiation"""
        persist_directory = "chroma_db"
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            
        # Clean tracking files
        for file in ['processed_files.json', 'processed_urls.json']:
            if os.path.exists(file):
                os.remove(file)
                
        logger.info("Storage reset complete")
    
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32
            }
        )
        logger.debug("Initializing ChromaRepository")
        self.persist_directory = "chroma_db"
        settings = chromadb.Settings(
            allow_reset=True,
            anonymized_telemetry=False,
            is_persistent=True
        )
        self.db = Chroma(
            persist_directory=self.persist_directory,
            client_settings=settings,
            embedding_function=embeddings
        )
        logger.info("ChromaRepository initialized")

    def get_retriever(self, search_k: Optional[int] = None):
        """Get retriever with configured search parameters"""
        return self.db.as_retriever(
            search_kwargs={"k": search_k if search_k else 4}
    )

    def add_documents(self, documents: List[Document]) -> None:
        logger.debug(f"Adding {len(documents)} documents to vector store")
        self.db.add_documents(documents)
        logger.info("Documents added successfully")

    def search(self, query: str, k: int = 4) -> List[Document]:
        logger.debug(f"Searching with query: {query}, k={k}")
        results = self.db.similarity_search(query, k=k)
        logger.info(f"Found {len(results)} results")
        return results
    
class RetrievalPipeline:
    """Pipeline for enhancing retrieval with configurable search strategies"""
    
    def __init__(self):
        # Initialize components
        self.query_expander = QueryExpander()
        self.cross_encoder = CrossEncoder()
        
        # Register available search strategies
        self.strategies: Dict[str, Type[SearchStrategy]] = {
            "similarity": SimilaritySearch,
            "hybrid": HybridSearch
        }
    
    def execute(
        self, 
        vector_store, 
        query: str, 
        k: int, 
        strategy: str = "similarity"
    ) -> List[Document]:
        """
        Execute search pipeline with specified strategy
        
        Args:
            vector_store: Vector store instance
            query: Search query
            k: Number of results to return
            strategy: Search strategy to use ("similarity" or "hybrid")
            
        Returns:
            List of retrieved documents
        """
        try:
            # Validate strategy
            if strategy not in self.strategies:
                available = list(self.strategies.keys())
                raise ValueError(
                    f"Unknown search strategy: {strategy}. "
                    f"Available strategies: {available}"
                )
            
            # Execute search with selected strategy
            search_strategy = self.strategies[strategy]()
            results = search_strategy.execute(vector_store, query, k)
            
            # Log search execution
            logger.debug(
                f"Search executed - Strategy: {strategy}, "
                f"Query: {query}, Results: {len(results)}"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
    
    def register_strategy(self, name: str, strategy_class: Type[SearchStrategy]) -> None:
        """
        Register a new search strategy
        
        Args:
            name: Strategy name
            strategy_class: Strategy class implementing SearchStrategy
        """
        if not issubclass(strategy_class, SearchStrategy):
            raise ValueError(
                f"Strategy must inherit from SearchStrategy, "
                f"got {strategy_class.__name__}"
            )
            
        self.strategies[name] = strategy_class
        logger.info(f"Registered new search strategy: {name}")
