import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings
import shutil
import os
import time

class ChromaStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize client and vector store
        self._initialize_store()
    
    def _initialize_store(self):
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(
                anonymized_telemetry=False
            )
        )
        
        # Initialize Chroma vector store
        self.vector_store = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=self.embeddings,
            client=self.client
        )
    
    def reset_store(self):
        """Completely reset the vector store by removing the persistence directory"""
        # First delete the collection and close connections
        if hasattr(self, 'client') and self.client is not None:
            try:
                self.client.delete_collection(settings.COLLECTION_NAME)
                # Explicitly delete references to force connection closure
                self.vector_store._collection = None
                self.vector_store = None
                self.client = None
            except:
                pass
            
        # Longer delay to ensure connections are fully closed
        time.sleep(2)
    
        # Force Python garbage collection
        import gc
        gc.collect()
    
        retry_count = 3
        while retry_count > 0:
            try:
                if os.path.exists(settings.CHROMA_PERSIST_DIR):
                    shutil.rmtree(settings.CHROMA_PERSIST_DIR)
                break
            except PermissionError:
                retry_count -= 1
                time.sleep(2)
                continue
    
        # Reinitialize the store
        self._initialize_store()    
    def add_documents(self, documents):
        # No need to call persist() as ChromaDB now handles persistence automatically
        self.vector_store.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = settings.SEARCH_K):
        return self.vector_store.similarity_search(query, k=k)

    def as_retriever(self, search_kwargs=None):
        search_kwargs = search_kwargs or {}        
        base_kwargs = {
            "k": settings.SEARCH_K,
            "fetch_k": settings.SEARCH_K * 2,
            "maximal_marginal_relevance": True
        }
        
        if settings.ENABLE_FILE_FILTER:
            base_kwargs["filter"] = self._parse_file_filters(settings.FILE_FILTER)
        
        search_kwargs.update(base_kwargs)
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def _parse_file_filters(self, filter_string: str) -> dict:
        patterns = [p.strip() for p in filter_string.split(',')]
        
        # Build OR conditions for file types
        return {
            "$or": [
                {"file_type": pattern} if not pattern.startswith('*') 
                else {"file_name": {"$contains": pattern[1:]}}
                for pattern in patterns
            ]
        }
