import asyncio
from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings
from loaders.web_loader import WebsiteLoader
import logging
import json
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class SearchParams(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=settings.SEARCH_K, ge=1, le=100)
    score_threshold: Optional[float] = Field(default=0.5, ge=0, le=1.0)
    
    @field_validator('query')
    @classmethod
    def clean_query(cls, v):
        # Strip whitespace and normalize
        return ' '.join(v.strip().split())

class ChromaStore:
    def __init__(self):
        self.processed_docs_path = Path("processed_docs.json")
        self.processed_docs = self._load_processed_docs()
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
    def _load_processed_docs(self) -> dict:
        if self.processed_docs_path.exists():
            with open(self.processed_docs_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_processed_docs(self):
        with open(self.processed_docs_path, 'w') as f:
            json.dump(self.processed_docs, f)

    def reset_store(self):
        """Reset the vector store and processed documents tracking"""
        logger.info("Resetting vector store...")
        
        # Reset ChromaDB collection
        if hasattr(self, 'client') and self.client is not None:
            try:
                self.client.delete_collection(settings.COLLECTION_NAME)
                self.vector_store = None
                self.client = None
            except Exception as e:
                logger.error(f"Error deleting collection: {e}")
        
        # Reset processed docs
        if self.processed_docs_path.exists():
            self.processed_docs_path.unlink()
        self.processed_docs = {}
        
        # Reinitialize store
        self._initialize_store()
        logger.info("Vector store reset complete")
    
       
    def add_website_content(self, urls: List[str], custom_metadata: Optional[dict] = None):
        logger.info(f"Processing {len(urls)} websites")
        loader = WebsiteLoader(max_depth=2)
        documents = []
        
        for url in urls:
            try:
                if url in self.processed_docs.values():
                    logger.info(f"Skipping already processed URL: {url}")
                    continue
                new_docs = loader.process_website(url, self.processed_docs, custom_metadata)
                if new_docs:
                    documents.extend(new_docs)
                    logger.info(f"Found {len(new_docs)} new documents from {url}")
                else:
                    logger.info(f"No new documents found from {url}")
            except Exception as e:
                logger.error(f"Failed to load content from {url}: {str(e)}")
                continue
        
        if documents:
            self.vector_store.add_documents(documents)
            self._save_processed_docs()
            logger.info(f"Added {len(documents)} new documents to vector store")
        else:
            logger.info("No new documents to add - all content already exists in store")
        
    def similarity_search(self, params: SearchParams):
        results = self.vector_store.similarity_search(
            query=params.query,
            k=params.k,
            score_threshold=params.score_threshold
        )
        return self._process_results(results)
    async def similarity_search_async(self, params: SearchParams) -> List[Document]:
        # 1. Perform async vector search
        results = await self.vector_store.asimilarity_search(
            query=params.query,
            k=params.k
        )
        
        # 2. Create concurrent tasks for processing each document
        processing_tasks = []
        for doc in results:
            task = asyncio.create_task(self._process_document_async(doc))
            processing_tasks.append(task)
        
        # 3. Wait for all processing to complete concurrently
        processed_docs = await asyncio.gather(*processing_tasks)
        return processed_docs

    async def _process_document_async(self, document: Document) -> Document:
        """
        Asynchronously process individual documents
        This could include:
        - Fetching additional metadata
        - Updating document attributes
        - Performing async transformations
        """
        try:
            # Example async operations
            metadata = await self._fetch_metadata_async(document.metadata)
            document.metadata.update(metadata)
            
            # Could add more async processing steps
            if 'requires_translation' in metadata:
                document.page_content = await self._translate_async(document.page_content)
                
            return document
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    async def _fetch_metadata_async(self, metadata: dict) -> dict:
        """
        Example of an async metadata fetch operation
        """
        # Simulate async API call or database query
        await asyncio.sleep(0.1)  # Represents async work
        return {"additional_info": "async fetched data"}
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
        return {
            "$or": [
                {"file_type": pattern} if not pattern.startswith('*') 
                else {"file_name": {"$contains": pattern[1:]}}
                for pattern in patterns
            ]
        }
