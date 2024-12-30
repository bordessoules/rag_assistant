import hashlib
import numpy as np
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document 
from .base_repository import VectorStoreRepository
from config import settings
import json
import logging

logger = logging.getLogger(__name__)

class SearchParams(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=settings.SEARCH_K, ge=1, le=100)
    score_threshold: Optional[float] = Field(default=0.5, ge=0, le=1.0)
    metadata_filters: Optional[dict] = None

    @field_validator('query')
    def clean_query(cls, v):
        return ' '.join(v.strip().split())

class ChromaRepository(VectorStoreRepository):
    def __init__(self, embedding_function=None):
        self.processed_docs_path = Path("processed_docs.json")
        self.processed_docs = self._load_processed_docs()
        self.embeddings = embedding_function or HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            #model_kwargs={'device': 'cuda'},  # GPU acceleration
            encode_kwargs={
                'normalize_embeddings': True,  # Creates unit vectors
                #'batch_size': 6,
                'convert_to_numpy': True
            }
        )
        self._initialize_store()

    def _load_processed_docs(self) -> dict:
        """Load processed documents from JSON file"""
        if self.processed_docs_path.exists():
            with open(self.processed_docs_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_processed_docs(self):
        """Save processed documents to JSON file"""
        with open(self.processed_docs_path, 'w') as f:
            json.dump(self.processed_docs, f)

    def _get_hash(self, content: str) -> str:
        """Generate hash for content"""
        return hashlib.md5(content.encode()).hexdigest()
    def _initialize_store(self):
        self.vector_store = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name=settings.COLLECTION_NAME
        )

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents directly to vector store with minimal processing to preserve context.
        
        Args:
            documents: List of Document objects to store
        """
        try:
            for doc in documents:
                # Prepare document while preserving structure
                prepared_doc = self._prepare_document(doc)
                doc_hash = self._get_hash(prepared_doc.page_content)
                
                if doc_hash not in self.processed_docs:
                    # Generate and process embeddings
                    embeddings = self.embeddings.embed_documents([prepared_doc.page_content])
                    processed_embeddings = self._process_embeddings(embeddings)
                    
                    # Store document with processed embeddings
                    self.vector_store.add_documents(
                        documents=[prepared_doc],
                        embeddings=processed_embeddings
                    )
                    self.processed_docs[doc_hash] = prepared_doc.metadata.get('source', '')
            
            self._save_processed_docs()
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise
    
    def add_documents2(self, documents: List[Document]) -> None:
        processed_docs: Dict[str, List[Document]] = {}
        try:
            # Prepare documents first - ensures proper encoding and content preparation
            documents = [self._prepare_document(doc) for doc in documents]
            
            # Group documents by section for organized processing
            for doc in documents:
                section = doc.metadata.get('section', 'default')
                if section not in processed_docs:
                    processed_docs[section] = []
                processed_docs[section].append(doc)
            
            # Process and merge documents by section while preserving context
            final_documents = []
            for section, docs in processed_docs.items():
                merged_content = self._merge_section_documents(docs)
                final_documents.extend(merged_content)
            
            # Add to vector store with deduplication and embedding processing
            for doc in final_documents:
                doc_hash = self._get_hash(doc.page_content)
                if doc_hash not in self.processed_docs:
                    # Generate and process embeddings
                    embeddings = self.embeddings.embed_documents([doc.page_content])
                    processed_embeddings = self._process_embeddings(embeddings)
                    
                    # Add document with processed embeddings
                    self.vector_store.add_documents(
                        documents=[doc],
                        embeddings=processed_embeddings
                    )
                    
                    # Track processed document
                    self.processed_docs[doc_hash] = doc.metadata.get('source', '')
                    
            # Save processed documents tracking
            self._save_processed_docs()
        
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    def search(self, query: str, k: int = 4) -> List[Document]:
        params = SearchParams(query=query, k=k)
        return self.vector_store.similarity_search(
            query=params.query,
            k=params.k
            #score_threshold=params.score_threshold
        )
    @staticmethod
    def parse_file_filters(filter_string: str) -> dict:
        patterns = [p.strip() for p in filter_string.split(',')]
        return {
            "$or": [
                {"file_type": pattern} if not pattern.startswith('*') 
                else {"file_name": {"$contains": pattern[1:]}}
                for pattern in patterns
            ]
        }

    @staticmethod
    def create_metadata_filter(**kwargs) -> dict:
        filters = {}
        if 'date_range' in kwargs:
            filters['created_at'] = {
                "$gte": kwargs['date_range']['start'],
                "$lte": kwargs['date_range']['end']
            }
        if 'source_type' in kwargs:
            filters['source_type'] = (
                {"$in": kwargs['source_type']} 
                if isinstance(kwargs['source_type'], list) 
                else kwargs['source_type']
            )
        if 'tags' in kwargs:
            filters['tags'] = {"$contains": kwargs['tags']}
        if 'custom_metadata' in kwargs:
            filters.update(kwargs['custom_metadata'])
        return filters

    def get_retriever(self, **kwargs):
        search_kwargs = {'k': kwargs.get('search_k', settings.SEARCH_K)}
        if settings.ENABLE_FILE_FILTER:
            search_kwargs['filter'] = self.parse_file_filters(settings.FILE_FILTER)
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def _merge_similar_content(self, existing_content: str, new_content: str, similarity_threshold: float = 0.8) -> str:
        """
        Merges similar content while preserving unique information
        Returns the merged content as a single string
        """
        # Calculate similarity ratio
        similarity = SequenceMatcher(None, existing_content, new_content).ratio()
        
        if similarity > similarity_threshold:
            # Split into sentences for granular merging
            existing_sentences = self._split_into_sentences(existing_content)
            new_sentences = self._split_into_sentences(new_content)
            
            # Build merged content preserving unique information
            merged_sentences = []
            seen_sentences = set()
            
            for sentence in existing_sentences + new_sentences:
                normalized = self._normalize_sentence(sentence)
                if normalized not in seen_sentences:
                    merged_sentences.append(sentence)
                    seen_sentences.add(normalized)
            
            return ' '.join(merged_sentences)
        
        return f"{existing_content}\n\n{new_content}"

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using basic rules"""
        import re
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    def _normalize_sentence(self, sentence: str) -> str:
        """Normalize sentence for comparison"""
        return ' '.join(sentence.lower().split())
   

    def _merge_section_documents(self, documents: List[Document]) -> List[Document]:
        """Merge documents within the same section"""
        merged_content = {}
        
        for doc in documents:
            subsection = doc.metadata.get('subsection', 'main')
            if subsection in merged_content:
                merged_content[subsection] = self._merge_similar_content(
                    merged_content[subsection],
                    doc.page_content
                )
            else:
                merged_content[subsection] = doc.page_content
        
        return [
            Document(
                page_content=content,
                metadata={
                    'section': doc.metadata.get('section', 'default'),
                    'subsection': subsection,
                    'source': doc.metadata.get('source', ''),
                }
            )
            for subsection, content in merged_content.items()
        ]
    # Add proper cleanup
    def __del__(self):
        if hasattr(self, 'vector_store'):
            self.vector_store = None
        if hasattr(self, 'embeddings'):
            self.embeddings = None

    def _process_embeddings(self, embeddings):
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()  # Convert to Python native type
        return embeddings
    
    def _prepare_document(self, doc: Document):
        content = doc.page_content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        #Add file path to content for better context
        content = f"File: {doc.metadata.get('source', 'Unknown')}\n\n{content}"
        return Document(page_content=content, metadata=doc.metadata)

    def reset(self) -> None:
        self.vector_store.delete_collection()
        if self.processed_docs_path.exists():
            self.processed_docs_path.unlink()
        self.processed_docs = {}
        self._initialize_store()