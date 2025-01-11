# vector_store/utils/guardrails.py
from typing import List
from langchain_core.documents import Document

class ContentValidator:
    def validate(self, document: Document) -> bool:
        """Validate document content"""
        content = document.page_content
        
        # Check for empty content
        if not content.strip():
            return False
        
        return True

class SourceVerifier:
    def verify(self, document: Document) -> bool:
        """Verify document source"""
        metadata = document.metadata
        
        # Check for required metadata
        if 'source' not in metadata:
            return False
            
        # Verify source exists
        source = metadata['source']
        if not source:
            return False
            
        return True

class SimilarityChecker:
    def check(self, query: str, document: Document) -> bool:
        """Check query-document similarity"""
        query_terms = set(query.lower().split())
        content_terms = set(document.page_content.lower().split())
        
        # Calculate overlap
        overlap = len(query_terms.intersection(content_terms))
        if overlap == 0:
            return False
            
        return True

class GuardrailsManager:
    def __init__(self):
        self.content_validator = ContentValidator()
        self.source_verifier = SourceVerifier()
        self.similarity_checker = SimilarityChecker()
        
    def validate(self, query: str, results: List[Document]) -> List[Document]:
        """Apply all guardrails to validate results"""
        validated_results = []
        
        for doc in results:
            if (self.content_validator.validate(doc) and 
                self.source_verifier.verify(doc) and
                self.similarity_checker.check(query, doc)):
                validated_results.append(doc)
                
        return validated_results