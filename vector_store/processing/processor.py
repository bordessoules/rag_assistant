# vector_store/processing/processor.py
from typing import List
from dataclasses import dataclass
from datetime import datetime
from langchain_core.documents import Document
from ..utils.guardrails import GuardrailsManager

@dataclass
class ChunkConfig:
    """Configuration for document chunking"""
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int
    max_chunk_size: int

class DocumentProcessor:
    """Processes documents through chunking and enrichment pipeline"""
    
    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig(
            chunk_size=1000,
            chunk_overlap=200,
            min_chunk_size=100,
            max_chunk_size=2000
        )
        self.guardrails = GuardrailsManager()

    def process(self, document: Document) -> List[Document]:
        """
        Process a document through the pipeline
        """
        chunks = self._chunk_document(document)
        enriched_chunks = self._enrich_metadata(chunks)
        validated_chunks = self._validate_chunks(enriched_chunks)
        return validated_chunks

    def _chunk_document(self, document: Document) -> List[Document]:
        """Split document into chunks"""
        text = document.page_content
        base_metadata = document.metadata.copy()
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            end_pos = min(current_pos + self.config.chunk_size, len(text))
            
            # If not at the end, try to find a good break point
            if end_pos < len(text):
                # Look for sentence end within overlap window
                overlap_start = end_pos - self.config.chunk_overlap
                break_point = self._find_break_point(text[overlap_start:end_pos])
                end_pos = overlap_start + break_point if break_point else end_pos
            
            chunk_text = text[current_pos:end_pos]
            if len(chunk_text.strip()) >= self.config.min_chunk_size:
                metadata = base_metadata.copy()
                metadata.update({
                    'chunk_index': len(chunks),
                    'chunk_start': current_pos,
                    'chunk_end': end_pos,
                    'processed_at': datetime.now().isoformat()
                })
                
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata=metadata
                ))
            
            current_pos = end_pos
        
        return chunks

    def _find_break_point(self, text: str) -> int:
        """Find a good break point in text (e.g., sentence end)"""
        break_chars = ['. ', '? ', '! ', '\n\n']
        best_pos = -1
        
        for char in break_chars:
            pos = text.rfind(char)
            if pos > best_pos:
                best_pos = pos
        
        return best_pos + 2 if best_pos >= 0 else 0

    def _enrich_metadata(self, chunks: List[Document]) -> List[Document]:
        """Add additional metadata to chunks"""
        enriched = []
        
        for chunk in chunks:
            metadata = chunk.metadata.copy()
            content = chunk.page_content
            
            # Add content statistics
            metadata.update({
                'content_length': len(content),
                'word_count': len(content.split()),
                'sentence_count': len([s for s in content.split('.') if s.strip()]),
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap
            })
            
            enriched.append(Document(
                page_content=content,
                metadata=metadata
            ))
        
        return enriched

    def _validate_chunks(self, chunks: List[Document]) -> List[Document]:
        """Validate chunks using guardrails"""
        validated = []
        
        for chunk in chunks:
            if len(chunk.page_content.strip()) >= self.config.min_chunk_size:
                if self.guardrails.validate(chunk):
                    validated.append(chunk)
        
        return validated