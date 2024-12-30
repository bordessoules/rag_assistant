from typing import List
from pathlib import Path
from .base_loader import DocumentLoader
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class FileLoader(DocumentLoader):
    def __init__(self, **kwargs):
        super().__init__()
        
    def load_file(self, file_path: str) -> List[Document]:
        """Load and process a single file"""
        file_hash = self._get_hash(file_path)
        
        # Check if file was already processed and hasn't changed
        if file_path in self.processed_files:
            if self.processed_files[file_path] == file_hash:
                logger.info(f"Skipping already processed file: {file_path}")
                return []
        
        # Process the file
        documents = self._process_file(file_path)
        
        # Update processed files record
        self.processed_files[file_path] = file_hash
        self._save_processed_files()
        
        return documents
    def _get_optimal_chunk_size(self, file_type: str) -> int:
        chunk_sizes = {
            '.py': 500,   # Code files
            '.js': 500,
            '.md': 2000,  # Documentation
            '.txt': 1500, # Text files
            'default': 1000
        }
        return chunk_sizes.get(file_type, chunk_sizes['default'])

