
import hashlib
import json
from pathlib import Path
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from config import settings

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self):
        self.processed_files_path = Path("processed_files.json")
        self.processed_files = self._load_processed_files()

    def reset_processed_files(self):
        """Reset the processed files tracking"""
        if self.processed_files_path.exists():
            self.processed_files_path.unlink()
        self.processed_files = {}
        self._save_processed_files()
        
    def _load_processed_files(self):
        if self.processed_files_path.exists():
            with open(self.processed_files_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_processed_files(self):
        with open(self.processed_files_path, 'w') as f:
            json.dump(self.processed_files, f)

    def _get_file_hash(self, file_path):
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def load_file(self, file_path: str) -> List:
        file_hash = self._get_file_hash(file_path)
        
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

    def _process_file(self, file_path: str) -> List:
        # Try UTF-8 first, fallback to other encodings if needed
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                loader = TextLoader(file_path, encoding=encoding)
                documents = loader.load()
                
                # Add source file path to metadata
                for doc in documents:
                    doc.metadata.update({
                        "source": file_path,
                        "file_type": Path(file_path).suffix,
                        "file_name": Path(file_path).name
                    })
                
                text_splitter = self._get_splitter(Path(file_path).suffix)
                
                split_docs = text_splitter.split_documents(documents)
                return split_docs
                
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error loading {file_path} with {encoding}: {str(e)}")
                continue
                
        logger.error(f"Failed to load {file_path} with any encoding")
        return []

    def _get_splitter(self, file_type: str):
        if file_type == '.py':
            return RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                separators=["\nclass ", "\ndef ", "\n\n", "\n", " "]
            )
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )