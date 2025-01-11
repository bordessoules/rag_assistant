# vector_store/repository/tracking.py
from typing import Dict, List
from pathlib import Path
import json
import hashlib
from datetime import datetime
from langchain_core.documents import Document

class DocumentTracker:
    """Handles document tracking and deduplication"""
    def __init__(self, tracking_file: Path = Path("processed_docs.json")):
        self.tracking_file = tracking_file
        self.processed_docs = self._load_tracking()
    
    def is_processed(self, document: Document) -> bool:
        doc_hash = self._get_hash(document)
        return doc_hash in self.processed_docs
        
    def track_documents(self, documents: List[Document]) -> None:
        for doc in documents:
            doc_hash = self._get_hash(doc)
            self.processed_docs[doc_hash] = {
                'source': doc.metadata.get('source', ''),
                'last_updated': datetime.now().isoformat(),
                'chunk_index': doc.metadata.get('chunk_index', 0)
            }
        self._save_tracking()
    
    def _get_hash(self, document: Document) -> str:
        return hashlib.md5(document.page_content.encode()).hexdigest()
        
    def _load_tracking(self) -> Dict:
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_tracking(self) -> None:
        with open(self.tracking_file, 'w') as f:
            json.dump(self.processed_docs, f, indent=2)