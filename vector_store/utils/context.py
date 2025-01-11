from typing import List, Dict
from collections import deque
from datetime import datetime
from langchain_core.documents import Document

class SlidingWindow:
    def __init__(self, max_size: int = 5):
        self.window = deque(maxlen=max_size)
        
    def add(self, item: Dict) -> None:
        self.window.append(item)
        
    def get_recent(self, n: int = None) -> List[Dict]:
        n = n or len(self.window)
        return list(self.window)[-n:]

class QueryAnalyzer:
    def analyze(self, query: str) -> Dict:
        return {
            'length': len(query),
            'terms': query.split(),
            'type': self._determine_query_type(query)
        }
        
    def _determine_query_type(self, query: str) -> str:
        # Add query type detection logic
        return 'informational'  # placeholder

class RelevanceTracker:
    def track(self, query: str, results: List[Document]) -> Dict:
        scores = {}
        for doc in results:
            score = self._calculate_relevance(query, doc)
            scores[doc.metadata.get('source')] = score
        return scores
        
    def _calculate_relevance(self, query: str, doc: Document) -> float:
        # Implement relevance calculation
        return doc.metadata.get('score', 0.5)

class EnhancedContextManager:
    def __init__(self, window_size: int = 5):
        self.context_window = SlidingWindow(window_size)
        self.query_analyzer = QueryAnalyzer()
        self.relevance_tracker = RelevanceTracker()
        self.shared_context: Dict = {}
        
    def update_context(self, query: str, results: List[Document]) -> None:
        query_analysis = self.query_analyzer.analyze(query)
        relevance_scores = self.relevance_tracker.track(query, results)
        
        context = {
            'query': query,
            'analysis': query_analysis,
            'relevance': relevance_scores,
            'timestamp': datetime.now()
        }
        self.context_window.add(context)
        self._update_shared_context()
        
    def _update_shared_context(self) -> None:
        recent_contexts = self.context_window.get_recent()
        self.shared_context = {
            'queries': [ctx['query'] for ctx in recent_contexts],
            'avg_relevance': self._calculate_avg_relevance(recent_contexts)
        }
        
    def _calculate_avg_relevance(self, contexts: List[Dict]) -> float:
        if not contexts:
            return 0.0
        scores = [
            score 
            for ctx in contexts 
            for score in ctx['relevance'].values()
        ]
        return sum(scores) / len(scores) if scores else 0.0
