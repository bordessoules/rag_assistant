# expansion.py
from typing import List

class QueryExpander:
    """Simple query expansion implementation"""
    def expand(self, query: str) -> List[str]:
        """For now, just return the original query"""
        return [query]

