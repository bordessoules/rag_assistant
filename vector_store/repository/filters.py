# vector_store/repository/filters.py
from typing import Dict

class FileFilter:
    """Handles file filtering logic"""
    @staticmethod
    def parse_filters(filter_str: str) -> Dict:
        filters = {"$and": []}
        for pattern in filter_str.split(','):
            if pattern.startswith('!'):
                filters["$and"].append({"source": {"$not": pattern[1:]}})
            else:
                filters["$and"].append({"source": {"$regex": pattern}})
        return filters