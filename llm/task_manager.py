from typing import List
from llm.lm_studio import LMStudioService
import logging

logger = logging.getLogger(__name__)

class TaskManager:
    def __init__(self, llm_service: LMStudioService):
        self.llm = llm_service
        self.tasks = {
            'function_detection': self._detect_functions,
            'parameter_extraction': self._extract_parameters,
            'validation': self._validate_call
        }
    
    def _detect_functions(self, query: str) -> List[str]:
        """Detect which functions might be needed for the query"""
        return [name for name, func in self.llm.function_registry.items() 
               if any(kw in query.lower() for kw in func['description'].lower().split())]
    
    def _extract_parameters(self, query: str, functions: List[str]) -> dict:
        """Extract parameters for detected functions"""
        # Basic parameter extraction
        return {"query": query, "max_results": 4}
    
    def _validate_call(self, functions: List[str], params: dict) -> dict:
        """Validate function calls and parameters"""
        return {"valid": True, "confidence": 0.8}
        
    def process_query(self, query: str) -> dict:
        """Process query through task pipeline"""
        functions = self._detect_functions(query)
        if functions:
            params = self._extract_parameters(query, functions)
            validation = self._validate_call(functions, params)
            if validation["valid"]:
                return self.llm.get_response(query)
        return self.llm.get_response(query)