from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from langchain_core.documents import Document

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None

class ValidationService:
    """
    Centralized validation service for RAG assistant components.
    Handles document, search, and function parameter validation.
    """
    
    # Document validation constants
    MIN_CONTENT_LENGTH = 10
    REQUIRED_METADATA_FIELDS = ['source']
    
    # Search validation constants
    MAX_SEARCH_TOKENS = 512
    VALID_SEARCH_STRATEGIES = ['hybrid', 'semantic', 'keyword']
    
    @classmethod
    def validate_document(cls, doc: Document) -> ValidationResult:
        """
        Validates document structure and content quality
        
        Checks:
        - Content length
        - Required metadata fields
        - Content quality indicators
        """
        errors = []
        warnings = []
        
        # Content validation
        if not doc.page_content:
            errors.append("Document content cannot be empty")
        elif len(doc.page_content) < cls.MIN_CONTENT_LENGTH:
            warnings.append(f"Document content length below recommended minimum ({cls.MIN_CONTENT_LENGTH} chars)")
            
        # Metadata validation    
        if not doc.metadata:
            errors.append("Document must have metadata")
        else:
            for field in cls.REQUIRED_METADATA_FIELDS:
                if field not in doc.metadata:
                    errors.append(f"Missing required metadata field: {field}")
                    
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors if errors else None,
            warnings=warnings if warnings else None
        )

    @classmethod
    def validate_search_params(cls, query: str, params: Dict[str, Any]) -> ValidationResult:
        """
        Validates search query and parameters
        
        Checks:
        - Query length and content
        - Search strategy validity
        - Parameter boundaries
        """
        errors = []
        warnings = []
        
        # Query validation
        if not query.strip():
            errors.append("Search query cannot be empty")
        
        # Strategy validation
        strategy = params.get('strategy', 'hybrid')
        if strategy not in cls.VALID_SEARCH_STRATEGIES:
            errors.append(f"Invalid search strategy. Must be one of: {cls.VALID_SEARCH_STRATEGIES}")
            
        # Parameter validation
        k = params.get('k', 4)
        if not isinstance(k, int) or k < 1:
            errors.append("k must be a positive integer")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors if errors else None,
            warnings=warnings if warnings else None
        )

    @classmethod
    def validate_function_params(
        cls,
        params: Dict[str, Any],
        schema: Dict[str, Dict[str, Any]]
    ) -> ValidationResult:
        """
        Validates function parameters against their schema
        
        Checks:
        - Required parameters presence
        - Parameter types
        - Value constraints
        """
        errors = []
        warnings = []
        
        for param_name, param_schema in schema.items():
            # Check required parameters
            if param_schema.get('required', False) and param_name not in params:
                errors.append(f"Missing required parameter: {param_name}")
                continue
                
            if param_name in params:
                param_value = params[param_name]
                param_type = param_schema.get('type')
                
                # Type validation
                if param_type == 'string' and not isinstance(param_value, str):
                    errors.append(f"Parameter {param_name} must be a string")
                elif param_type == 'integer' and not isinstance(param_value, int):
                    errors.append(f"Parameter {param_name} must be an integer")
                elif param_type == 'float' and not isinstance(param_value, (int, float)):
                    errors.append(f"Parameter {param_name} must be a number")
                    
                # Value constraints
                if 'min_value' in param_schema and param_value < param_schema['min_value']:
                    errors.append(f"Parameter {param_name} below minimum value")
                if 'max_value' in param_schema and param_value > param_schema['max_value']:
                    errors.append(f"Parameter {param_name} above maximum value")
                    
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors if errors else None,
            warnings=warnings if warnings else None
        )

# Usage example:
"""
# Document validation
doc = Document(page_content="Test content", metadata={"source": "test.txt"})
result = ValidationService.validate_document(doc)
if not result.is_valid:
    print(f"Validation failed: {result.errors}")
    
# Search validation
search_params = {"strategy": "hybrid", "k": 4}
result = ValidationService.validate_search_params("test query", search_params)

# Function parameter validation
function_schema = {
    "max_results": {
        "type": "integer",
        "required": True,
        "min_value": 1,
        "max_value": 10
    }
}
params = {"max_results": 5}
result = ValidationService.validate_function_params(params, function_schema)
"""
