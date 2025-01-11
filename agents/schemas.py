from dataclasses import dataclass
from typing import Dict, List, Any
from enum import Enum

class FunctionTaskType(Enum):
    NEXT_BEST_FUNCTION = "next_best_function"
    NESTED_FUNCTION = "nested_function"
    FUNCTION_CHAIN = "function_chain"
    PARALLEL_FUNCTION = "parallel_function"

@dataclass
class FunctionSchema:
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]
    required: List[str]
    task_type: FunctionTaskType
    max_retries: int = 2

@dataclass 
class FunctionExecutionResult:
    success: bool
    result: Any = None
    error: str = None
    timestamp: str = None
