import logging
from langchain_core.language_models.llms import LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import requests
from config import settings
from typing import Optional, List

logger = logging.getLogger(__name__)

from typing import Optional, List
from enum import Enum

class LLMError(Exception):
    """Base exception class for LLM errors"""
    pass

class ErrorType(Enum):
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    INVALID_RESPONSE = "invalid_response"
    CONTEXT_LENGTH = "context_length"

class CustomLMStudio(LLM):
    base_url: str = settings.LLM_BASE_URL
    temperature: float = 0.35
    max_tokens: int = 4096
    top_p: float = 0.95
    min_p: float = 0.05
    top_k: int = 40
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    

    def _validate_response(self, response: dict) -> None:
        """
        Validates the LLM API response format
        """
        if not isinstance(response, dict):
            raise LLMError(ErrorType.INVALID_RESPONSE, "Response must be a dictionary")
            
        if "choices" not in response:
            raise LLMError(ErrorType.INVALID_RESPONSE, "Response missing 'choices' field")
            
        if not response["choices"]:
            raise LLMError(ErrorType.INVALID_RESPONSE, "Response contains no choices")
            
        choice = response["choices"][0]
        if not isinstance(choice, dict) or "message" not in choice:
            raise LLMError(ErrorType.INVALID_RESPONSE, "Invalid choice format")
            
        if "content" not in choice["message"]:
            raise LLMError(ErrorType.INVALID_RESPONSE, "Missing message content")
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Content-Type": "application/json"}
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": stop 
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=2400 # Explicit timeout
            )
            response.raise_for_status()
            
            result = response.json()
            self._validate_response(result)
            
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            raise LLMError(ErrorType.API_ERROR, "Request timed out")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API error: {str(e)}")
            raise LLMError(ErrorType.API_ERROR, str(e))
            
        except (KeyError, IndexError) as e:
            logger.error(f"Invalid response format: {str(e)}")
            raise LLMError(ErrorType.INVALID_RESPONSE, str(e))
        
        except Exception as e:
            logger.error(f"LLM error: {str(e)}")
            raise LLMError(ErrorType.API_ERROR, str(e))

    @property
    def _llm_type(self) -> str:
        return "custom_lm_studio"
    
class LMStudioService:
    def __init__(self, vector_store, **params):
        self.llm = CustomLMStudio(**params)
        self.function_registry = {}
        self.default_analysis_prompt = """Execute a comprehensive technical deep-dive:

1. Architecture & System Design:
   * Map complete component hierarchy and inheritance chains
   * Identify all design patterns and SOLID principles applied
   * Analyze coupling metrics between components
   * Evaluate system boundaries and integration points

2. Implementation Analysis:
   * Review thread safety and concurrency handling
   * Examine error propagation and recovery mechanisms
   * Analyze memory management and resource utilization
   * Evaluate type system usage and interface contracts

3. Performance Engineering:
   * Profile critical execution paths
   * Identify algorithmic complexity bottlenecks
   * Analyze space-time trade-offs in current implementation
   * Examine caching strategies and data locality

4. Technical Debt Assessment:
   * Identify code duplication and abstraction leaks
   * Review dependency management and version constraints
   * Analyze test coverage gaps and edge cases
   * Evaluate documentation completeness and accuracy

5. Advanced Optimization Opportunities:
   * Propose parallel processing implementations
   * Design caching layer improvements
   * Suggest architectural refactoring for scalability
   * Detail concrete code examples for each improvement

Provide specific code examples and complexity analysis for each point."""

        self.prompt_template = """You are a helpful AI assistant. Use the following context to answer the question.
                
        Context: {context}
        Question: {question}
        Answer: """
        
        # Get retriever first
        self.retriever = vector_store
        # Initialize QA chain
        self.qa_chain = self._create_qa_chain(self.retriever)
        logger.info("QA chain created")
    
    def register_function(self, name: str, description: str, parameters: dict):
        # Validates with existing prompt template system
        self.function_coordinator.register_function(
            name=name,
            description=description,
            parameters=parameters
        )
        # Update QA chain prompt to include function context
        self._update_qa_chain_prompt()
        
    def _create_qa_chain(self, retriever):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=self.prompt_template,
                    input_variables=["context", "question"]
                )
            }
        )
        
    def set_prompt_template(self, template: str):
        logger.info(f"Setting new prompt template: {template}")
        self.prompt_template = template
        self._create_qa_chain(self.retriever)
        
    def get_response(self, query: str):
        logger.info(f"Processing query: {query}")
        response = self.qa_chain.invoke({"query": query})
        logger.info("Response generated")
        return {
            "answer": response["result"],
            "source_documents": response["source_documents"]
        }

    def _update_qa_chain_prompt(self):
        """Update QA chain prompt to include registered functions"""
        function_context = "\n".join([
            f"Function: {f['name']}\nDescription: {f['description']}\nParameters: {f['parameters']}"
            for f in self.function_registry.values()
        ])
        
        self.prompt_template = f"""You are a helpful AI assistant. Use the following context and available functions to answer the question.
        
        Available Functions:
        {function_context}
        
        Context: {{context}}
        Question: {{question}}
        Answer: """
        
        # Recreate QA chain with existing retriever
        self.qa_chain = self._create_qa_chain(self.retriever)
