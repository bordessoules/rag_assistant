import logging
from langchain_core.language_models.llms import LLM
#from langchain_community.llms import LLM
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
                timeout=600 # Explicit timeout
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
    def __init__(self, 
        vector_store,
        temperature=0.35,
        max_tokens=8192,
        repeat_penalty=1.1,
        top_p=0.95,
        min_p=0.05,
        top_k=40,
        frequency_penalty=0.0,
        presence_penalty=0.0):
        
        self.llm = CustomLMStudio(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        logger.info("LLM initialized")
        
        self.prompt_template = """You are a helpful AI assistant. Use the following context to answer the question.
                
        Context: {context}
        
        Question: {question}
       
        Answer: """
        
        self._create_qa_chain(vector_store)
        logger.info("QA chain created")
        
    def _create_qa_chain(self, vector_store):
        logger.info("Creating QA chain")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=self.prompt_template,
                    input_variables=["context", "question"]  # Explicitly define both variables
                ),
                "document_variable_name": "context"  # Add this line
            }
        )
        
    def set_prompt_template(self, template: str):
        logger.info(f"Setting new prompt template: {template}")
        self.prompt_template = template
        self._create_qa_chain(self.qa_chain.retriever)
        
    def get_response(self, query: str):
        logger.info(f"Processing query: {query}")
        response = self.qa_chain.invoke({"query": query})
        logger.info("Response generated")
        return {
            "answer": response["result"],
            "source_documents": response["source_documents"]
        }
