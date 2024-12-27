import logging
from langchain_core.language_models.llms import LLM
#from langchain_community.llms import LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import requests
from config import settings
from typing import Optional, List

logger = logging.getLogger(__name__)

class CustomLMStudio(LLM):
    base_url: str = settings.LLM_BASE_URL
    temperature: float = 0.35
    max_tokens: int = 32768
    top_p: float = 0.95
    min_p: float = 0.05
    top_k: int = 40
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Content-Type": "application/json"}
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
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
        
        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"]
    @property
    def _llm_type(self) -> str:
        return "custom_lm_studio"
class LMStudioService:
    def __init__(self, 
        vector_store,
        temperature=0.35,
        n_predict=32768,
        repeat_penalty=1.1,
        top_p=0.95,
        min_p=0.05,
        top_k=40,
        frequency_penalty=0.0,
        presence_penalty=0.0):
        
        self.llm = CustomLMStudio(
            temperature=temperature,
            max_tokens=n_predict,
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
        self._create_qa_chain(self.qa_chain.retriever.vectorstore)
        
    def get_response(self, query: str):
        logger.info(f"Processing query: {query}")
        response = self.qa_chain.invoke({"query": query})
        logger.info("Response generated")
        return {
            "answer": response["result"],
            "source_documents": response["source_documents"]
        }
