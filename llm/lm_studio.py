from langchain_core.language_models.llms import LLM
import logging
from typing import Any, Dict
import requests
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import settings
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)
# Define the prompt template with the required input variables
prompt_template = PromptTemplate(
    template="Use the following context to answer the question.\nContext: {context}\nQuestion: {question}\nAnswer: ",
    input_variables=["context", "question"]
)
class CustomLMStudio(LLM):
    """Simple LLM interface for LM Studio"""
    _base_url: str = PrivateAttr()
    _params: dict = PrivateAttr()
    def __init__(self, **kwargs):
        super().__init__()
        self._base_url = settings.LLM_BASE_URL
        self._params = kwargs
    def _call(self, prompt: str, **kwargs) -> str:
        response = requests.post(
            f"{self._base_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "messages": [{"role": "user", "content": prompt}],
                **self._params
            },
            timeout=1000
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    @property
    def _llm_type(self) -> str:
        """Required LLM property implementation need checking"""
        return "custom_lm_studio"

class LMStudioService:
    """High-level service for LLM operations"""
    DEFAULT_TEMPLATE = """Use the following context to answer the question.
    Context: {context}
    Question: {question}
    Answer: """

    def __init__(self, vector_store, **params):
        self.llm = CustomLMStudio(**params)
        self.retriever = vector_store
        self.prompt_template = self.DEFAULT_TEMPLATE
        self.qa_chain = self._create_qa_chain()
        logger.info("LMStudioService initialized")

    def _create_qa_chain(self):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs= {"prompt": prompt_template}
        )

    def set_prompt_template(self, template: str):
        self.prompt_template = template
        self.qa_chain = self._create_qa_chain()
        logger.info("Prompt template updated")

    def get_response(self, query: str) -> Dict:
        response = self.qa_chain.invoke({"query": query})
        return {
            "answer": response["result"],
            "source_documents": response["source_documents"]
        }
    
    def _get_context(self, query: str) -> Dict:
        """Cache and return context to avoid multiple retrievals"""
        if not self._context:
            self._context = self.llm.retriever.invoke(query)
