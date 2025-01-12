from llm.lm_studio import LMStudioService
import logging

logger = logging.getLogger(__name__)

class TaskManager:
    def __init__(self, llm_service: LMStudioService):
        self.llm = llm_service
       
    
    def process_query(self, query: str) -> dict:
        """Process query through task pipeline"""
        return self.llm.get_response(query)