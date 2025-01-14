from typing import Dict
import logging
from .prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class TaskManager:
    def __init__(self, llm_service):
        self.llm = llm_service
        self.prompt_manager = PromptManager()
        self._context = None  # Cache for context
        
    def _get_context(self, query: str) -> Dict:
        """Cache and return context to avoid multiple retrievals"""
        if not self._context:
            self._context = self.llm.retriever.invoke(query)
        return self._context
    
    def execute_single_query(self, prompt_name: str, query: str) -> dict:
        """Execute single prompt with cached context"""
        prompt = self.prompt_manager.get_prompt(prompt_name)
        self.llm.set_prompt_template(prompt)
        context = self._get_context(query)
        return self.llm.get_response(query)
    
    def execute_chain(self, chain_name: str, query: str) -> Dict[str, dict]:
        """Execute chain of prompts with context reuse"""
        steps = self.prompt_manager.get_chain(chain_name)
        results = {}
        
        # Get context once for entire chain
        self._context = self._get_context(query)
        
        for step in steps:
            results[step] = self.execute_single_query(step, query)
            # Update query with previous results for CoT
            query = f"Previous analysis:\n{results[step]['answer']}\n\nOriginal query: {query}"
            
        # Clear context cache after chain completion
        self._context = None
        return results

    def reset_context(self):
        """Explicitly reset context cache if needed"""
        self._context = None