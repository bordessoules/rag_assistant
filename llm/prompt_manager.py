from typing import Dict
import json
import logging

logger = logging.getLogger(__name__)

class PromptManager:
    def __init__(self, prompt_file: str = 'config/prompts.json'):
        self.prompts = self._load_prompts(prompt_file)
    
    def _load_prompts(self, file_path: str) -> Dict:
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def get_prompt(self, prompt_name: str) -> str:
        return self.prompts['single_prompts'][prompt_name]['prompt']
    
    def get_chain(self, chain_name: str) -> list:
        return self.prompts['prompt_chains'][chain_name]['steps']
