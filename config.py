from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Dict

class LLMProfile:
    def __init__(self, **kwargs):
        self.temperature = kwargs.get('temperature', 0.35)
        self.top_p = kwargs.get('top_p', 0.95)
        self.min_p = kwargs.get('min_p', 0.05)
        self.top_k = kwargs.get('top_k', 40)
        self.repeat_penalty = kwargs.get('repeat_penalty', 1.1)
        self.max_tokens = kwargs.get('max_tokens', 32768)
        self.context_size = kwargs.get('context_size', 32768)
        self.eval_batch_size = kwargs.get('eval_batch_size', 32768)
        self.frequency_penalty = kwargs.get('frequency_penalty', 0.0)
        self.presence_penalty = kwargs.get('presence_penalty', 0.0)


class Settings(BaseSettings):
    
    # Logging Settings
    LOG_DIR: str = "./logs"
    LOG_DEBUG_DIR: str = "./logs/debug"
    LOG_NAME_FORMAT: str = "{model_name}_ctx-{context_size}_eval-{eval_batch_size}_date-{timestamp}"
    LOG_DATE_FORMAT: str = "%Y%m%d_%H%M%S"
    LOG_FILE_EXT: str = ".log"
    LOG_DEBUG_SUFFIX: str = "_debug"

    # Vector Store Settings
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "rag_assistant"
    
     # File filtering settings
    ENABLE_FILE_FILTER: bool = True
    FILE_FILTER: str = ".py,!.json,!venv/*,!.git/*,!__pycache__/*"  # Empty string as default, can be comma-separated patterns

    # LLM Base Settings
    LLM_BASE_URL: str = "http://localhost:1234/v1"
    
    # Search and Processing Settings
    SEARCH_K: int = 15
    CHUNK_SIZE: int = 10000
    CHUNK_OVERLAP: int = 200

     # Web Loader Settings
    WEB_MAX_DEPTH: int = 2
    WEB_CHUNK_SIZE: int = 10000
    
    # Model Profiles - declared as a field
    LLM_PROFILES: Dict[str, LLMProfile] = {
        "qwen-2.5": LLMProfile(
            temperature=0.35,
            top_p=0.95,
            min_p=0.05,
            max_tokens=32768
        ),    
        "exaone-3.5": LLMProfile(
            temperature=0.45,
            top_p=0.88,
            min_p=0.04,
            max_tokens=32768
        ),
         "granite-3.1-2b": LLMProfile(
            temperature=0.35,  
            top_p=0.95,       
            min_p=0.05,       
            max_tokens=131072   # Maximizing context
        ),
        "granite-3.1-8b": LLMProfile(
            temperature=0.35,  
            top_p=0.95,       
            min_p=0.05,       
            max_tokens=98304   # Maximizing context for my 12gb vram gpu
        ),
        "default": LLMProfile(
            temperature=0.35,
            top_p=0.95,
            min_p=0.05,
            max_tokens=32768,       
        )
    }
    
    def get_llm_profile(self, model_name: str) -> LLMProfile:
        if model_name in self.LLM_PROFILES:
            return self.LLM_PROFILES[model_name]
        return self.LLM_PROFILES["default"]
    
    class Config:
        env_prefix = "RAG_"

# Create settings instance
settings = Settings()
