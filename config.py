from pydantic_settings import BaseSettings
from typing import Dict, List, Optional  
from dataclasses import dataclass 

@dataclass
class LLMProfile:
    """Profile configuration for Language Models"""
    temperature: float = 0.1
    top_p: float = 0.95
    min_p: float = 0.05
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 32768
    context_size: int = 32768
    eval_batch_size: int = 32768
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

@dataclass
class GuardrailsConfig:
    """Configuration for guardrails system"""
    max_retries: int = 2
    reflection_enabled: bool = True
    min_score: float = 0.5
    required_validations: List[str] = None

    def __post_init__(self):
        if self.required_validations is None:
            self.required_validations = [
                'format_validation',
                'parameter_validation',
                'response_validation'
            ]

@dataclass
class TEPConfig:
    """Configuration for Task Execution Procedure"""
    default_steps: List[str] = None
    validation_required: bool = True
    reflection_on_error: bool = True

    def __post_init__(self):
        if self.default_steps is None:
            self.default_steps = [
                'understand_query',
                'extract_parameters',
                'validate_params',
                'execute_function',
                'validate_response'
            ]

@dataclass
class ParameterValidationRules:
    """Rules for parameter validation"""
    type: str
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    required: bool = False
    min_value: Optional[float] = None  # Added for numeric types
    max_value: Optional[float] = None  # Added for numeric types

class Settings(BaseSettings):
    """Main configuration settings class"""
    
    # Logging Settings
    LOG_DIR: str = "./logs"
    LOG_DEBUG_DIR: str = "./logs/debug"
    LOG_NAME_FORMAT: str = "{model_name}_ctx-{context_size}_eval-{eval_batch_size}_date-{timestamp}"
    LOG_DATE_FORMAT: str = "%Y%m%d_%H%M%S"
    LOG_FILE_EXT: str = ".log"
    LOG_DEBUG_SUFFIX: str = "_debug"
    LOG_LEVEL: str = "INFO"  # Added log level setting

    # Vector Store Settings
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "rag_assistant"
    
    # File filtering settings
    ENABLE_FILE_FILTER: bool = True
    FILE_FILTER: str = ".py,!.json,!venv/*,!.git/*,!__pycache__/*"

    # LLM Base Settings
    LLM_BASE_URL: str = "http://localhost:1234/v1"
    LLM_TIMEOUT: int = 600  # Added timeout setting in seconds
    
    # API Endpoints
    CHAT_COMPLETION_ENDPOINT: str = "/chat/completions"
    EMBEDDINGS_ENDPOINT: str = "/embeddings"
    COMPLETION_ENDPOINT: str = "/completions"
    
    # Search and Processing Settings
    SEARCH_K: int = 50
    CHUNK_SIZE: int = 10000
    CHUNK_OVERLAP: int = 200
    MIN_CHUNK_SIZE: int = 100  # Added minimum chunk size
    MAX_CHUNK_SIZE: int = 50000  # Added maximum chunk size

    # Web Loader Settings
    WEB_MAX_DEPTH: int = 2
    WEB_CHUNK_SIZE: int = 10000
    WEB_TIMEOUT: int = 30  # Added web request timeout
    WEB_MAX_RETRIES: int = 3  # Added max retries for web requests
    
    # Guardrails Settings
    ENABLE_GUARDRAILS: bool = True
    GUARDRAILS_CONFIG: GuardrailsConfig = GuardrailsConfig()

    # Task Execution Procedure (TEP) Settings
    TEP_CONFIG: TEPConfig = TEPConfig()

    # Parameter Validation Rules
    PARAMETER_VALIDATION_RULES: Dict[str, ParameterValidationRules] = {
        "merchant_id": ParameterValidationRules(
            type="string",
            min_length=6,
            max_length=8,
            pattern="^[A-Za-z0-9]+$",
            required=True
        ),
        "restaurant_name": ParameterValidationRules(
            type="string",
            min_length=1,
            max_length=100,
            required=True
        ),
        "price": ParameterValidationRules(
            type="float",
            min_length=1,
            max_length=10,
            pattern="^\\d+\\.?\\d*$",
            required=True,
            min_value=0.0,
            max_value=1000000.0
        ),
        "menu_item": ParameterValidationRules(
            type="string",
            min_length=1,
            max_length=200,
            required=True
        )
    }

    # Model Profiles
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
            max_tokens=131072  
        ),
        "granite-3.1-8b": LLMProfile(
            temperature=0.35,  
            top_p=0.95,       
            min_p=0.05,       
            max_tokens=98304   
        ),
        "default": LLMProfile()
    }
    
    def get_llm_profile(self, model_name: str) -> LLMProfile:
        """Get LLM profile by model name, fallback to default if not found"""
        return self.LLM_PROFILES.get(model_name, self.LLM_PROFILES["default"])
    
    def get_parameter_rules(self, param_name: str) -> Optional[ParameterValidationRules]:
        """Get parameter validation rules by parameter name"""
        return self.PARAMETER_VALIDATION_RULES.get(param_name)

    def get_guardrails_config(self) -> GuardrailsConfig:
        """Get guardrails configuration if enabled"""
        return self.GUARDRAILS_CONFIG if self.ENABLE_GUARDRAILS else GuardrailsConfig()

    def get_tep_config(self) -> TEPConfig:
        """Get Task Execution Procedure configuration"""
        return self.TEP_CONFIG

    def get_api_url(self, endpoint: str) -> str:
        """Get full API URL for a given endpoint"""
        return f"{self.LLM_BASE_URL.rstrip('/')}{endpoint}"

    class Config:
        env_prefix = "RAG_"
        case_sensitive = False  # Added to make env vars case-insensitive
        validate_assignment = True  # Added to validate values on assignment

# Create settings instance
settings = Settings()