# test_integration.py
import pytest
from llm.lm_studio import LMStudioService
from llm.task_manager import TaskManager
from vector_store.repository.chroma import ChromaRepository
from config import settings
from unittest.mock import  patch

@pytest.fixture
def mock_llm_response():
    return {
        "choices": [
            {
                "message": {
                    "content": "Test response"
                }
            }
        ]
    }

@pytest.fixture
def repository():
    """Create a test ChromaRepository instance"""
    from langchain_core.documents import Document
    
    repo = ChromaRepository()
    test_docs = [
        Document(
            page_content="test content 1",
            metadata={"source": "test1"}
        ),
        Document(
            page_content="test content 2", 
            metadata={"source": "test2"}
        )
    ]
    repo.add_documents(test_docs)
    return repo
@pytest.fixture
def llm_service(repository):
    """Create a test LLMStudioService instance"""
    with patch('requests.post') as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Test response"
                    }
                }
            ]
        }
        mock_post.return_value.status_code = 200
        
        service = LMStudioService(
            vector_store=repository,
            temperature=0.35,
            max_tokens=4096
        )
        return service

def test_llm_validation(llm_service):
    """Test LLM response validation"""
    valid_response = {
        "choices": [
            {
                "message": {
                    "content": "Test content"
                }
            }
        ]
    }
    
    invalid_responses = [
        {},  # Empty response
        {"choices": []},  # Empty choices
        {"choices": [{}]},  # Missing message
        {"choices": [{"message": {}}]},  # Missing content
    ]
    
    # Test valid response
    try:
        llm_service.llm._validate_response(valid_response)
    except Exception as e:
        pytest.fail(f"Valid response validation failed: {str(e)}")
    
    # Test invalid responses
    for invalid_resp in invalid_responses:
        with pytest.raises(Exception):
            llm_service.llm._validate_response(invalid_resp)

def test_function_registration(llm_service):
    """Test function registration in LLM service"""
    function_data = {
        "name": "test_function",
        "description": "Test function for unit testing",
        "parameters": {
            "param1": {
                "type": "string",
                "description": "First parameter"
            },
            "param2": {
                "type": "integer",
                "description": "Second parameter"
            }
        }
    }
    
    llm_service.register_function(**function_data)
    assert "test_function" in llm_service.function_registry
    registered_func = llm_service.function_registry["test_function"]
    assert registered_func["name"] == function_data["name"]
    assert registered_func["description"] == function_data["description"]
    assert registered_func["parameters"] == function_data["parameters"]

def test_hybrid_search(repository):
    """Test hybrid search functionality"""
    query = "test query"
    results = repository.search(
        query=query,
        k=4,
        strategy="hybrid"
    )
    assert len(results) <= 4
    for doc in results:
        assert hasattr(doc, 'page_content')
        assert hasattr(doc, 'metadata')

def test_task_manager(llm_service):
    """Test task manager functionality"""
    task_manager = TaskManager(llm_service)
    query = "Find documents about testing"
    response = task_manager.process_query(query)
    
    assert isinstance(response, dict)
    assert "answer" in response
    assert isinstance(response["answer"], str)
    if "source_documents" in response:
        assert isinstance(response["source_documents"], list)

def test_guardrails():
    """Test guardrails configuration and validation"""
    guardrails_config = settings.get_guardrails_config()
    assert guardrails_config.max_retries == 2
    assert guardrails_config.reflection_enabled == True
    assert len(guardrails_config.required_validations) > 0

def test_parameter_validation():
    """Test parameter validation rules"""
    # Test merchant_id validation
    merchant_rules = settings.get_parameter_rules("merchant_id")
    assert merchant_rules.type == "string"
    assert merchant_rules.min_length == 6
    assert merchant_rules.max_length == 8
    assert merchant_rules.required == True
    
    # Test price validation
    price_rules = settings.get_parameter_rules("price")
    assert price_rules.type == "float"
    assert price_rules.min_value is not None
    assert price_rules.max_value is not None