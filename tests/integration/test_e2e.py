from llm.lm_studio import LMStudioService
from llm.task_manager import TaskManager
from vector_store.repository.chroma import ChromaRepository
import pytest
from unittest.mock import patch


def test_full_pipeline():
    """Test the complete RAG pipeline"""
    with patch('requests.post') as mock_post:
        # Mock LLM responses
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
        
        # Initialize components
        repository = ChromaRepository()
        llm_service = LMStudioService(
            vector_store=repository,
            temperature=0.35,
            max_tokens=4096
        )
        
        # Register test function
        llm_service.register_function(
            name="test_function",
            description="Test function",
            parameters={
                "param1": "string"
            }
        )
        
        # Create task manager
        task_manager = TaskManager(llm_service)
        
        # Add test document
        test_docs = [
            {"page_content": "test content", "metadata": {"source": "test"}}
        ]
        repository.add_documents(test_docs)
        
        # Process test query
        response = task_manager.process_query("test query")
        
        # Validate response
        assert isinstance(response, dict)
        assert "answer" in response
        assert isinstance(response["answer"], str)
        if "source_documents" in response:
            assert isinstance(response["source_documents"], list)

def test_error_handling():
    """Test error handling in the pipeline"""
    with patch('requests.post') as mock_post:
        # Mock error response
        mock_post.return_value.status_code = 500
        
        repository = ChromaRepository()
        llm_service = LMStudioService(
            vector_store=repository,
            temperature=0.35,
            max_tokens=4096
        )
        
        task_manager = TaskManager(llm_service)
        
        # Test error handling
        with pytest.raises(Exception):
            task_manager.process_query("test query")