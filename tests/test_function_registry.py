def test_function_registration():
    repository = ChromaRepository()
    coordinator = FunctionCoordinator(repository)
    
    test_function = {
        "name": "test_func",
        "description": "Test function",
        "parameters": {"param1": "string"}
    }
    
    coordinator.register_function(**test_function)
    assert "test_func" in coordinator.function_registry
