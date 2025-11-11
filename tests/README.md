# Testing Guide

## Running Tests

### Run all tests:
```bash
pytest
```

### Run with verbose output:
```bash
pytest -v
```

### Run specific test file:
```bash
pytest tests/test_router.py
pytest tests/test_agents.py
pytest tests/test_analytics.py
```

### Run specific test class:
```bash
pytest tests/test_router.py::TestRouter
```

### Run specific test function:
```bash
pytest tests/test_router.py::TestRouter::test_router_initialization
```

### Run with coverage:
```bash
pytest --cov=. --cov-report=html
# View coverage report at htmlcov/index.html
```

### Skip slow tests:
```bash
pytest -m "not slow"
```

## Test Structure

```
tests/
├── __init__.py              # Makes tests a package
├── conftest.py              # Shared fixtures and configuration
├── test_router.py           # Tests for routing logic
├── test_agents.py           # Tests for all agent classes
└── test_analytics.py        # Tests for analytics tracking
```

## What's Being Tested

### test_router.py
- ✅ Router initialization
- ✅ Rule-based routing logic
- ✅ Keyword detection for each agent
- ✅ File upload routing (image/PDF)
- ✅ Query priority (intent over file type)
- ✅ Edge cases (empty queries, case sensitivity)

### test_agents.py
- ✅ Agent initialization (without API calls)
- ✅ AI Scientist Agent text queries
- ✅ Image Analyst Agent (text & vision modes)
- ✅ Paper Reviewer Agent (with/without PDF)
- ✅ Base agent abstract interface

### test_analytics.py
- ✅ Analytics data collection
- ✅ Query logging
- ✅ File upload tracking
- ✅ User feedback storage
- ✅ Summary statistics generation
- ✅ Data persistence
- ✅ Export for paper metrics

## Key Testing Concepts

### 1. Mocking
We use `unittest.mock` to avoid calling real OpenAI API:
```python
@patch('agents.AI_scientist_agent.get_llm')
def test_agent(mock_llm):
    mock_llm.return_value = Mock()
    # Test without real API
```

### 2. Fixtures
Reusable test data defined in `conftest.py`:
```python
def test_something(mock_openai_response):
    # Use the fixture
    assert mock_openai_response.content == "Mocked AI response"
```

### 3. Test Organization
- **Class-based**: Group related tests in classes
- **setup/teardown**: Initialize/cleanup for each test
- **Descriptive names**: Test names explain what they test

## Adding New Tests

1. Create test file: `tests/test_<module>.py`
2. Import what you're testing
3. Use mocks for external dependencies
4. Write descriptive test names
5. Use assertions to verify behavior

Example:
```python
def test_new_feature():
    """Test that new feature works correctly."""
    # Arrange
    input_data = "test"
    
    # Act  
    result = my_function(input_data)
    
    # Assert
    assert result == expected_output
```

## Continuous Integration

Tests run automatically on GitHub Actions when you push code.
See `.github/workflows/tests.yml` for CI configuration.
