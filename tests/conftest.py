"""
Tests configuration and fixtures.
This file is automatically loaded by pytest.
"""

import pytest
import os
from unittest.mock import Mock


@pytest.fixture
def mock_openai_response():
    """Fixture for mocking OpenAI API responses."""
    mock_response = Mock()
    mock_response.content = "Mocked AI response"
    return mock_response


@pytest.fixture
def mock_retriever():
    """Fixture for mocking ChromaDB retriever."""
    mock = Mock()
    mock_doc = Mock()
    mock_doc.page_content = "Retrieved document content"
    mock_doc.metadata = {"source": "test.pdf"}
    mock.invoke.return_value = [mock_doc]
    return mock


@pytest.fixture
def test_session_id():
    """Fixture providing a consistent test session ID."""
    return "test_session_12345"


@pytest.fixture
def sample_query():
    """Fixture providing a sample query."""
    return "What is adaptive optics in microscopy?"


@pytest.fixture
def temp_test_file(tmp_path):
    """Fixture creating a temporary file path."""
    return tmp_path / "test_file.txt"


# Configure pytest to skip tests that require real API calls
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring real API calls"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Set environment variable for testing
os.environ['TESTING'] = '1'
