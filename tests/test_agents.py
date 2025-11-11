"""
Tests for agent classes.
Uses mocking to avoid calling real OpenAI API.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agents.AI_scientist_agent import AIScientistAgent
from agents.Image_analyst_agent import ImageAnalystAgent
from agents.paper_reviewer_agent import PaperReviewerAgent


class TestAIScientistAgent:
    """Test the AI Scientist Agent."""
    
    @patch('agents.AI_scientist_agent.get_llm')
    @patch('agents.AI_scientist_agent.get_vectorstore')
    def test_agent_initialization(self, mock_vectorstore, mock_llm):
        """Test that agent initializes without errors."""
        mock_vectorstore.return_value.as_retriever.return_value = Mock()
        mock_llm.return_value = Mock()
        
        agent = AIScientistAgent()
        
        assert agent is not None
        assert agent.llm is not None
        assert agent.vectorstore is not None
    
    @patch('agents.AI_scientist_agent.get_llm')
    @patch('agents.AI_scientist_agent.get_vectorstore')
    @patch('agents.AI_scientist_agent.GLOBAL_MEMORY')
    def test_agent_run_method(self, mock_memory, mock_vectorstore, mock_llm):
        """Test that run method executes without calling real API."""
        # Setup mocks
        mock_vectorstore.return_value.as_retriever.return_value = Mock()
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value.content = "Test response"
        mock_llm.return_value = mock_llm_instance
        
        mock_session = Mock()
        mock_session.messages = []
        mock_memory.get_session.return_value = mock_session
        
        # Create agent
        agent = AIScientistAgent()
        
        # Mock the chain
        with patch.object(agent, 'chat_chain') as mock_chain:
            mock_chain.invoke.return_value = "Mocked answer"
            
            # Test
            response = agent.run("What is microscopy?", session_id="test")
            
            assert response == "Mocked answer"
            mock_chain.invoke.assert_called_once()


class TestImageAnalystAgent:
    """Test the Image Analyst Agent."""
    
    @patch('agents.Image_analyst_agent.get_vision_llm')
    @patch('agents.Image_analyst_agent.get_llm')
    @patch('agents.Image_analyst_agent.get_vectorstore')
    def test_agent_initialization(self, mock_vectorstore, mock_llm, mock_vision_llm):
        """Test that image analyst initializes correctly."""
        mock_vectorstore.return_value.as_retriever.return_value = Mock()
        mock_llm.return_value = Mock()
        mock_vision_llm.return_value = (Mock(), Mock())
        
        agent = ImageAnalystAgent()
        
        assert agent is not None
        assert agent.text_llm is not None
        assert agent.vision_llm is not None
    
    @patch('agents.Image_analyst_agent.get_vision_llm')
    @patch('agents.Image_analyst_agent.get_llm')
    @patch('agents.Image_analyst_agent.get_vectorstore')
    @patch('agents.Image_analyst_agent.GLOBAL_MEMORY')
    def test_text_only_query(self, mock_memory, mock_vectorstore, mock_llm, mock_vision_llm):
        """Test text-only query without image."""
        # Mock retriever to return documents
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Retrieved content"
        mock_retriever.invoke.return_value = [mock_doc]
        
        mock_vectorstore.return_value.as_retriever.return_value = mock_retriever
        mock_llm.return_value = Mock()
        mock_vision_llm.return_value = (Mock(), Mock())
        
        # Mock memory
        mock_session = Mock()
        mock_session.messages = []
        mock_memory.get_session.return_value = mock_session
        
        agent = ImageAnalystAgent()
        
        with patch.object(agent, 'chat_chain') as mock_chain:
            mock_response = Mock()
            mock_response.content = "Workflow suggestion"
            mock_chain.invoke.return_value = mock_response
            
            response = agent.run("Suggest a workflow", session_id="test")
            
            assert "Workflow suggestion" in response


class TestPaperReviewerAgent:
    """Test the Paper Reviewer Agent."""
    
    @patch('agents.paper_reviewer_agent.get_llm')
    @patch('agents.paper_reviewer_agent.get_vectorstore')
    def test_agent_initialization(self, mock_vectorstore, mock_llm):
        """Test that reviewer agent initializes correctly."""
        mock_vectorstore.return_value.as_retriever.return_value = Mock()
        mock_llm.return_value = Mock()
        
        agent = PaperReviewerAgent()
        
        assert agent is not None
        assert agent.retriever is not None
    
    @patch('agents.paper_reviewer_agent.get_llm')
    @patch('agents.paper_reviewer_agent.get_vectorstore')
    def test_text_query_without_pdf(self, mock_vectorstore, mock_llm):
        """Test literature review without PDF upload."""
        mock_vectorstore.return_value.as_retriever.return_value = Mock()
        mock_llm.return_value = Mock()
        
        agent = PaperReviewerAgent()
        
        with patch.object(agent, 'chat_chain') as mock_chain:
            mock_chain.invoke.return_value = "Review response"
            
            response = agent.run("Summarize papers on imaging", session_id="test")
            
            assert response == "Review response"
    
    @patch('agents.paper_reviewer_agent.extract_text_images_tables')
    @patch('agents.paper_reviewer_agent.get_llm')
    @patch('agents.paper_reviewer_agent.get_vectorstore')
    def test_pdf_extraction(self, mock_vectorstore, mock_llm, mock_extract):
        """Test PDF content extraction."""
        mock_vectorstore.return_value.as_retriever.return_value = Mock()
        mock_llm.return_value = Mock()
        mock_extract.return_value = {
            "text": "Paper content",
            "tables": [],
            "figures": []
        }
        
        agent = PaperReviewerAgent()
        
        with patch.object(agent, 'chat_chain') as mock_chain:
            mock_chain.invoke.return_value = "Paper critique"
            
            response = agent.run(
                "Review this paper", 
                pdf_path="test.pdf", 
                session_id="test"
            )
            
            mock_extract.assert_called_once_with("test.pdf")
            assert response == "Paper critique"


class TestBaseAgent:
    """Test the base agent interface."""
    
    def test_base_agent_is_abstract(self):
        """Test that base agent cannot be instantiated directly."""
        from agents.base_agent import BaseAgent
        
        with pytest.raises(TypeError):
            # Should raise error because run() is abstract
            agent = BaseAgent()
