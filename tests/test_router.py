"""
Tests for the router module.
Tests routing logic without calling real OpenAI API.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from core.router import Router


class TestRouter:
    """Test the Router class."""
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        self.router = Router(use_llm=False)  # Use rule-based routing for tests
    
    def test_router_initialization(self):
        """Test that router initializes correctly."""
        assert self.router is not None
        assert "scientist" in self.router.agent_map
        assert "analyst" in self.router.agent_map
        assert "reviewer" in self.router.agent_map
    
    def test_rule_based_route_scientist(self):
        """Test routing to scientist agent based on keywords."""
        label = self.router._rule_based_route("What is adaptive optics in microscopy?")
        assert label == "scientist"
    
    def test_rule_based_route_analyst_with_keywords(self):
        """Test routing to analyst agent with segmentation keywords."""
        label = self.router._rule_based_route("How do I do segmentation on these cells?")
        assert label == "analyst"
    
    def test_rule_based_route_analyst_with_image(self):
        """Test routing to analyst when image is provided."""
        label = self.router._rule_based_route(
            "Analyze this", 
            image_path="test.png"
        )
        assert label == "analyst"
    
    def test_rule_based_route_reviewer(self):
        """Test routing to reviewer agent with paper keywords."""
        label = self.router._rule_based_route("Review this paper on cell imaging")
        assert label == "reviewer"
    
    def test_rule_based_route_reviewer_with_pdf(self):
        """Test routing to reviewer with PDF keywords."""
        label = self.router._rule_based_route(
            "Critique the methodology",
            pdf_path="paper.pdf"
        )
        assert label == "reviewer"
    
    def test_default_fallback(self):
        """Test that unknown queries fall back to scientist."""
        label = self.router._rule_based_route("random query without keywords")
        assert label == "scientist"
    
    @patch('core.router.GLOBAL_MEMORY')
    @patch('core.router.ANALYTICS')
    def test_route_query_with_mocked_agent(self, mock_analytics, mock_memory):
        """Test full route_query flow with mocked agent."""
        # Mock the agent's run method
        with patch.object(self.router, '_get_agent_instance') as mock_get_agent:
            mock_agent = Mock()
            mock_agent.run.return_value = "Mocked response"
            mock_get_agent.return_value = mock_agent
            
            # Execute
            response, label = self.router.route_query(
                query="What is microscopy?",
                session_id="test_session"
            )
            
            # Verify
            assert response == "Mocked response"
            assert label == "scientist"
            mock_agent.run.assert_called_once()
            mock_memory.add_user_message.assert_called_once()
            mock_memory.add_ai_message.assert_called_once()
    
    def test_query_intent_priority_over_file(self):
        """Test that query keywords take priority over uploaded files."""
        # Even with image, "review" keyword should route to reviewer
        label = self.router._rule_based_route(
            "review this paper",
            image_path="image.png"
        )
        assert label == "reviewer"


class TestRouterEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_query(self):
        """Test handling of empty query."""
        router = Router(use_llm=False)
        label = router._rule_based_route("")
        assert label == "scientist"  # Default fallback
    
    def test_case_insensitive_routing(self):
        """Test that routing is case-insensitive."""
        router = Router(use_llm=False)
        label1 = router._rule_based_route("REVIEW THIS PAPER")
        label2 = router._rule_based_route("review this paper")
        assert label1 == label2 == "reviewer"
    
    def test_multiple_keywords(self):
        """Test query with multiple agent keywords."""
        router = Router(use_llm=False)
        # "review" comes first in priority, should win
        label = router._rule_based_route("review this segmentation paper")
        assert label == "reviewer"
