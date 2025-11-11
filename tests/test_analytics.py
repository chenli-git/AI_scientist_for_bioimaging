"""
Tests for analytics module.
"""

import pytest
import os
import json
from pathlib import Path
from core.analytics import UsageAnalytics


class TestAnalytics:
    """Test the analytics system."""
    
    def setup_method(self):
        """Create a temporary analytics file for testing."""
        self.test_file = "data/test_analytics.json"
        # Clean up if exists
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        self.analytics = UsageAnalytics(log_file=self.test_file)
    
    def teardown_method(self):
        """Clean up test file after each test."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_initialization(self):
        """Test that analytics initializes with empty data."""
        assert self.analytics.data["total_queries"] == 0
        assert self.analytics.data["session_count"] == 0
        assert self.analytics.data["image_uploads"] == 0
        assert self.analytics.data["pdf_uploads"] == 0
    
    def test_log_query(self):
        """Test logging a query."""
        self.analytics.log_query(
            query="Test query",
            agent_used="scientist",
            session_id="test_session",
            response_time=1.5
        )
        
        assert self.analytics.data["total_queries"] == 1
        assert self.analytics.data["agent_usage"]["scientist"] == 1
        assert len(self.analytics.data["query_history"]) == 1
        assert len(self.analytics.data["response_times"]) == 1
    
    def test_log_query_with_image(self):
        """Test logging a query with image upload."""
        self.analytics.log_query(
            query="Analyze image",
            agent_used="analyst",
            session_id="test",
            has_image=True
        )
        
        assert self.analytics.data["image_uploads"] == 1
        assert self.analytics.data["agent_usage"]["analyst"] == 1
    
    def test_log_query_with_pdf(self):
        """Test logging a query with PDF upload."""
        self.analytics.log_query(
            query="Review paper",
            agent_used="reviewer",
            session_id="test",
            has_pdf=True
        )
        
        assert self.analytics.data["pdf_uploads"] == 1
        assert self.analytics.data["agent_usage"]["reviewer"] == 1
    
    def test_log_feedback(self):
        """Test logging user feedback."""
        self.analytics.log_feedback(
            session_id="test",
            rating=5,
            comment="Great tool!"
        )
        
        assert len(self.analytics.data["user_feedback"]) == 1
        assert self.analytics.data["user_feedback"][0]["rating"] == 5
    
    def test_log_new_session(self):
        """Test session counter."""
        initial_count = self.analytics.data["session_count"]
        self.analytics.log_new_session()
        assert self.analytics.data["session_count"] == initial_count + 1
    
    def test_get_summary_stats(self):
        """Test summary statistics generation."""
        # Log some data
        self.analytics.log_query("Q1", "scientist", "s1", 1.0)
        self.analytics.log_query("Q2", "analyst", "s2", 2.0, has_image=True)
        self.analytics.log_feedback("s1", 4, "Good")
        
        stats = self.analytics.get_summary_stats()
        
        assert stats["total_queries"] == 2
        assert stats["image_uploads"] == 1
        assert stats["avg_response_time_sec"] == 1.5  # (1.0 + 2.0) / 2
        assert stats["avg_user_rating"] == 4.0
    
    def test_persistence(self):
        """Test that data persists to disk."""
        self.analytics.log_query("Test", "scientist", "test", 1.0)
        
        # Create new instance with same file
        new_analytics = UsageAnalytics(log_file=self.test_file)
        
        # Should load previous data
        assert new_analytics.data["total_queries"] == 1
    
    def test_export_for_paper(self):
        """Test exporting metrics for paper."""
        self.analytics.log_query("Q1", "scientist", "s1", 1.5)
        self.analytics.log_feedback("s1", 5, "Excellent")
        
        output_file = "data/test_paper_metrics.txt"
        report = self.analytics.export_for_paper(output_file)
        
        assert os.path.exists(output_file)
        assert "Total Queries" in report
        assert "Average Response Time" in report
        
        # Clean up
        os.remove(output_file)
    
    def test_query_history_limit(self):
        """Test that query history is limited to 1000 entries."""
        # Log more than 1000 queries
        for i in range(1100):
            self.analytics.log_query(f"Query {i}", "scientist", "test")
        
        # Should only keep last 1000
        assert len(self.analytics.data["query_history"]) == 1000
    
    def test_multiple_agents(self):
        """Test tracking multiple agents."""
        self.analytics.log_query("Q1", "scientist", "s1")
        self.analytics.log_query("Q2", "analyst", "s2")
        self.analytics.log_query("Q3", "reviewer", "s3")
        self.analytics.log_query("Q4", "scientist", "s4")
        
        assert self.analytics.data["agent_usage"]["scientist"] == 2
        assert self.analytics.data["agent_usage"]["analyst"] == 1
        assert self.analytics.data["agent_usage"]["reviewer"] == 1
