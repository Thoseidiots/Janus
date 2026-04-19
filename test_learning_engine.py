"""
Test suite for LearningEngine YouTube and Web Search integration
Tests Tasks 1.3 and 1.4 implementations
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from pathlib import Path
import sqlite3

from janus_autonomous_worker import (
    LearningEngine,
    LearningResource,
    SkillLevel,
    Skill
)


class TestYouTubeIntegration:
    """Test YouTube API integration (Task 1.3)"""
    
    @pytest.fixture
    def learning_engine(self):
        """Create a LearningEngine instance for testing"""
        engine = LearningEngine()
        yield engine
        # Cleanup
        if Path("janus_worker.db").exists():
            Path("janus_worker.db").unlink()
    
    @pytest.mark.asyncio
    async def test_youtube_search_with_api_key(self, learning_engine):
        """Test YouTube search when API key is configured"""
        learning_engine.youtube_api_key = "test_key"
        
        # Mock the requests.get call
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "items": [
                    {
                        "id": {"videoId": "dQw4w9WgXcQ"},
                        "snippet": {
                            "title": "Python Tutorial for Beginners",
                            "description": "Learn Python basics"
                        }
                    }
                ]
            }
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            # Mock video details
            with patch.object(learning_engine, '_get_youtube_video_details', 
                            new_callable=AsyncMock) as mock_details:
                mock_details.return_value = {"duration_minutes": 45}
                
                results = await learning_engine._search_youtube("python tutorial")
        
        assert len(results) > 0
        assert results[0].type == "youtube"
        assert "python" in results[0].title.lower()
        assert results[0].duration_minutes == 45
    
    @pytest.mark.asyncio
    async def test_youtube_search_without_api_key(self, learning_engine):
        """Test YouTube search when API key is not configured"""
        learning_engine.youtube_api_key = None
        
        results = await learning_engine._search_youtube("python tutorial")
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_youtube_rate_limiting(self, learning_engine):
        """Test YouTube API rate limiting"""
        learning_engine.youtube_api_key = "test_key"
        learning_engine.youtube_requests_made = 100
        learning_engine.youtube_rate_limit_max = 100
        
        # Should return False when rate limit is reached
        can_request = learning_engine._check_youtube_rate_limit()
        assert can_request is False
    
    @pytest.mark.asyncio
    async def test_youtube_video_details_parsing(self, learning_engine):
        """Test parsing of YouTube video duration"""
        learning_engine.youtube_api_key = "test_key"
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "items": [
                    {
                        "contentDetails": {
                            "duration": "PT45M30S"  # 45 minutes 30 seconds
                        }
                    }
                ]
            }
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            details = await learning_engine._get_youtube_video_details("dQw4w9WgXcQ")
        
        assert details["duration_minutes"] == 46  # Rounded up
    
    @pytest.mark.asyncio
    async def test_youtube_error_handling(self, learning_engine):
        """Test YouTube API error handling"""
        learning_engine.youtube_api_key = "test_key"
        
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            results = await learning_engine._search_youtube("python tutorial")
        
        assert len(results) == 0


class TestWebSearchIntegration:
    """Test Web Search API integration (Task 1.4)"""
    
    @pytest.fixture
    def learning_engine(self):
        """Create a LearningEngine instance for testing"""
        engine = LearningEngine()
        yield engine
        # Cleanup
        if Path("janus_worker.db").exists():
            Path("janus_worker.db").unlink()
    
    @pytest.mark.asyncio
    async def test_web_search_with_api_key(self, learning_engine):
        """Test web search when API key is configured"""
        learning_engine.web_search_api_key = "test_key"
        learning_engine.web_search_engine_id = "test_engine_id"
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "items": [
                    {
                        "title": "Python Tutorial - Learn Python Basics",
                        "link": "https://example.com/python-tutorial",
                        "snippet": "Learn Python programming from scratch. This comprehensive guide covers all the basics."
                    }
                ]
            }
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            results = await learning_engine._search_web("python tutorial")
        
        assert len(results) > 0
        assert results[0].type == "article"
        assert "python" in results[0].title.lower()
        assert results[0].url == "https://example.com/python-tutorial"
    
    @pytest.mark.asyncio
    async def test_web_search_without_api_key(self, learning_engine):
        """Test web search when API key is not configured"""
        learning_engine.web_search_api_key = None
        
        results = await learning_engine._search_web("python tutorial")
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_web_search_filters_paywalled_content(self, learning_engine):
        """Test that web search filters out paywalled content"""
        learning_engine.web_search_api_key = "test_key"
        learning_engine.web_search_engine_id = "test_engine_id"
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "items": [
                    {
                        "title": "Free Python Tutorial",
                        "link": "https://example.com/free-tutorial",
                        "snippet": "Free tutorial"
                    },
                    {
                        "title": "Premium Python Course",
                        "link": "https://example.com/premium-paywall",
                        "snippet": "Premium content"
                    }
                ]
            }
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            results = await learning_engine._search_web("python tutorial")
        
        # Should only include the free tutorial
        assert len(results) == 1
        assert "free" in results[0].title.lower()
    
    @pytest.mark.asyncio
    async def test_web_search_rate_limiting(self, learning_engine):
        """Test web search API rate limiting"""
        learning_engine.web_search_api_key = "test_key"
        learning_engine.web_search_requests_made = 100
        learning_engine.web_search_rate_limit_max = 100
        
        can_request = learning_engine._check_web_search_rate_limit()
        assert can_request is False
    
    @pytest.mark.asyncio
    async def test_web_search_error_handling(self, learning_engine):
        """Test web search API error handling"""
        learning_engine.web_search_api_key = "test_key"
        learning_engine.web_search_engine_id = "test_engine_id"
        
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            results = await learning_engine._search_web("python tutorial")
        
        assert len(results) == 0


class TestConceptExtraction:
    """Test concept extraction from resources"""
    
    @pytest.fixture
    def learning_engine(self):
        """Create a LearningEngine instance for testing"""
        engine = LearningEngine()
        yield engine
        # Cleanup
        if Path("janus_worker.db").exists():
            Path("janus_worker.db").unlink()
    
    @pytest.mark.asyncio
    async def test_extract_concepts_from_youtube(self, learning_engine):
        """Test extracting concepts from YouTube video"""
        resource = LearningResource(
            url="https://www.youtube.com/watch?v=test",
            title="Python Programming Basics Tutorial",
            type="youtube",
            topic="python",
            duration_minutes=45
        )
        
        concepts = await learning_engine._extract_concepts_from_youtube(resource)
        
        assert len(concepts) > 0
        assert isinstance(concepts, list)
    
    @pytest.mark.asyncio
    async def test_extract_concepts_from_web(self, learning_engine):
        """Test extracting concepts from web article"""
        resource = LearningResource(
            url="https://example.com/python-tutorial",
            title="Python Programming Guide",
            type="article",
            topic="python",
            duration_minutes=10
        )
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.text = "<html><body><p>Python programming tutorial with functions and classes</p></body></html>"
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            concepts = await learning_engine._extract_concepts_from_web(resource)
        
        assert len(concepts) > 0
        assert isinstance(concepts, list)
    
    @pytest.mark.asyncio
    async def test_learn_from_resource(self, learning_engine):
        """Test learning from a resource"""
        resource = LearningResource(
            url="https://www.youtube.com/watch?v=test",
            title="Python Programming Basics",
            type="youtube",
            topic="python",
            duration_minutes=45
        )
        
        concepts = await learning_engine.learn_from_resource(resource)
        
        assert resource.completed is True
        assert len(concepts) >= 0
        assert resource.learned_concepts == concepts


class TestDatabasePersistence:
    """Test database persistence of learning resources"""
    
    @pytest.fixture
    def learning_engine(self):
        """Create a LearningEngine instance for testing"""
        engine = LearningEngine()
        yield engine
        # Cleanup
        if Path("janus_worker.db").exists():
            Path("janus_worker.db").unlink()
    
    def test_store_learning_resource(self, learning_engine):
        """Test storing learning resource in database"""
        resource = LearningResource(
            url="https://example.com/tutorial",
            title="Python Tutorial",
            type="article",
            topic="python",
            duration_minutes=30,
            completed=True,
            learned_concepts=["variables", "functions", "loops"]
        )
        
        learning_engine._store_learning_resource(resource)
        
        # Verify it was stored
        conn = sqlite3.connect(learning_engine.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM learning WHERE url = ?", (resource.url,))
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[1] == "Python Tutorial"  # title
        assert row[2] == "python"  # topic
    
    def test_retrieve_learning_resource(self, learning_engine):
        """Test retrieving learning resource from database"""
        resource = LearningResource(
            url="https://example.com/tutorial",
            title="Python Tutorial",
            type="article",
            topic="python",
            duration_minutes=30,
            completed=True,
            learned_concepts=["variables", "functions"]
        )
        
        learning_engine._store_learning_resource(resource)
        
        # Retrieve from database
        conn = sqlite3.connect(learning_engine.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM learning WHERE url = ?", (resource.url,))
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[0] == resource.url
        assert row[1] == resource.title


class TestRetryLogic:
    """Test exponential backoff retry logic"""
    
    @pytest.fixture
    def learning_engine(self):
        """Create a LearningEngine instance for testing"""
        engine = LearningEngine()
        yield engine
        # Cleanup
        if Path("janus_worker.db").exists():
            Path("janus_worker.db").unlink()
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_success(self, learning_engine):
        """Test retry logic succeeds on first attempt"""
        call_count = 0
        
        def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await learning_engine._retry_with_backoff(test_func)
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_timeout(self, learning_engine):
        """Test retry logic handles timeouts"""
        import requests
        
        call_count = 0
        
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise requests.exceptions.Timeout("Timeout")
            return "success"
        
        result = await learning_engine._retry_with_backoff(test_func, max_retries=5)
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_max_retries(self, learning_engine):
        """Test retry logic respects max retries"""
        import requests
        
        call_count = 0
        
        def test_func():
            nonlocal call_count
            call_count += 1
            raise requests.exceptions.Timeout("Timeout")
        
        with pytest.raises(requests.exceptions.Timeout):
            await learning_engine._retry_with_backoff(test_func, max_retries=3)
        
        assert call_count == 3


class TestIntegration:
    """Integration tests for learning engine"""
    
    @pytest.fixture
    def learning_engine(self):
        """Create a LearningEngine instance for testing"""
        engine = LearningEngine()
        yield engine
        # Cleanup
        if Path("janus_worker.db").exists():
            Path("janus_worker.db").unlink()
    
    @pytest.mark.asyncio
    async def test_find_learning_resources_complete_flow(self, learning_engine):
        """Test complete flow of finding learning resources"""
        learning_engine.youtube_api_key = "test_key"
        learning_engine.web_search_api_key = "test_key"
        learning_engine.web_search_engine_id = "test_engine_id"
        
        with patch('requests.get') as mock_get:
            # Mock YouTube response
            youtube_response = Mock()
            youtube_response.json.return_value = {
                "items": [
                    {
                        "id": {"videoId": "test_video"},
                        "snippet": {
                            "title": "Python Tutorial",
                            "description": "Learn Python"
                        }
                    }
                ]
            }
            youtube_response.raise_for_status = Mock()
            
            # Mock web search response
            web_response = Mock()
            web_response.json.return_value = {
                "items": [
                    {
                        "title": "Python Guide",
                        "link": "https://example.com/guide",
                        "snippet": "Python programming guide"
                    }
                ]
            }
            web_response.raise_for_status = Mock()
            
            mock_get.side_effect = [youtube_response, youtube_response, web_response]
            
            with patch.object(learning_engine, '_get_youtube_video_details',
                            new_callable=AsyncMock) as mock_details:
                mock_details.return_value = {"duration_minutes": 45}
                
                resources = await learning_engine.find_learning_resources("python", "coding")
        
        # Should have found resources from both YouTube and web
        assert len(resources) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
