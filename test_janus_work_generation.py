"""
test_janus_work_generation.py
=============================
Tests for Janus work generation and quality validation systems.

Tests cover:
- YouTube API integration
- Web search integration
- Work generation (Avus AI and template-based)
- Quality validation
- Concept extraction
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sqlite3

from janus_autonomous_worker import (
    Job, JobStatus, Skill, SkillLevel, LearningResource,
    LearningEngine, WorkGenerator, QualityValidator,
    JanusAutonomousWorker
)


# ═══════════════════════════════════════════════════════════════════════════════
# YOUTUBE API INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestYouTubeIntegration:
    """Tests for YouTube API integration"""
    
    @pytest.fixture
    def learning_engine(self):
        """Create a learning engine instance"""
        return LearningEngine()
    
    @pytest.mark.asyncio
    async def test_youtube_search_without_api_key(self, learning_engine):
        """Test YouTube search gracefully handles missing API key"""
        learning_engine.youtube_api_key = None
        results = await learning_engine._search_youtube("python tutorial")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_youtube_search_rate_limit_check(self, learning_engine):
        """Test YouTube rate limit checking"""
        learning_engine.youtube_api_key = "test_key"
        learning_engine.youtube_requests_made = 100
        learning_engine.youtube_rate_limit_max = 100
        
        # Should return False when rate limit reached
        assert not learning_engine._check_youtube_rate_limit()
    
    @pytest.mark.asyncio
    async def test_youtube_search_filters_educational_content(self, learning_engine):
        """Test YouTube search filters for educational content"""
        learning_engine.youtube_api_key = "test_key"
        
        # Mock the API response
        mock_response = {
            "items": [
                {
                    "id": {"videoId": "test_id_1"},
                    "snippet": {
                        "title": "Python Tutorial for Beginners",
                        "description": "Learn Python programming basics"
                    }
                },
                {
                    "id": {"videoId": "test_id_2"},
                    "snippet": {
                        "title": "Music Video - Not Educational",
                        "description": "Just a music video"
                    }
                }
            ]
        }
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = Mock()
            
            with patch.object(learning_engine, '_get_youtube_video_details', return_value={"duration_minutes": 30}):
                results = await learning_engine._search_youtube("python")
        
        # Should only return educational content
        assert len(results) == 1
        assert "Python" in results[0].title
    
    @pytest.mark.asyncio
    async def test_youtube_video_details_extraction(self, learning_engine):
        """Test YouTube video details extraction"""
        learning_engine.youtube_api_key = "test_key"
        
        mock_response = {
            "items": [
                {
                    "contentDetails": {
                        "duration": "PT1H30M45S"  # 1 hour 30 minutes 45 seconds
                    }
                }
            ]
        }
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = Mock()
            
            details = await learning_engine._get_youtube_video_details("test_id")
        
        # Should correctly parse ISO 8601 duration
        assert details["duration_minutes"] == 91  # 90 + 1 for seconds


# ═══════════════════════════════════════════════════════════════════════════════
# WEB SEARCH INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestWebSearchIntegration:
    """Tests for web search integration"""
    
    @pytest.fixture
    def learning_engine(self):
        """Create a learning engine instance"""
        return LearningEngine()
    
    @pytest.mark.asyncio
    async def test_web_search_without_api_key(self, learning_engine):
        """Test web search gracefully handles missing API key"""
        learning_engine.web_search_api_key = None
        results = await learning_engine._search_web("python tutorial")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_web_search_filters_paywalled_content(self, learning_engine):
        """Test web search filters out paywalled content"""
        learning_engine.web_search_api_key = "test_key"
        learning_engine.web_search_engine_id = "test_engine"
        
        mock_response = {
            "items": [
                {
                    "link": "https://example.com/free-tutorial",
                    "title": "Free Python Tutorial",
                    "snippet": "Learn Python for free"
                },
                {
                    "link": "https://premium.example.com/paywall",
                    "title": "Premium Python Course",
                    "snippet": "Premium content behind paywall"
                }
            ]
        }
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = Mock()
            
            results = await learning_engine._search_web("python")
        
        # Should only return non-paywalled content
        assert len(results) == 1
        assert "Free" in results[0].title
    
    @pytest.mark.asyncio
    async def test_web_search_filters_low_quality_sources(self, learning_engine):
        """Test web search filters out low-quality sources"""
        learning_engine.web_search_api_key = "test_key"
        learning_engine.web_search_engine_id = "test_engine"
        
        mock_response = {
            "items": [
                {
                    "link": "https://example.com/tutorial",
                    "title": "Python Tutorial",
                    "snippet": "Learn Python"
                },
                {
                    "link": "https://pinterest.com/python-pins",
                    "title": "Python Pins",
                    "snippet": "Pinterest pins about Python"
                }
            ]
        }
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = Mock()
            
            results = await learning_engine._search_web("python")
        
        # Should filter out Pinterest
        assert len(results) == 1
        assert "pinterest" not in results[0].url.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# WORK GENERATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorkGeneration:
    """Tests for work generation system"""
    
    @pytest.fixture
    def work_generator(self):
        """Create a work generator instance"""
        return WorkGenerator()
    
    @pytest.fixture
    def sample_job(self):
        """Create a sample job"""
        return Job(
            id="job_001",
            title="Write a Python Tutorial",
            description="Write a comprehensive tutorial on Python basics",
            required_skills=["writing", "python"],
            budget=500.0,
            deadline=datetime.now() + timedelta(days=7),
            platform="upwork"
        )
    
    def test_job_type_detection_writing(self, work_generator):
        """Test job type detection for writing jobs"""
        job = Job(
            id="job_001",
            title="Write an Article",
            description="Write a blog post about AI",
            required_skills=["writing"],
            budget=100.0,
            deadline=datetime.now() + timedelta(days=7),
            platform="upwork"
        )
        
        job_type = work_generator._detect_job_type(job)
        assert job_type == "writing"
    
    def test_job_type_detection_coding(self, work_generator):
        """Test job type detection for coding jobs"""
        job = Job(
            id="job_001",
            title="Develop a Python Script",
            description="Write Python code to process data",
            required_skills=["python"],
            budget=200.0,
            deadline=datetime.now() + timedelta(days=7),
            platform="upwork"
        )
        
        job_type = work_generator._detect_job_type(job)
        assert job_type == "coding"
    
    def test_job_type_detection_research(self, work_generator):
        """Test job type detection for research jobs"""
        job = Job(
            id="job_001",
            title="Research Market Trends",
            description="Analyze and research current market trends",
            required_skills=["research"],
            budget=300.0,
            deadline=datetime.now() + timedelta(days=7),
            platform="upwork"
        )
        
        job_type = work_generator._detect_job_type(job)
        assert job_type == "research"
    
    def test_prompt_building(self, work_generator, sample_job):
        """Test prompt building for work generation"""
        prompt = work_generator._build_prompt(sample_job)
        
        # Verify prompt contains key information
        assert sample_job.title in prompt
        assert sample_job.description in prompt
        assert "python" in prompt.lower()
        assert "writing" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_template_based_generation_writing(self, work_generator):
        """Test template-based work generation for writing"""
        job = Job(
            id="job_001",
            title="Write a Python Tutorial",
            description="Write a comprehensive tutorial on Python basics",
            required_skills=["writing", "python"],
            budget=500.0,
            deadline=datetime.now() + timedelta(days=7),
            platform="upwork"
        )
        
        work = work_generator._generate_template_based(job)
        
        # Verify work is generated
        assert work is not None
        assert len(work) > 300  # Should be substantial
        assert "Python" in work or "python" in work.lower()
    
    @pytest.mark.asyncio
    async def test_template_based_generation_coding(self, work_generator):
        """Test template-based work generation for coding"""
        job = Job(
            id="job_001",
            title="Develop a Python Script",
            description="Write Python code to process data",
            required_skills=["python"],
            budget=200.0,
            deadline=datetime.now() + timedelta(days=7),
            platform="upwork"
        )
        
        work = work_generator._generate_template_based(job)
        
        # Verify work is generated and contains code
        assert work is not None
        assert "def " in work or "class " in work
        assert "import" in work
    
    @pytest.mark.asyncio
    async def test_work_generation_fallback(self, work_generator, sample_job):
        """Test work generation falls back to template when Avus unavailable"""
        work_generator.avus_available = False
        
        work = await work_generator.generate_work(sample_job)
        
        # Should generate work using template
        assert work is not None
        assert len(work) > 300


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestQualityValidation:
    """Tests for quality validation system"""
    
    @pytest.fixture
    def quality_validator(self):
        """Create a quality validator instance"""
        return QualityValidator()
    
    @pytest.fixture
    def sample_job(self):
        """Create a sample job"""
        return Job(
            id="job_001",
            title="Write a Python Tutorial",
            description="Write a comprehensive tutorial on Python basics",
            required_skills=["writing", "python"],
            budget=500.0,
            deadline=datetime.now() + timedelta(days=7),
            platform="upwork"
        )
    
    def test_length_validation_sufficient(self, quality_validator, sample_job):
        """Test length validation with sufficient content"""
        work = "This is a comprehensive tutorial about Python. " * 50  # ~500 words
        
        score = quality_validator._validate_length(work, sample_job)
        
        # Should have good score for sufficient length
        assert score >= 0.8
    
    def test_length_validation_insufficient(self, quality_validator, sample_job):
        """Test length validation with insufficient content"""
        work = "This is too short."
        
        score = quality_validator._validate_length(work, sample_job)
        
        # Should have low score for insufficient length
        assert score < 0.5
    
    def test_coherence_validation_good(self, quality_validator):
        """Test coherence validation with well-structured content"""
        work = """
        # Introduction
        This is the introduction.
        
        # Main Content
        This is the main content with multiple sentences.
        Each sentence provides valuable information.
        The structure is clear and logical.
        
        # Conclusion
        This is the conclusion summarizing the content.
        """
        
        score = quality_validator._validate_coherence(work)
        
        # Should have good score for well-structured content
        assert score >= 0.7
    
    def test_coherence_validation_poor(self, quality_validator):
        """Test coherence validation with poorly-structured content"""
        work = "word word word word word word word word word word"
        
        score = quality_validator._validate_coherence(work)
        
        # Should have low score for poor structure
        assert score < 0.7
    
    def test_relevance_validation_relevant(self, quality_validator, sample_job):
        """Test relevance validation with relevant content"""
        work = """
        Python Tutorial: Comprehensive Guide to Python Basics
        
        This tutorial covers Python programming fundamentals.
        Learn about Python syntax, functions, and libraries.
        Master Python for data analysis and web development.
        """
        
        score = quality_validator._validate_relevance(work, sample_job)
        
        # Should have good score for relevant content
        assert score >= 0.6
    
    def test_relevance_validation_irrelevant(self, quality_validator, sample_job):
        """Test relevance validation with irrelevant content"""
        work = "This is about cooking recipes and food preparation."
        
        score = quality_validator._validate_relevance(work, sample_job)
        
        # Should have low score for irrelevant content
        assert score < 0.5
    
    def test_overall_quality_score(self, quality_validator, sample_job):
        """Test overall quality score calculation"""
        work = """
        # Python Tutorial
        
        Python is a powerful programming language.
        This tutorial covers Python basics including syntax, functions, and libraries.
        
        ## Getting Started
        To get started with Python, you need to install it first.
        Python can be downloaded from the official website.
        
        ## Basic Syntax
        Python uses indentation for code blocks.
        Variables are created by assignment.
        Functions are defined using the def keyword.
        
        ## Conclusion
        Python is an excellent language for beginners and professionals alike.
        """ * 3  # Make it long enough
        
        score = quality_validator.validate_quality(work, sample_job)
        
        # Should have reasonable score
        assert 0.5 <= score <= 1.0
    
    def test_quality_score_with_generic_content(self, quality_validator, sample_job):
        """Test quality score penalizes generic content"""
        work = """
        This is a placeholder for the tutorial.
        TODO: Add actual content here.
        Fix me: This needs to be implemented.
        """
        
        score = quality_validator.validate_quality(work, sample_job)
        
        # Should have low score for generic/placeholder content
        assert score < 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# CONCEPT EXTRACTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestConceptExtraction:
    """Tests for concept extraction from learning resources"""
    
    @pytest.fixture
    def learning_engine(self):
        """Create a learning engine instance"""
        return LearningEngine()
    
    @pytest.mark.asyncio
    async def test_extract_concepts_from_youtube(self, learning_engine):
        """Test concept extraction from YouTube videos"""
        resource = LearningResource(
            url="https://youtube.com/watch?v=test",
            title="Python Programming Tutorial for Beginners",
            type="youtube",
            topic="python",
            duration_minutes=30,
            completed=False
        )
        
        concepts = await learning_engine._extract_concepts_from_youtube(resource)
        
        # Should extract concepts from title
        assert len(concepts) > 0
        assert any("python" in c.lower() for c in concepts)
    
    @pytest.mark.asyncio
    async def test_extract_concepts_from_web(self, learning_engine):
        """Test concept extraction from web articles"""
        resource = LearningResource(
            url="https://example.com/python-tutorial",
            title="Python Tutorial: Learn Python Programming",
            type="article",
            topic="python",
            duration_minutes=10,
            completed=False
        )
        
        mock_html = """
        <html>
        <body>
        <h1>Python Tutorial</h1>
        <p>Python is a powerful programming language used for data analysis and web development.</p>
        <p>Learn about Python syntax, functions, and libraries.</p>
        </body>
        </html>
        """
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.text = mock_html
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            concepts = await learning_engine._extract_concepts_from_web(resource)
        
        # Should extract concepts from content
        assert len(concepts) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests for work generation and quality validation"""
    
    @pytest.mark.asyncio
    async def test_complete_work_generation_workflow(self):
        """Test complete workflow: generate work and validate quality"""
        # Create worker
        worker = JanusAutonomousWorker()
        
        # Create a sample job
        job = Job(
            id="job_001",
            title="Write a Python Tutorial",
            description="Write a comprehensive tutorial on Python basics",
            required_skills=["writing", "python"],
            budget=500.0,
            deadline=datetime.now() + timedelta(days=7),
            platform="upwork",
            status=JobStatus.CLAIMED
        )
        
        # Generate work
        work = await worker.work_generator.generate_work(job)
        assert work is not None
        
        # Validate quality
        quality_score = worker.quality_validator.validate_quality(work, job)
        assert 0 <= quality_score <= 1.0
        
        # Should have reasonable quality
        assert quality_score >= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
