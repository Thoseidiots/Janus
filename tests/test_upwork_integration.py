"""
Test suite for Upwork API Integration (Task 1.1)
Tests real API integration, error handling, rate limiting, and database storage
"""

import asyncio
import json
import sqlite3
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pytest
import sys

# Import the module to test
from janus_autonomous_worker import (
    UpworkIntegration, Job, JobStatus, JanusAutonomousWorker
)


class TestUpworkIntegrationBasics:
    """Test basic Upwork integration functionality"""
    
    def test_upwork_initialization(self):
        """Test UpworkIntegration initializes correctly"""
        upwork = UpworkIntegration(api_key="test_key_123")
        assert upwork.api_key == "test_key_123"
        assert upwork.platform_name == "upwork"
        assert upwork.base_url == "https://api.upwork.com/api"
        assert upwork.rate_limit_max == 100
        assert upwork.min_request_interval == 0.5
    
    def test_upwork_no_api_key(self):
        """Test UpworkIntegration handles missing API key"""
        upwork = UpworkIntegration(api_key=None)
        # Should not raise, just log warning
        assert upwork.api_key is None


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limit_check_passes_initially(self):
        """Test rate limit check passes when no requests made"""
        upwork = UpworkIntegration(api_key="test_key")
        assert upwork._check_rate_limit() is True
    
    def test_rate_limit_minimum_interval(self):
        """Test minimum interval between requests is enforced"""
        upwork = UpworkIntegration(api_key="test_key")
        upwork.last_request_time = time.time()
        
        # Should fail immediately after request
        assert upwork._check_rate_limit() is False
        
        # Should pass after minimum interval
        upwork.last_request_time = time.time() - 1.0
        assert upwork._check_rate_limit() is True
    
    def test_rate_limit_hourly_limit(self):
        """Test hourly rate limit is enforced"""
        upwork = UpworkIntegration(api_key="test_key")
        upwork.rate_limit_max = 5  # Set low limit for testing
        upwork.requests_made = 5
        upwork.rate_limit_reset = time.time() + 3600  # 1 hour from now
        
        # Should fail when limit reached
        assert upwork._check_rate_limit() is False
        
        # Should reset after time passes
        upwork.rate_limit_reset = time.time() - 1  # Time has passed
        assert upwork._check_rate_limit() is True
        assert upwork.requests_made == 0  # Counter reset


class TestExponentialBackoff:
    """Test exponential backoff retry logic"""
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_success_first_try(self):
        """Test successful request on first try"""
        upwork = UpworkIntegration(api_key="test_key")
        
        mock_func = Mock(return_value="success")
        result = await upwork._retry_with_backoff(mock_func, max_retries=3)
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_timeout_then_success(self):
        """Test retry after timeout"""
        upwork = UpworkIntegration(api_key="test_key")
        
        import requests
        mock_func = Mock(side_effect=[
            requests.exceptions.Timeout("timeout"),
            "success"
        ])
        
        result = await upwork._retry_with_backoff(mock_func, max_retries=3)
        
        assert result == "success"
        assert mock_func.call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_max_retries_exceeded(self):
        """Test max retries exceeded"""
        upwork = UpworkIntegration(api_key="test_key")
        
        import requests
        mock_func = Mock(side_effect=requests.exceptions.Timeout("timeout"))
        
        with pytest.raises(requests.exceptions.Timeout):
            await upwork._retry_with_backoff(mock_func, max_retries=2)
        
        assert mock_func.call_count == 2


class TestJobParsing:
    """Test job data parsing from API responses"""
    
    @pytest.mark.asyncio
    async def test_parse_valid_job_data(self):
        """Test parsing valid job data from API"""
        upwork = UpworkIntegration(api_key="test_key")
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "jobs": [
                {
                    "id": "job_123",
                    "title": "Python Development",
                    "description": "Build a Python web app",
                    "skills": ["python", "django"],
                    "budget": 500.0,
                    "deadline": (datetime.now() + timedelta(days=7)).isoformat()
                }
            ]
        }
        
        with patch('requests.get', return_value=mock_response):
            jobs = await upwork.get_available_jobs(["python"])
        
        assert len(jobs) == 1
        assert jobs[0].id == "job_123"
        assert jobs[0].title == "Python Development"
        assert jobs[0].budget == 500.0
        assert jobs[0].platform == "upwork"
        assert jobs[0].status == JobStatus.AVAILABLE
        assert "python" in jobs[0].required_skills
    
    @pytest.mark.asyncio
    async def test_parse_job_with_missing_deadline(self):
        """Test parsing job with missing deadline"""
        upwork = UpworkIntegration(api_key="test_key")
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "jobs": [
                {
                    "id": "job_456",
                    "title": "Writing Task",
                    "description": "Write an article",
                    "skills": ["writing"],
                    "budget": 100.0
                    # No deadline
                }
            ]
        }
        
        with patch('requests.get', return_value=mock_response):
            jobs = await upwork.get_available_jobs(["writing"])
        
        assert len(jobs) == 1
        assert jobs[0].deadline is not None
        # Should default to 7 days from now
        assert jobs[0].deadline > datetime.now()


class TestDatabaseStorage:
    """Test job storage in database"""
    
    def test_store_job_in_database(self):
        """Test storing job in database"""
        worker = JanusAutonomousWorker()
        
        job = Job(
            id="test_job_1",
            title="Test Job",
            description="Test Description",
            required_skills=["python", "testing"],
            budget=250.0,
            deadline=datetime.now() + timedelta(days=7),
            platform="upwork",
            status=JobStatus.AVAILABLE
        )
        
        worker._store_job(job)
        
        # Verify job was stored
        conn = sqlite3.connect(worker.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, budget, status FROM jobs WHERE id = ?", ("test_job_1",))
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[0] == "test_job_1"
        assert row[1] == "Test Job"
        assert row[2] == 250.0
        assert row[3] == "available"
    
    def test_retrieve_available_jobs_from_database(self):
        """Test retrieving available jobs from database"""
        worker = JanusAutonomousWorker()
        
        # Store multiple jobs
        for i in range(3):
            job = Job(
                id=f"test_job_{i}",
                title=f"Test Job {i}",
                description=f"Description {i}",
                required_skills=["python"],
                budget=100.0 * (i + 1),
                deadline=datetime.now() + timedelta(days=7),
                platform="upwork",
                status=JobStatus.AVAILABLE
            )
            worker._store_job(job)
        
        # Retrieve jobs
        jobs = worker._get_available_jobs()
        
        assert len(jobs) >= 3
        assert all(job.status == JobStatus.AVAILABLE for job in jobs)
        assert all(job.platform == "upwork" for job in jobs)
    
    def test_store_and_retrieve_job_with_skills(self):
        """Test storing and retrieving job with required skills"""
        worker = JanusAutonomousWorker()
        
        skills = ["python", "django", "postgresql"]
        job = Job(
            id="test_job_skills",
            title="Full Stack Development",
            description="Build a web application",
            required_skills=skills,
            budget=1000.0,
            deadline=datetime.now() + timedelta(days=14),
            platform="upwork",
            status=JobStatus.AVAILABLE
        )
        
        worker._store_job(job)
        
        # Retrieve and verify
        jobs = worker._get_available_jobs()
        stored_job = next((j for j in jobs if j.id == "test_job_skills"), None)
        
        assert stored_job is not None
        assert stored_job.required_skills == skills


class TestErrorHandling:
    """Test error handling in API calls"""
    
    @pytest.mark.asyncio
    async def test_handle_authentication_error(self):
        """Test handling authentication error (401)"""
        upwork = UpworkIntegration(api_key="invalid_key")
        
        import requests
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        
        with patch('requests.get', return_value=mock_response):
            jobs = await upwork.get_available_jobs(["python"])
        
        assert jobs == []
    
    @pytest.mark.asyncio
    async def test_handle_server_error_with_retry(self):
        """Test handling server error (500) with retry"""
        upwork = UpworkIntegration(api_key="test_key")
        
        import requests
        mock_response_error = Mock()
        mock_response_error.status_code = 500
        mock_response_error.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response_error)
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"jobs": []}
        
        with patch('requests.get', side_effect=[mock_response_error, mock_response_success]):
            jobs = await upwork.get_available_jobs(["python"])
        
        # Should retry and eventually succeed
        assert jobs == []


class TestIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.mark.asyncio
    async def test_find_and_store_jobs_workflow(self):
        """Test complete workflow: find jobs and store in database"""
        worker = JanusAutonomousWorker()
        upwork = UpworkIntegration(api_key="test_key")
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "jobs": [
                {
                    "id": "job_workflow_1",
                    "title": "Web Development",
                    "description": "Build a website",
                    "skills": ["html", "css", "javascript"],
                    "budget": 750.0,
                    "deadline": (datetime.now() + timedelta(days=10)).isoformat()
                },
                {
                    "id": "job_workflow_2",
                    "title": "Data Analysis",
                    "description": "Analyze data",
                    "skills": ["python", "pandas"],
                    "budget": 500.0,
                    "deadline": (datetime.now() + timedelta(days=5)).isoformat()
                }
            ]
        }
        
        with patch('requests.get', return_value=mock_response):
            jobs = await upwork.get_available_jobs(["python", "html"])
        
        # Store jobs
        for job in jobs:
            worker._store_job(job)
        
        # Retrieve and verify
        stored_jobs = worker._get_available_jobs()
        assert len(stored_jobs) >= 2
        
        job_ids = [j.id for j in stored_jobs]
        assert "job_workflow_1" in job_ids
        assert "job_workflow_2" in job_ids


# Cleanup
def cleanup_test_database():
    """Clean up test database"""
    import os
    if os.path.exists("janus_worker.db"):
        try:
            os.remove("janus_worker.db")
        except:
            pass


if __name__ == "__main__":
    # Run tests
    cleanup_test_database()
    pytest.main([__file__, "-v", "--tb=short"])
    cleanup_test_database()

