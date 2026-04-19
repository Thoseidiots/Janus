"""
Test suite for Fiverr API Integration (Task 1.2)

Tests the real Fiverr API integration with:
- Authentication
- Job/gig discovery
- Error handling with exponential backoff
- Rate limiting
- Data parsing
- Database storage
"""

import pytest
import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import requests
from pathlib import Path

# Import the classes we're testing
import sys
sys.path.insert(0, '.')
from janus_autonomous_worker import (
    FiverrIntegration, Job, JobStatus, JanusAutonomousWorker
)


class TestFiverrIntegrationBasics:
    """Test basic Fiverr integration setup"""
    
    def test_fiverr_init_with_api_key(self):
        """Test FiverrIntegration initializes with API key"""
        integration = FiverrIntegration(api_key="test_key_123")
        assert integration.api_key == "test_key_123"
        assert integration.platform_name == "fiverr"
        assert integration.base_url == "https://api.fiverr.com/v1"
    
    def test_fiverr_init_without_api_key(self):
        """Test FiverrIntegration initializes without API key (from env)"""
        integration = FiverrIntegration(api_key=None)
        # Should load from environment or be None
        assert integration.platform_name == "fiverr"
    
    def test_fiverr_rate_limit_initialization(self):
        """Test rate limiting is properly initialized"""
        integration = FiverrIntegration(api_key="test_key")
        assert integration.rate_limit_max == 100
        assert integration.min_request_interval == 0.5
        assert integration.requests_made == 0


class TestFiverrRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limit_check_passes_initially(self):
        """Test rate limit check passes on first request"""
        integration = FiverrIntegration(api_key="test_key")
        assert integration._check_rate_limit() is True
    
    def test_rate_limit_respects_minimum_interval(self):
        """Test minimum interval between requests is enforced"""
        import time
        integration = FiverrIntegration(api_key="test_key")
        
        # First request should pass
        assert integration._check_rate_limit() is True
        integration.last_request_time = time.time()
        
        # Immediate second request should fail
        assert integration._check_rate_limit() is False
    
    def test_rate_limit_resets_after_hour(self):
        """Test hourly rate limit counter resets"""
        import time
        integration = FiverrIntegration(api_key="test_key")
        
        # Simulate reaching rate limit
        integration.requests_made = 100
        integration.rate_limit_reset = time.time() + 100  # 100 seconds in future
        
        # Should be rate limited
        assert integration._check_rate_limit() is False
        
        # Simulate time passing (reset time reached)
        integration.rate_limit_reset = time.time() - 1  # Already passed
        
        # Should reset and allow request
        assert integration._check_rate_limit() is True
        assert integration.requests_made == 0


class TestFiverrRetryLogic:
    """Test exponential backoff retry logic"""
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_success_first_try(self):
        """Test successful request on first try"""
        integration = FiverrIntegration(api_key="test_key")
        
        mock_func = Mock(return_value="success")
        result = await integration._retry_with_backoff(mock_func, max_retries=3)
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_timeout_then_success(self):
        """Test retry after timeout"""
        integration = FiverrIntegration(api_key="test_key")
        
        # First call raises timeout, second succeeds
        mock_func = Mock(side_effect=[
            requests.exceptions.Timeout("timeout"),
            "success"
        ])
        
        result = await integration._retry_with_backoff(mock_func, max_retries=3)
        
        assert result == "success"
        assert mock_func.call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_connection_error(self):
        """Test retry after connection error"""
        integration = FiverrIntegration(api_key="test_key")
        
        # First call raises connection error, second succeeds
        mock_func = Mock(side_effect=[
            requests.exceptions.ConnectionError("connection failed"),
            "success"
        ])
        
        result = await integration._retry_with_backoff(mock_func, max_retries=3)
        
        assert result == "success"
        assert mock_func.call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_rate_limit_429(self):
        """Test retry on 429 rate limit error"""
        integration = FiverrIntegration(api_key="test_key")
        
        # Create mock response with 429 status
        mock_response = Mock()
        mock_response.status_code = 429
        
        # First call raises 429, second succeeds
        mock_func = Mock(side_effect=[
            requests.exceptions.HTTPError(response=mock_response),
            "success"
        ])
        
        result = await integration._retry_with_backoff(mock_func, max_retries=3)
        
        assert result == "success"
        assert mock_func.call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_server_error_5xx(self):
        """Test retry on 5xx server errors"""
        integration = FiverrIntegration(api_key="test_key")
        
        # Create mock response with 500 status
        mock_response = Mock()
        mock_response.status_code = 500
        
        # First call raises 500, second succeeds
        mock_func = Mock(side_effect=[
            requests.exceptions.HTTPError(response=mock_response),
            "success"
        ])
        
        result = await integration._retry_with_backoff(mock_func, max_retries=3)
        
        assert result == "success"
        assert mock_func.call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_client_error_no_retry(self):
        """Test no retry on 4xx client errors (except 429)"""
        integration = FiverrIntegration(api_key="test_key")
        
        # Create mock response with 400 status
        mock_response = Mock()
        mock_response.status_code = 400
        
        mock_func = Mock(side_effect=requests.exceptions.HTTPError(response=mock_response))
        
        with pytest.raises(requests.exceptions.HTTPError):
            await integration._retry_with_backoff(mock_func, max_retries=3)
        
        # Should only try once (no retry on 4xx)
        assert mock_func.call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_max_retries_exceeded(self):
        """Test max retries exceeded"""
        integration = FiverrIntegration(api_key="test_key")
        
        # Always raise timeout
        mock_func = Mock(side_effect=requests.exceptions.Timeout("timeout"))
        
        with pytest.raises(requests.exceptions.Timeout):
            await integration._retry_with_backoff(mock_func, max_retries=2)
        
        # Should try max_retries times
        assert mock_func.call_count == 2


class TestFiverrGigParsing:
    """Test parsing of Fiverr gig data"""
    
    @pytest.mark.asyncio
    async def test_parse_gig_data_basic(self):
        """Test parsing basic gig data"""
        integration = FiverrIntegration(api_key="test_key")
        
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "gigs": [
                {
                    "id": "gig_123",
                    "title": "Write Python Code",
                    "description": "I will write Python code for you",
                    "price": 50.0,
                    "delivery_time_in_days": 3,
                    "tags": ["python", "coding"],
                    "requirements": ["Python 3.9+"]
                }
            ]
        }
        
        with patch.object(integration, '_retry_with_backoff', return_value=mock_response):
            jobs = await integration.get_available_jobs(["python"])
        
        assert len(jobs) == 1
        job = jobs[0]
        assert job.id == "gig_123"
        assert job.title == "Write Python Code"
        assert job.budget == 50.0
        assert job.platform == "fiverr"
        assert job.status == JobStatus.AVAILABLE
    
    @pytest.mark.asyncio
    async def test_parse_gig_data_with_alternative_price_field(self):
        """Test parsing gig with alternative price field"""
        integration = FiverrIntegration(api_key="test_key")
        
        # Mock API response with starting_price instead of price
        mock_response = Mock()
        mock_response.json.return_value = {
            "gigs": [
                {
                    "id": "gig_456",
                    "title": "Design Logo",
                    "description": "Professional logo design",
                    "price": 0,  # No price field
                    "starting_price": 75.0,  # Use starting_price instead
                    "delivery_time_in_days": 5,
                    "tags": ["design"],
                }
            ]
        }
        
        with patch.object(integration, '_retry_with_backoff', return_value=mock_response):
            jobs = await integration.get_available_jobs(["design"])
        
        assert len(jobs) == 1
        assert jobs[0].budget == 75.0
    
    @pytest.mark.asyncio
    async def test_parse_gig_deadline_calculation(self):
        """Test deadline is calculated from delivery_time_in_days"""
        integration = FiverrIntegration(api_key="test_key")
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "gigs": [
                {
                    "id": "gig_789",
                    "title": "Test Gig",
                    "description": "Test",
                    "price": 100.0,
                    "delivery_time_in_days": 7,
                    "tags": [],
                }
            ]
        }
        
        with patch.object(integration, '_retry_with_backoff', return_value=mock_response):
            jobs = await integration.get_available_jobs([])
        
        assert len(jobs) == 1
        job = jobs[0]
        # Deadline should be approximately 7 days from now
        expected_deadline = datetime.now() + timedelta(days=7)
        assert abs((job.deadline - expected_deadline).total_seconds()) < 60  # Within 1 minute
    
    @pytest.mark.asyncio
    async def test_parse_multiple_gigs(self):
        """Test parsing multiple gigs"""
        integration = FiverrIntegration(api_key="test_key")
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "gigs": [
                {
                    "id": f"gig_{i}",
                    "title": f"Gig {i}",
                    "description": f"Description {i}",
                    "price": 50.0 * (i + 1),
                    "delivery_time_in_days": 3,
                    "tags": ["tag1", "tag2"],
                }
                for i in range(5)
            ]
        }
        
        with patch.object(integration, '_retry_with_backoff', return_value=mock_response):
            jobs = await integration.get_available_jobs(["tag1"])
        
        assert len(jobs) == 5
        for i, job in enumerate(jobs):
            assert job.id == f"gig_{i}"
            assert job.budget == 50.0 * (i + 1)


class TestFiverrAPIErrors:
    """Test error handling for Fiverr API"""
    
    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty_list(self):
        """Test get_available_jobs returns empty list when no API key"""
        integration = FiverrIntegration(api_key=None)
        jobs = await integration.get_available_jobs(["python"])
        assert jobs == []
    
    @pytest.mark.asyncio
    async def test_authentication_error_401(self):
        """Test handling of 401 authentication error"""
        integration = FiverrIntegration(api_key="invalid_key")
        
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        
        with patch.object(integration, '_retry_with_backoff', 
                         side_effect=requests.exceptions.HTTPError(response=mock_response)):
            jobs = await integration.get_available_jobs(["python"])
        
        assert jobs == []
    
    @pytest.mark.asyncio
    async def test_permission_error_403(self):
        """Test handling of 403 permission error"""
        integration = FiverrIntegration(api_key="test_key")
        
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        
        with patch.object(integration, '_retry_with_backoff',
                         side_effect=requests.exceptions.HTTPError(response=mock_response)):
            jobs = await integration.get_available_jobs(["python"])
        
        assert jobs == []
    
    @pytest.mark.asyncio
    async def test_generic_exception_handling(self):
        """Test handling of generic exceptions"""
        integration = FiverrIntegration(api_key="test_key")
        
        with patch.object(integration, '_retry_with_backoff',
                         side_effect=Exception("Network error")):
            jobs = await integration.get_available_jobs(["python"])
        
        assert jobs == []


class TestFiverrJobClaiming:
    """Test job claiming functionality"""
    
    @pytest.mark.asyncio
    async def test_claim_job_success(self):
        """Test successful job claiming"""
        integration = FiverrIntegration(api_key="test_key")
        
        mock_response = Mock()
        
        with patch.object(integration, '_retry_with_backoff', return_value=mock_response):
            result = await integration.claim_job("gig_123")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_claim_job_no_api_key(self):
        """Test claiming job without API key"""
        integration = FiverrIntegration(api_key=None)
        result = await integration.claim_job("gig_123")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_claim_job_not_found(self):
        """Test claiming non-existent job"""
        integration = FiverrIntegration(api_key="test_key")
        
        mock_response = Mock()
        mock_response.status_code = 404
        
        with patch.object(integration, '_retry_with_backoff',
                         side_effect=requests.exceptions.HTTPError(response=mock_response)):
            result = await integration.claim_job("gig_999")
        
        assert result is False


class TestFiverrWorkSubmission:
    """Test work submission functionality"""
    
    @pytest.mark.asyncio
    async def test_submit_work_success(self):
        """Test successful work submission"""
        integration = FiverrIntegration(api_key="test_key")
        
        mock_response = Mock()
        
        with patch.object(integration, '_retry_with_backoff', return_value=mock_response):
            result = await integration.submit_work("gig_123", "My work content")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_submit_work_no_api_key(self):
        """Test submitting work without API key"""
        integration = FiverrIntegration(api_key=None)
        result = await integration.submit_work("gig_123", "content")
        assert result is False


class TestFiverrPaymentRetrieval:
    """Test payment retrieval functionality"""
    
    @pytest.mark.asyncio
    async def test_get_payment_success(self):
        """Test successful payment retrieval"""
        integration = FiverrIntegration(api_key="test_key")
        
        mock_response = Mock()
        mock_response.json.return_value = {"amount": 50.0}
        
        with patch.object(integration, '_retry_with_backoff', return_value=mock_response):
            amount = await integration.get_payment("gig_123")
        
        assert amount == 50.0
    
    @pytest.mark.asyncio
    async def test_get_payment_no_api_key(self):
        """Test getting payment without API key"""
        integration = FiverrIntegration(api_key=None)
        amount = await integration.get_payment("gig_123")
        assert amount is None
    
    @pytest.mark.asyncio
    async def test_get_payment_not_found(self):
        """Test getting payment for non-existent gig"""
        integration = FiverrIntegration(api_key="test_key")
        
        mock_response = Mock()
        mock_response.status_code = 404
        
        with patch.object(integration, '_retry_with_backoff',
                         side_effect=requests.exceptions.HTTPError(response=mock_response)):
            amount = await integration.get_payment("gig_999")
        
        assert amount is None


class TestFiverrDatabaseIntegration:
    """Test integration with database storage"""
    
    def test_fiverr_gigs_stored_in_database(self):
        """Test that Fiverr gigs are stored in database"""
        # Create a temporary worker
        worker = JanusAutonomousWorker()
        
        # Create a test job
        job = Job(
            id="fiverr_gig_123",
            title="Write Python Code",
            description="Professional Python code",
            required_skills=["python", "coding"],
            budget=100.0,
            deadline=datetime.now() + timedelta(days=3),
            platform="fiverr",
            status=JobStatus.AVAILABLE
        )
        
        # Store in database
        worker._store_job(job)
        
        # Retrieve from database
        retrieved_jobs = worker._get_available_jobs()
        
        # Verify job was stored and retrieved
        assert len(retrieved_jobs) > 0
        found_job = next((j for j in retrieved_jobs if j.id == "fiverr_gig_123"), None)
        assert found_job is not None
        assert found_job.title == "Write Python Code"
        assert found_job.platform == "fiverr"
        assert found_job.budget == 100.0
        
        # Cleanup
        import os
        if os.path.exists(worker.db_path):
            os.remove(worker.db_path)


class TestFiverrIntegrationAcceptanceCriteria:
    """Test acceptance criteria for Task 1.2"""
    
    @pytest.mark.asyncio
    async def test_authenticate_with_fiverr_api_key(self):
        """AC: Authenticate with Fiverr using API key"""
        integration = FiverrIntegration(api_key="test_fiverr_key_123")
        assert integration.api_key == "test_fiverr_key_123"
        assert integration.platform_name == "fiverr"
    
    @pytest.mark.asyncio
    async def test_query_fiverr_api_for_gigs(self):
        """AC: Successfully query Fiverr API for available gigs"""
        integration = FiverrIntegration(api_key="test_key")
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "gigs": [
                {
                    "id": "gig_1",
                    "title": "Test Gig",
                    "description": "Test",
                    "price": 50.0,
                    "delivery_time_in_days": 3,
                    "tags": ["test"],
                }
            ]
        }
        
        with patch.object(integration, '_retry_with_backoff', return_value=mock_response):
            jobs = await integration.get_available_jobs(["test"])
        
        assert len(jobs) > 0
    
    @pytest.mark.asyncio
    async def test_parse_gig_data_correctly(self):
        """AC: Parse gig data (id, title, description, price, requirements)"""
        integration = FiverrIntegration(api_key="test_key")
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "gigs": [
                {
                    "id": "gig_123",
                    "title": "Write Article",
                    "description": "1000 word article",
                    "price": 75.0,
                    "delivery_time_in_days": 5,
                    "tags": ["writing", "content"],
                    "requirements": ["1000 words", "SEO optimized"]
                }
            ]
        }
        
        with patch.object(integration, '_retry_with_backoff', return_value=mock_response):
            jobs = await integration.get_available_jobs(["writing"])
        
        job = jobs[0]
        assert job.id == "gig_123"
        assert job.title == "Write Article"
        assert job.description == "1000 word article"
        assert job.budget == 75.0
        assert "writing" in job.required_skills
    
    @pytest.mark.asyncio
    async def test_handle_api_errors_with_exponential_backoff(self):
        """AC: Handle API errors with exponential backoff retry logic"""
        integration = FiverrIntegration(api_key="test_key")
        
        # Simulate failures then success
        mock_func = Mock(side_effect=[
            requests.exceptions.Timeout("timeout"),
            requests.exceptions.ConnectionError("connection error"),
            Mock(json=lambda: {"gigs": []})
        ])
        
        result = await integration._retry_with_backoff(mock_func, max_retries=5)
        
        # Should succeed after retries
        assert result is not None
        assert mock_func.call_count == 3
    
    @pytest.mark.asyncio
    async def test_implement_rate_limiting(self):
        """AC: Implement rate limiting to avoid account suspension"""
        integration = FiverrIntegration(api_key="test_key")
        
        # Verify rate limiting is configured
        assert integration.rate_limit_max == 100
        assert integration.min_request_interval == 0.5
        
        # Verify rate limit check works
        assert integration._check_rate_limit() is True
    
    @pytest.mark.asyncio
    async def test_store_gigs_in_database_with_correct_status(self):
        """AC: Store gigs in database with correct status"""
        worker = JanusAutonomousWorker()
        
        job = Job(
            id="fiverr_test_gig",
            title="Test Gig",
            description="Test",
            required_skills=["test"],
            budget=50.0,
            deadline=datetime.now() + timedelta(days=3),
            platform="fiverr",
            status=JobStatus.AVAILABLE
        )
        
        worker._store_job(job)
        retrieved = worker._get_available_jobs()
        
        assert len(retrieved) > 0
        found = next((j for j in retrieved if j.id == "fiverr_test_gig"), None)
        assert found is not None
        assert found.status == JobStatus.AVAILABLE
        
        # Cleanup
        import os
        if os.path.exists(worker.db_path):
            os.remove(worker.db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
