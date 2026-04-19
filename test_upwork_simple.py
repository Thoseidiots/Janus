"""
Simple test script for Upwork API Integration (Task 1.1)
Tests real API integration, error handling, rate limiting, and database storage
"""

import sys
import os
import json
import sqlite3
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add current directory to path to avoid __init__.py issues
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from the module
import importlib.util
spec = importlib.util.spec_from_file_location("janus_autonomous_worker", "janus_autonomous_worker.py")
janus_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(janus_module)

UpworkIntegration = janus_module.UpworkIntegration
Job = janus_module.Job
JobStatus = janus_module.JobStatus
JanusAutonomousWorker = janus_module.JanusAutonomousWorker


def test_upwork_initialization():
    """Test UpworkIntegration initializes correctly"""
    print("TEST: Upwork initialization...")
    upwork = UpworkIntegration(api_key="test_key_123")
    assert upwork.api_key == "test_key_123"
    assert upwork.platform_name == "upwork"
    assert upwork.base_url == "https://api.upwork.com/api"
    assert upwork.rate_limit_max == 100
    assert upwork.min_request_interval == 0.5
    print("✓ PASSED: Upwork initialization")


def test_upwork_no_api_key():
    """Test UpworkIntegration handles missing API key"""
    print("TEST: Upwork with no API key...")
    upwork = UpworkIntegration(api_key=None)
    assert upwork.api_key is None
    print("✓ PASSED: Upwork with no API key")


def test_rate_limit_check_passes_initially():
    """Test rate limit check passes when no requests made"""
    print("TEST: Rate limit check passes initially...")
    upwork = UpworkIntegration(api_key="test_key")
    assert upwork._check_rate_limit() is True
    print("✓ PASSED: Rate limit check passes initially")


def test_rate_limit_minimum_interval():
    """Test minimum interval between requests is enforced"""
    print("TEST: Rate limit minimum interval...")
    upwork = UpworkIntegration(api_key="test_key")
    upwork.last_request_time = time.time()
    
    # Should fail immediately after request
    assert upwork._check_rate_limit() is False
    
    # Should pass after minimum interval
    upwork.last_request_time = time.time() - 1.0
    assert upwork._check_rate_limit() is True
    print("✓ PASSED: Rate limit minimum interval")


def test_rate_limit_hourly_limit():
    """Test hourly rate limit is enforced"""
    print("TEST: Rate limit hourly limit...")
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
    print("✓ PASSED: Rate limit hourly limit")


def test_store_job_in_database():
    """Test storing job in database"""
    print("TEST: Store job in database...")
    
    # Clean up any existing test database
    if os.path.exists("test_janus_worker.db"):
        os.remove("test_janus_worker.db")
    
    worker = JanusAutonomousWorker()
    worker.db_path = "test_janus_worker.db"
    worker._init_database()
    
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
    
    # Clean up
    os.remove("test_janus_worker.db")
    print("✓ PASSED: Store job in database")


def test_retrieve_available_jobs_from_database():
    """Test retrieving available jobs from database"""
    print("TEST: Retrieve available jobs from database...")
    
    # Clean up any existing test database
    if os.path.exists("test_janus_worker.db"):
        os.remove("test_janus_worker.db")
    
    worker = JanusAutonomousWorker()
    worker.db_path = "test_janus_worker.db"
    worker._init_database()
    
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
    
    # Clean up
    os.remove("test_janus_worker.db")
    print("✓ PASSED: Retrieve available jobs from database")


def test_store_and_retrieve_job_with_skills():
    """Test storing and retrieving job with required skills"""
    print("TEST: Store and retrieve job with skills...")
    
    # Clean up any existing test database
    if os.path.exists("test_janus_worker.db"):
        os.remove("test_janus_worker.db")
    
    worker = JanusAutonomousWorker()
    worker.db_path = "test_janus_worker.db"
    worker._init_database()
    
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
    
    # Clean up
    os.remove("test_janus_worker.db")
    print("✓ PASSED: Store and retrieve job with skills")


def test_database_schema():
    """Test database schema is created correctly"""
    print("TEST: Database schema...")
    
    # Clean up any existing test database
    if os.path.exists("test_janus_worker.db"):
        os.remove("test_janus_worker.db")
    
    worker = JanusAutonomousWorker()
    worker.db_path = "test_janus_worker.db"
    worker._init_database()
    
    # Verify tables exist
    conn = sqlite3.connect(worker.db_path)
    cursor = conn.cursor()
    
    # Check jobs table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
    assert cursor.fetchone() is not None
    
    # Check skills table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='skills'")
    assert cursor.fetchone() is not None
    
    # Check learning table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='learning'")
    assert cursor.fetchone() is not None
    
    # Check financials table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='financials'")
    assert cursor.fetchone() is not None
    
    # Check indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_jobs_status'")
    assert cursor.fetchone() is not None
    
    conn.close()
    
    # Clean up
    os.remove("test_janus_worker.db")
    print("✓ PASSED: Database schema")


def test_job_status_enum():
    """Test JobStatus enum values"""
    print("TEST: JobStatus enum...")
    assert JobStatus.AVAILABLE.value == "available"
    assert JobStatus.CLAIMED.value == "claimed"
    assert JobStatus.IN_PROGRESS.value == "in_progress"
    assert JobStatus.COMPLETED.value == "completed"
    assert JobStatus.FAILED.value == "failed"
    assert JobStatus.PAID.value == "paid"
    print("✓ PASSED: JobStatus enum")


def test_job_data_model():
    """Test Job data model"""
    print("TEST: Job data model...")
    
    deadline = datetime.now() + timedelta(days=7)
    job = Job(
        id="job_123",
        title="Test Job",
        description="Test Description",
        required_skills=["python", "testing"],
        budget=500.0,
        deadline=deadline,
        platform="upwork",
        status=JobStatus.AVAILABLE
    )
    
    assert job.id == "job_123"
    assert job.title == "Test Job"
    assert job.description == "Test Description"
    assert job.required_skills == ["python", "testing"]
    assert job.budget == 500.0
    assert job.deadline == deadline
    assert job.platform == "upwork"
    assert job.status == JobStatus.AVAILABLE
    assert job.claimed_by is None
    assert job.payment_received is False
    
    print("✓ PASSED: Job data model")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("UPWORK API INTEGRATION TEST SUITE (Task 1.1)")
    print("="*60 + "\n")
    
    tests = [
        test_upwork_initialization,
        test_upwork_no_api_key,
        test_rate_limit_check_passes_initially,
        test_rate_limit_minimum_interval,
        test_rate_limit_hourly_limit,
        test_job_status_enum,
        test_job_data_model,
        test_database_schema,
        test_store_job_in_database,
        test_retrieve_available_jobs_from_database,
        test_store_and_retrieve_job_with_skills,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
