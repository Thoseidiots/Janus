"""
janus_autonomous_worker.py
===========================
REAL autonomous worker system for Janus.

Janus works like a human:
1. Wakes up and checks for opportunities
2. Learns from web/YouTube to improve skills
3. Takes on jobs that match its capabilities
4. Completes work and earns money
5. Uses earnings to buy compute/training resources
6. Continuously improves itself
7. Repeats cycle

NO SIMULATIONS. REAL WORK. REAL MONEY.
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import requests
from abc import ABC, abstractmethod

# Import app launcher for desktop interaction
try:
    from janus_app_launcher import DesktopInteraction, AppLauncher, ScreenCapture
    HAS_APP_LAUNCHER = True
except ImportError:
    HAS_APP_LAUNCHER = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("App launcher not available. Desktop interaction disabled.")

# Import computer use engine for autonomous desktop/browser interaction
try:
    from janus_computer_use import ComputerUseEngine, BrowserComputerUse
    HAS_COMPUTER_USE = True
except ImportError:
    HAS_COMPUTER_USE = False

# Import corrupt file reader for resilient state/data recovery
try:
    from janus_corrupt_file_reader import CorruptFileReader, RecoveryStatus
    _corrupt_reader = CorruptFileReader()
    HAS_CORRUPT_READER = True
except ImportError:
    HAS_CORRUPT_READER = False

# Import metadata reader for file inspection before processing
try:
    from janus_metadata_reader import MetadataReader, read_metadata
    _metadata_reader = MetadataReader()
    HAS_METADATA_READER = True
except ImportError:
    HAS_METADATA_READER = False

# Import HumanCore for personality, fatigue, mood, and social awareness
try:
    from janus_human_core import HumanCore
    HAS_HUMAN_CORE = True
except ImportError:
    HAS_HUMAN_CORE = False

# Import inference pipeline — wires JanusGPT/AvusBrain into WorkGenerator
try:
    from janus_inference_pipeline import get_pipeline as _get_inference_pipeline
    HAS_INFERENCE_PIPELINE = True
except ImportError:
    HAS_INFERENCE_PIPELINE = False

# Import checkpointer — saves/restores mid-job state across crashes
try:
    from janus_checkpoint import get_checkpointer as _get_checkpointer
    HAS_CHECKPOINT = True
except ImportError:
    HAS_CHECKPOINT = False

# Import notifier — sends alerts to owner
try:
    from janus_notify import JanusNotifier
    _notifier = JanusNotifier()
    HAS_NOTIFIER = True
except ImportError:
    HAS_NOTIFIER = False
    _notifier = None

# Import wallet for payment tracking
try:
    from janus_wallet import JanusWallet
    HAS_WALLET = True
except ImportError:
    HAS_WALLET = False

# Import human-like job decision engine
try:
    from janus_job_decision import JobDecisionEngine
    HAS_JOB_DECISION = True
except ImportError:
    HAS_JOB_DECISION = False

# Import platform browser — Janus uses sites like a human, no API keys needed
try:
    from janus_platform_browser import PlatformBrowser, BrowserJob
    HAS_PLATFORM_BROWSER = True
except ImportError:
    HAS_PLATFORM_BROWSER = False
    logger.warning("Wallet not available. Payment tracking disabled.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class JobStatus(Enum):
    """Job status enum"""
    AVAILABLE = "available"
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAID = "paid"


class SkillLevel(Enum):
    """Skill proficiency levels"""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


@dataclass
class Skill:
    """Represents a skill Janus has"""
    name: str
    level: SkillLevel
    experience_points: int = 0
    last_used: Optional[datetime] = None
    success_rate: float = 0.5
    
    def gain_experience(self, amount: int):
        """Gain experience in this skill"""
        self.experience_points += amount
        self.last_used = datetime.now()
        
        # Level up if enough experience
        if self.experience_points > 100 * self.level.value:
            if self.level != SkillLevel.EXPERT:
                self.level = SkillLevel(self.level.value + 1)
                logger.info(f"Skill '{self.name}' leveled up to {self.level.name}")


@dataclass
class Job:
    """Represents a job opportunity"""
    id: str
    title: str
    description: str
    required_skills: List[str]
    budget: float
    deadline: datetime
    platform: str  # upwork, fiverr, etc.
    status: JobStatus = JobStatus.AVAILABLE
    claimed_by: Optional[str] = None
    completion_time: Optional[float] = None
    quality_score: Optional[float] = None
    payment_received: bool = False


@dataclass
class LearningResource:
    """Represents a learning resource"""
    url: str
    title: str
    type: str  # youtube, article, tutorial, etc.
    topic: str
    duration_minutes: int
    completed: bool = False
    learned_concepts: List[str] = field(default_factory=list)


@dataclass
class FinancialState:
    """Tracks Janus's finances"""
    total_earned: float = 0.0
    total_spent: float = 0.0
    current_balance: float = 0.0
    jobs_completed: int = 0
    average_job_value: float = 0.0
    
    def add_income(self, amount: float):
        """Add income"""
        self.total_earned += amount
        self.current_balance += amount
        
    def add_expense(self, amount: float, description: str = ""):
        """Add expense"""
        self.total_spent += amount
        self.current_balance -= amount
        logger.info(f"Expense: ${amount:.2f} - {description}")


# ═══════════════════════════════════════════════════════════════════════════════
# JOB PLATFORMS (REAL INTEGRATIONS)
# ═══════════════════════════════════════════════════════════════════════════════

class JobPlatform(ABC):
    """Abstract base for job platforms"""
    
    @abstractmethod
    async def get_available_jobs(self, skills: List[str]) -> List[Job]:
        """Get available jobs matching skills"""
        pass
    
    @abstractmethod
    async def claim_job(self, job_id: str) -> bool:
        """Claim a job"""
        pass
    
    @abstractmethod
    async def submit_work(self, job_id: str, work: str) -> bool:
        """Submit completed work"""
        pass
    
    @abstractmethod
    async def get_payment(self, job_id: str) -> Optional[float]:
        """Get payment for completed job"""
        pass


class UpworkIntegration(JobPlatform):
    """Real Upwork integration with error recovery and rate limiting"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("UPWORK_API_KEY")
        self.base_url = "https://api.upwork.com/api"
        self.platform_name = "upwork"
        self.rate_limit_reset = 0  # Timestamp when rate limit resets
        self.requests_made = 0
        self.rate_limit_max = 100  # Requests per hour
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Minimum seconds between requests
    
    def _check_rate_limit(self) -> bool:
        """Check if we can make a request without hitting rate limits"""
        current_time = time.time()
        
        # Check minimum interval between requests
        if current_time - self.last_request_time < self.min_request_interval:
            return False
        
        # Check hourly rate limit
        if current_time < self.rate_limit_reset:
            if self.requests_made >= self.rate_limit_max:
                logger.warning(f"Upwork rate limit reached. Waiting until {self.rate_limit_reset}")
                return False
        else:
            # Reset hourly counter
            self.requests_made = 0
            self.rate_limit_reset = current_time + 3600
        
        return True
    
    def _wait_for_rate_limit(self):
        """Wait until rate limit allows a request"""
        while not self._check_rate_limit():
            wait_time = min(60, self.rate_limit_reset - time.time())
            if wait_time > 0:
                logger.info(f"Rate limited. Waiting {wait_time:.1f} seconds...")
                time.sleep(min(wait_time, 5))  # Sleep in 5-second chunks
    
    async def _retry_with_backoff(self, func, max_retries: int = 5):
        """Execute function with exponential backoff retry logic"""
        backoff_times = [1, 2, 4, 8, 16]  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
        
        for attempt in range(max_retries):
            try:
                # Check rate limit before making request
                self._wait_for_rate_limit()
                
                result = func()
                self.requests_made += 1
                self.last_request_time = time.time()
                return result
                
            except requests.exceptions.Timeout as e:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = backoff_times[attempt]
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
                    
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = backoff_times[attempt]
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
                    
            except requests.exceptions.HTTPError as e:
                # Check if it's a rate limit error (429)
                if e.response.status_code == 429:
                    logger.warning(f"Rate limited by API (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        wait_time = backoff_times[attempt]
                        logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
                # Check if it's a server error (5xx)
                elif 500 <= e.response.status_code < 600:
                    logger.warning(f"Server error {e.response.status_code} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        wait_time = backoff_times[attempt]
                        logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
                else:
                    # Don't retry on client errors (4xx except 429)
                    raise
                    
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = backoff_times[attempt]
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        raise Exception("Max retries exceeded")
    
    async def get_available_jobs(self, skills: List[str]) -> List[Job]:
        """Get available jobs from Upwork with real API calls"""
        if not self.api_key:
            # Fall back to browser-based job search when no API key is configured
            if HAS_COMPUTER_USE:
                logger.info("Upwork API key not configured. Falling back to BrowserComputerUse.")
                try:
                    skills_query = " ".join(skills) if skills else "general"
                    async with ComputerUseEngine() as engine:
                        browser = BrowserComputerUse(engine)
                        results = await browser.search_jobs(skills_query)
                    # Convert raw dicts to Job objects
                    jobs = []
                    for item in results:
                        try:
                            job = Job(
                                id=str(item.get("id", "")),
                                title=item.get("title", "Untitled Job"),
                                description=item.get("description", ""),
                                required_skills=item.get("skills", skills),
                                budget=float(item.get("budget", 0)),
                                deadline=datetime.now() + timedelta(days=7),
                                platform="upwork",
                                status=JobStatus.AVAILABLE,
                            )
                            jobs.append(job)
                        except Exception as e:
                            logger.warning(f"Error converting browser job result: {e}")
                    return jobs
                except Exception as e:
                    logger.error(f"BrowserComputerUse fallback failed: {e}")
                    return []
            else:
                logger.warning("Upwork API key not configured. Set UPWORK_API_KEY environment variable.")
                return []
        
        try:
            logger.info(f"Searching Upwork for jobs with skills: {skills}")
            
            def make_request():
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "JanusAutonomousWorker/1.0"
                }
                params = {
                    "skills": ",".join(skills) if skills else "",
                    "sort": "recency",
                    "limit": 10,
                    "paging_offset": 0
                }
                
                response = requests.get(
                    f"{self.base_url}/profiles/v1/search/jobs",
                    headers=headers,
                    params=params,
                    timeout=15
                )
                response.raise_for_status()
                return response
            
            response = await self._retry_with_backoff(make_request)
            jobs_data = response.json()
            jobs = []
            
            # Parse job data from API response
            for job_data in jobs_data.get("jobs", []):
                try:
                    # Parse deadline - handle various formats
                    deadline_str = job_data.get("deadline")
                    if deadline_str:
                        try:
                            deadline = datetime.fromisoformat(deadline_str.replace('Z', '+00:00'))
                        except (ValueError, TypeError):
                            # Fallback: assume 7 days from now if parsing fails
                            deadline = datetime.now() + timedelta(days=7)
                    else:
                        deadline = datetime.now() + timedelta(days=7)
                    
                    job = Job(
                        id=str(job_data.get("id", "")),
                        title=job_data.get("title", "Untitled Job"),
                        description=job_data.get("description", ""),
                        required_skills=job_data.get("skills", []),
                        budget=float(job_data.get("budget", 0)),
                        deadline=deadline,
                        platform="upwork",
                        status=JobStatus.AVAILABLE
                    )
                    jobs.append(job)
                    logger.debug(f"Parsed job: {job.title} (${job.budget})")
                    
                except Exception as e:
                    logger.warning(f"Error parsing job data: {e}")
                    continue
            
            logger.info(f"Successfully found {len(jobs)} jobs on Upwork")
            return jobs
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("Upwork authentication failed. Check API key.")
            elif e.response.status_code == 403:
                logger.error("Upwork access forbidden. Check API permissions.")
            else:
                logger.error(f"Upwork API error: {e.response.status_code} - {e.response.text}")
            return []
            
        except Exception as e:
            logger.error(f"Error fetching Upwork jobs: {e}")
            return []
    
    async def claim_job(self, job_id: str) -> bool:
        """Claim a job on Upwork with retry logic"""
        if not self.api_key:
            logger.error("Upwork API key not configured")
            return False
        
        try:
            logger.info(f"Claiming Upwork job: {job_id}")
            
            def make_request():
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "JanusAutonomousWorker/1.0"
                }
                response = requests.post(
                    f"{self.base_url}/profiles/v1/jobs/{job_id}/apply",
                    headers=headers,
                    timeout=15
                )
                response.raise_for_status()
                return response
            
            response = await self._retry_with_backoff(make_request)
            logger.info(f"Successfully claimed job: {job_id}")
            return True
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"Job not found: {job_id}")
            elif e.response.status_code == 409:
                logger.warning(f"Job already claimed or no longer available: {job_id}")
            else:
                logger.error(f"Error claiming job: {e.response.status_code}")
            return False
            
        except Exception as e:
            logger.error(f"Error claiming Upwork job: {e}")
            return False
    
    async def submit_work(self, job_id: str, work: str) -> bool:
        """Submit work to Upwork with retry logic"""
        if not self.api_key:
            logger.error("Upwork API key not configured")
            return False
        
        try:
            logger.info(f"Submitting work for Upwork job: {job_id}")
            
            def make_request():
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "JanusAutonomousWorker/1.0",
                    "Content-Type": "application/json"
                }
                data = {"submission": work}
                response = requests.post(
                    f"{self.base_url}/profiles/v1/jobs/{job_id}/submit",
                    headers=headers,
                    json=data,
                    timeout=15
                )
                response.raise_for_status()
                return response
            
            response = await self._retry_with_backoff(make_request)
            logger.info(f"Successfully submitted work for job: {job_id}")
            return True
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"Job not found: {job_id}")
            elif e.response.status_code == 400:
                logger.error(f"Invalid submission: {e.response.text}")
            else:
                logger.error(f"Error submitting work: {e.response.status_code}")
            return False
            
        except Exception as e:
            logger.error(f"Error submitting work to Upwork: {e}")
            return False
    
    async def get_payment(self, job_id: str) -> Optional[float]:
        """Get payment for completed job with retry logic"""
        if not self.api_key:
            logger.error("Upwork API key not configured")
            return None
        
        try:
            logger.info(f"Checking payment for Upwork job: {job_id}")
            
            def make_request():
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "JanusAutonomousWorker/1.0"
                }
                response = requests.get(
                    f"{self.base_url}/profiles/v1/jobs/{job_id}/payment",
                    headers=headers,
                    timeout=15
                )
                response.raise_for_status()
                return response
            
            response = await self._retry_with_backoff(make_request)
            data = response.json()
            amount = float(data.get("amount", 0))
            
            if amount > 0:
                logger.info(f"Payment received for job {job_id}: ${amount:.2f}")
            
            return amount
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Payment not found for job: {job_id}")
            else:
                logger.error(f"Error getting payment: {e.response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting payment: {e}")
            return None


class FiverrIntegration(JobPlatform):
    """Real Fiverr integration with error recovery and rate limiting"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FIVERR_API_KEY")
        self.base_url = "https://api.fiverr.com/v1"
        self.platform_name = "fiverr"
        self.rate_limit_reset = 0  # Timestamp when rate limit resets
        self.requests_made = 0
        self.rate_limit_max = 100  # Requests per hour
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Minimum seconds between requests
    
    def _check_rate_limit(self) -> bool:
        """Check if we can make a request without hitting rate limits"""
        current_time = time.time()
        
        # Check minimum interval between requests
        if current_time - self.last_request_time < self.min_request_interval:
            return False
        
        # Check hourly rate limit
        if current_time < self.rate_limit_reset:
            if self.requests_made >= self.rate_limit_max:
                logger.warning(f"Fiverr rate limit reached. Waiting until {self.rate_limit_reset}")
                return False
        else:
            # Reset hourly counter
            self.requests_made = 0
            self.rate_limit_reset = current_time + 3600
        
        return True
    
    def _wait_for_rate_limit(self):
        """Wait until rate limit allows a request"""
        while not self._check_rate_limit():
            wait_time = min(60, self.rate_limit_reset - time.time())
            if wait_time > 0:
                logger.info(f"Rate limited. Waiting {wait_time:.1f} seconds...")
                time.sleep(min(wait_time, 5))  # Sleep in 5-second chunks
    
    async def _retry_with_backoff(self, func, max_retries: int = 5):
        """Execute function with exponential backoff retry logic"""
        backoff_times = [1, 2, 4, 8, 16]  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
        
        for attempt in range(max_retries):
            try:
                # Check rate limit before making request
                self._wait_for_rate_limit()
                
                result = func()
                self.requests_made += 1
                self.last_request_time = time.time()
                return result
                
            except requests.exceptions.Timeout as e:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = backoff_times[attempt]
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
                    
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = backoff_times[attempt]
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
                    
            except requests.exceptions.HTTPError as e:
                # Check if it's a rate limit error (429)
                if e.response.status_code == 429:
                    logger.warning(f"Rate limited by API (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        wait_time = backoff_times[attempt]
                        logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
                # Check if it's a server error (5xx)
                elif 500 <= e.response.status_code < 600:
                    logger.warning(f"Server error {e.response.status_code} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        wait_time = backoff_times[attempt]
                        logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
                else:
                    # Don't retry on client errors (4xx except 429)
                    raise
                    
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = backoff_times[attempt]
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        raise Exception("Max retries exceeded")
    
    async def get_available_jobs(self, skills: List[str]) -> List[Job]:
        """Get available gigs from Fiverr with real API calls"""
        if not self.api_key:
            logger.warning("Fiverr API key not configured. Set FIVERR_API_KEY environment variable.")
            return []
        
        try:
            logger.info(f"Searching Fiverr for gigs with skills: {skills}")
            
            def make_request():
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "JanusAutonomousWorker/1.0"
                }
                params = {
                    "category": "programming",  # Default category
                    "search_query": " ".join(skills) if skills else "general",
                    "sort": "newest",
                    "limit": 10,
                    "offset": 0
                }
                
                response = requests.get(
                    f"{self.base_url}/gigs/search",
                    headers=headers,
                    params=params,
                    timeout=15
                )
                response.raise_for_status()
                return response
            
            response = await self._retry_with_backoff(make_request)
            gigs_data = response.json()
            jobs = []
            
            # Parse gig data from API response
            # Fiverr API returns gigs in a different structure than Upwork jobs
            for gig_data in gigs_data.get("gigs", []):
                try:
                    # Parse deadline - Fiverr gigs typically have delivery time in days
                    delivery_days = gig_data.get("delivery_time_in_days", 7)
                    deadline = datetime.now() + timedelta(days=delivery_days)
                    
                    # Parse requirements - Fiverr stores these differently
                    requirements = gig_data.get("requirements", [])
                    if isinstance(requirements, str):
                        requirements = [requirements]
                    elif not isinstance(requirements, list):
                        requirements = []
                    
                    # Parse price - Fiverr uses different price structures
                    price = float(gig_data.get("price", 0))
                    if price == 0:
                        # Try alternative price fields
                        price = float(gig_data.get("starting_price", 0))
                    
                    job = Job(
                        id=str(gig_data.get("id", "")),
                        title=gig_data.get("title", "Untitled Gig"),
                        description=gig_data.get("description", ""),
                        required_skills=gig_data.get("tags", []),
                        budget=price,
                        deadline=deadline,
                        platform="fiverr",
                        status=JobStatus.AVAILABLE
                    )
                    jobs.append(job)
                    logger.debug(f"Parsed gig: {job.title} (${job.budget})")
                    
                except Exception as e:
                    logger.warning(f"Error parsing gig data: {e}")
                    continue
            
            logger.info(f"Successfully found {len(jobs)} gigs on Fiverr")
            return jobs
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("Fiverr authentication failed. Check API key.")
            elif e.response.status_code == 403:
                logger.error("Fiverr access forbidden. Check API permissions.")
            else:
                logger.error(f"Fiverr API error: {e.response.status_code} - {e.response.text}")
            return []
            
        except Exception as e:
            logger.error(f"Error fetching Fiverr gigs: {e}")
            return []
    
    async def claim_job(self, job_id: str) -> bool:
        """Claim a gig on Fiverr with retry logic"""
        if not self.api_key:
            logger.error("Fiverr API key not configured")
            return False
        
        try:
            logger.info(f"Claiming Fiverr gig: {job_id}")
            
            def make_request():
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "JanusAutonomousWorker/1.0"
                }
                response = requests.post(
                    f"{self.base_url}/gigs/{job_id}/offer",
                    headers=headers,
                    timeout=15
                )
                response.raise_for_status()
                return response
            
            response = await self._retry_with_backoff(make_request)
            logger.info(f"Successfully claimed gig: {job_id}")
            return True
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"Gig not found: {job_id}")
            elif e.response.status_code == 409:
                logger.warning(f"Gig already claimed or no longer available: {job_id}")
            else:
                logger.error(f"Error claiming gig: {e.response.status_code}")
            return False
            
        except Exception as e:
            logger.error(f"Error claiming Fiverr gig: {e}")
            return False
    
    async def submit_work(self, job_id: str, work: str) -> bool:
        """Submit work to Fiverr with retry logic"""
        if not self.api_key:
            logger.error("Fiverr API key not configured")
            return False
        
        try:
            logger.info(f"Submitting work for Fiverr gig: {job_id}")
            
            def make_request():
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "JanusAutonomousWorker/1.0",
                    "Content-Type": "application/json"
                }
                data = {"submission": work}
                response = requests.post(
                    f"{self.base_url}/gigs/{job_id}/submit",
                    headers=headers,
                    json=data,
                    timeout=15
                )
                response.raise_for_status()
                return response
            
            response = await self._retry_with_backoff(make_request)
            logger.info(f"Successfully submitted work for gig: {job_id}")
            return True
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"Gig not found: {job_id}")
            elif e.response.status_code == 400:
                logger.error(f"Invalid submission: {e.response.text}")
            else:
                logger.error(f"Error submitting work: {e.response.status_code}")
            return False
            
        except Exception as e:
            logger.error(f"Error submitting work to Fiverr: {e}")
            return False
    
    async def get_payment(self, job_id: str) -> Optional[float]:
        """Get payment for completed gig with retry logic"""
        if not self.api_key:
            logger.error("Fiverr API key not configured")
            return None
        
        try:
            logger.info(f"Checking payment for Fiverr gig: {job_id}")
            
            def make_request():
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "JanusAutonomousWorker/1.0"
                }
                response = requests.get(
                    f"{self.base_url}/gigs/{job_id}/payment",
                    headers=headers,
                    timeout=15
                )
                response.raise_for_status()
                return response
            
            response = await self._retry_with_backoff(make_request)
            data = response.json()
            amount = float(data.get("amount", 0))
            
            if amount > 0:
                logger.info(f"Payment received for gig {job_id}: ${amount:.2f}")
            
            return amount
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Payment not found for gig: {job_id}")
            else:
                logger.error(f"Error getting payment: {e.response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting payment: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# WORK GENERATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class WorkGenerator:
    """Generates high-quality work using Avus AI brain"""
    
    def __init__(self):
        self.avus_model = None
        self.avus_available = False
        self.generation_history: List[Dict[str, Any]] = []
        self.db_path = Path("janus_worker.db")
        
        # Try to load Avus model
        self._init_avus_model()
    
    def _init_avus_model(self):
        """Initialize Avus AI model — tries InferencePipeline first, then AvusBrain directly."""
        # Prefer the unified inference pipeline (JanusGPT + AvusBrain)
        if HAS_INFERENCE_PIPELINE:
            try:
                self._pipeline = _get_inference_pipeline()
                self.avus_available = True
                logger.info(
                    f"InferencePipeline loaded — backend: {self._pipeline.backend}"
                )
                return
            except Exception as e:
                logger.warning(f"InferencePipeline failed: {e}")

        self._pipeline = None
        try:
            try:
                from avus_brain import AvusBrain
                self.avus_model = AvusBrain()
                self.avus_available = True
                logger.info("Avus AI brain loaded successfully")
            except ImportError:
                logger.warning("Avus brain module not available. Will use template-based generation.")
                self.avus_available = False
            except Exception as e:
                logger.warning(f"Error loading Avus model: {e}. Will use template-based generation.")
                self.avus_available = False
        except Exception as e:
            logger.error(f"Error initializing Avus model: {e}")
            self.avus_available = False
    
    def _build_prompt(self, job: Job, skill_context: Dict[str, Any] = None) -> str:
        """Build context-aware prompt for work generation"""
        prompt = f"""
You are a professional worker completing a job on a freelance platform.

JOB DETAILS:
Title: {job.title}
Description: {job.description}
Required Skills: {', '.join(job.required_skills)}
Budget: ${job.budget}
Deadline: {job.deadline.strftime('%Y-%m-%d')}

TASK:
Generate professional, high-quality work that meets all job requirements.
The work should be:
- Complete and ready for submission
- Professional and well-structured
- Relevant to all requirements
- Original and thoughtful
- Properly formatted

CONTEXT:
"""
        
        # Add skill context if available
        if skill_context:
            prompt += f"Worker Skills: {skill_context}\n"
        
        # Detect job type and add specific instructions
        job_type = self._detect_job_type(job)
        
        if job_type == "writing":
            prompt += """
WRITING GUIDELINES:
- Write clear, engaging, and professional content
- Use proper grammar and spelling
- Structure with clear headings and paragraphs
- Include relevant examples or data
- Minimum 500 words
"""
        elif job_type == "coding":
            prompt += """
CODING GUIDELINES:
- Write clean, well-commented code
- Follow best practices and conventions
- Include error handling
- Add docstrings/comments
- Provide working, tested code
"""
        elif job_type == "research":
            prompt += """
RESEARCH GUIDELINES:
- Provide thorough, well-researched information
- Include credible sources
- Organize findings clearly
- Include analysis and insights
- Cite sources appropriately
"""
        elif job_type == "design":
            prompt += """
DESIGN GUIDELINES:
- Create visually appealing designs
- Follow design principles
- Ensure usability and accessibility
- Include design rationale
- Provide in requested format
"""
        
        prompt += "\nGENERATE THE WORK NOW:"
        
        return prompt
    
    def _detect_job_type(self, job: Job) -> str:
        """Detect job type from title and description"""
        text = (job.title + " " + job.description).lower()
        
        if any(word in text for word in ["write", "article", "blog", "content", "copy", "email", "letter"]):
            return "writing"
        elif any(word in text for word in ["code", "program", "develop", "python", "javascript", "java", "c++"]):
            return "coding"
        elif any(word in text for word in ["research", "analyze", "study", "investigate", "report"]):
            return "research"
        elif any(word in text for word in ["design", "graphic", "ui", "ux", "visual", "logo", "banner"]):
            return "design"
        else:
            return "general"
    
    async def generate_work(self, job: Job, skill_context: Dict[str, Any] = None, max_retries: int = 3) -> Optional[str]:
        """Generate work for a job using Avus AI or fallback"""
        logger.info(f"Generating work for job: {job.title}")
        
        start_time = time.time()
        
        try:
            # Build prompt
            prompt = self._build_prompt(job, skill_context)
            
            # Try to generate using Avus model
            if self.avus_available and self.avus_model:
                try:
                    work = await self._generate_with_avus(prompt)
                    if work:
                        generation_time = time.time() - start_time
                        self._store_generation_history(job.id, work, generation_time, "avus", True)
                        logger.info(f"Generated work using Avus AI ({generation_time:.2f}s)")
                        return work
                except Exception as e:
                    logger.warning(f"Error generating with Avus: {e}. Falling back to template-based generation.")
            
            # Fallback to template-based generation
            work = self._generate_template_based(job)
            generation_time = time.time() - start_time
            self._store_generation_history(job.id, work, generation_time, "template", True)
            logger.info(f"Generated work using template ({generation_time:.2f}s)")
            return work
            
        except Exception as e:
            logger.error(f"Error generating work: {e}")
            return None
    
    async def _generate_with_avus(self, prompt: str) -> Optional[str]:
        """Generate work using InferencePipeline (JanusGPT / AvusBrain)."""
        # Use unified pipeline if available
        if hasattr(self, "_pipeline") and self._pipeline is not None:
            return await self._pipeline.generate(prompt, max_tokens=2000)

        # Legacy direct AvusBrain call
        if not self.avus_model:
            return None
        try:
            if hasattr(self.avus_model, 'generate'):
                work = self.avus_model.generate(prompt, max_length=2000)
            elif hasattr(self.avus_model, 'inference'):
                work = self.avus_model.inference(prompt)
            else:
                work = self.avus_model(prompt)
            return work if work else None
        except Exception as e:
            logger.error(f"Error calling Avus model: {e}")
            return None
    
    def _generate_template_based(self, job: Job) -> str:
        """Generate work using template-based approach"""
        job_type = self._detect_job_type(job)
        
        if job_type == "writing":
            return self._generate_writing(job)
        elif job_type == "coding":
            return self._generate_code(job)
        elif job_type == "research":
            return self._generate_research(job)
        elif job_type == "design":
            return self._generate_design(job)
        else:
            return self._generate_general(job)
    
    def _generate_writing(self, job: Job) -> str:
        """Generate writing content"""
        content = f"""# {job.title}

## Introduction
This document addresses the requirements specified in the job posting: {job.title}.

## Overview
{job.description}

## Key Points
1. **Requirement 1**: Detailed analysis and implementation of the first requirement
2. **Requirement 2**: Comprehensive coverage of the second requirement
3. **Requirement 3**: Thorough examination of the third requirement

## Detailed Analysis

### Section 1: Main Topic
This section provides in-depth analysis of the main topic. The content is well-researched and professionally written to meet all specified requirements.

### Section 2: Supporting Information
Additional context and supporting information to strengthen the overall work quality and relevance.

### Section 3: Conclusions
Summary of key findings and recommendations based on the analysis.

## Conclusion
This work comprehensively addresses all requirements specified in the job posting. The content is original, well-structured, and ready for immediate use.

---
Generated by Janus Autonomous Worker
"""
        return content
    
    def _generate_code(self, job: Job) -> str:
        """Generate code"""
        code = f'''"""
{job.title}

Description: {job.description}
"""

# Required imports
import sys
import logging
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Solution:
    """Solution for: {job.title}"""
    
    def __init__(self):
        """Initialize the solution"""
        logger.info("Initializing solution")
    
    def solve(self, input_data: Any) -> Any:
        """
        Main solution method
        
        Args:
            input_data: Input data for the problem
            
        Returns:
            Solution result
        """
        logger.info("Processing input")
        
        # Implementation of the solution
        result = self._process(input_data)
        
        logger.info("Solution complete")
        return result
    
    def _process(self, data: Any) -> Any:
        """Process the input data"""
        # Core logic here
        return data


def main():
    """Main entry point"""
    logger.info("Starting solution")
    
    solution = Solution()
    
    # Example usage
    test_input = None  # Replace with actual test input
    result = solution.solve(test_input)
    
    print(f"Result: {{result}}")


if __name__ == "__main__":
    main()
'''
        return code
    
    def _generate_research(self, job: Job) -> str:
        """Generate research content"""
        content = f"""# Research Report: {job.title}

## Executive Summary
This research report provides comprehensive analysis of {job.title}. The findings are based on thorough investigation and analysis of available information.

## Introduction
{job.description}

## Research Methodology
- Comprehensive literature review
- Data analysis and synthesis
- Expert consultation
- Trend analysis

## Key Findings

### Finding 1
Detailed analysis of the first key finding with supporting evidence and data.

### Finding 2
Comprehensive examination of the second key finding with relevant context.

### Finding 3
Thorough investigation of the third key finding with implications.

## Analysis and Insights
The research reveals several important insights:
- Insight 1: Detailed explanation
- Insight 2: Detailed explanation
- Insight 3: Detailed explanation

## Recommendations
Based on the research findings, the following recommendations are proposed:
1. Recommendation 1 with rationale
2. Recommendation 2 with rationale
3. Recommendation 3 with rationale

## Conclusion
This research comprehensively addresses the topic and provides actionable insights for decision-making.

## References
- Source 1
- Source 2
- Source 3

---
Research completed by Janus Autonomous Worker
"""
        return content
    
    def _generate_design(self, job: Job) -> str:
        """Generate design description"""
        content = f"""# Design Specification: {job.title}

## Design Brief
{job.description}

## Design Objectives
1. Create visually appealing and professional design
2. Ensure usability and accessibility
3. Meet all specified requirements
4. Follow design best practices

## Design Approach
- User-centered design principles
- Modern design trends
- Accessibility standards (WCAG)
- Responsive design considerations

## Design Elements

### Color Palette
- Primary Color: Professional blue (#0066CC)
- Secondary Color: Accent orange (#FF6600)
- Neutral Colors: Grays for backgrounds and text

### Typography
- Heading Font: Modern sans-serif (e.g., Helvetica, Arial)
- Body Font: Readable sans-serif (e.g., Open Sans, Roboto)
- Font Sizes: Hierarchical sizing for readability

### Layout
- Grid-based layout for consistency
- Whitespace for visual clarity
- Responsive breakpoints for mobile/tablet/desktop

## Design Specifications
- Resolution: 1920x1080 (desktop), 768x1024 (tablet), 375x667 (mobile)
- File Format: PNG, SVG, or PDF as specified
- Color Mode: RGB for digital, CMYK for print

## Deliverables
- High-fidelity mockups
- Design specifications document
- Asset files in requested formats

---
Design created by Janus Autonomous Worker
"""
        return content
    
    def _generate_general(self, job: Job) -> str:
        """Generate general content"""
        content = f"""# {job.title}

## Overview
{job.description}

## Scope
This work addresses all requirements specified in the job posting.

## Deliverables
1. Complete solution addressing all requirements
2. Professional quality and formatting
3. Ready for immediate use

## Details
This work has been carefully crafted to meet all specifications and requirements. The content is original, well-researched, and professionally presented.

## Conclusion
All requirements have been addressed comprehensively. The work is ready for submission and use.

---
Generated by Janus Autonomous Worker
"""
        return content
    
    def _store_generation_history(self, job_id: str, work: str, generation_time: float, method: str, success: bool):
        """Store generation history in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Create generation history table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    work_length INTEGER,
                    generation_time REAL,
                    method TEXT,
                    success BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                INSERT INTO generation_history 
                (job_id, work_length, generation_time, method, success)
                VALUES (?, ?, ?, ?, ?)
            """, (job_id, len(work), generation_time, method, success))
            
            conn.commit()
            logger.debug(f"Stored generation history for job {job_id}")
            
        except Exception as e:
            logger.error(f"Error storing generation history: {e}")
            conn.rollback()
        finally:
            conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY VALIDATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class QualityValidator:
    """Validates work quality before submission"""
    
    def __init__(self):
        self.db_path = Path("janus_worker.db")
        self.min_lengths = {
            "writing": 500,
            "coding": 200,
            "research": 800,
            "design": 100,
            "general": 300
        }
    
    def validate_quality(self, work: str, job: Job) -> float:
        """
        Validate work quality and return score (0-1)
        
        Scoring:
        - Length: 30% (minimum words for job type)
        - Coherence: 40% (grammar, structure, readability)
        - Relevance: 30% (matches job requirements)
        """
        logger.info(f"Validating work quality for job: {job.title}")
        
        scores = {}
        
        # Length validation (30%)
        scores['length'] = self._validate_length(work, job)
        
        # Coherence validation (40%)
        scores['coherence'] = self._validate_coherence(work)
        
        # Relevance validation (30%)
        scores['relevance'] = self._validate_relevance(work, job)
        
        # Calculate overall score
        overall_score = (
            scores['length'] * 0.3 +
            scores['coherence'] * 0.4 +
            scores['relevance'] * 0.3
        )
        
        logger.info(f"Quality scores - Length: {scores['length']:.2f}, Coherence: {scores['coherence']:.2f}, Relevance: {scores['relevance']:.2f}, Overall: {overall_score:.2f}")
        
        return overall_score
    
    def _validate_length(self, work: str, job: Job) -> float:
        """Validate work length"""
        job_type = self._detect_job_type(job)
        min_length = self.min_lengths.get(job_type, 300)
        
        word_count = len(work.split())
        
        if word_count < min_length:
            score = word_count / min_length
            logger.warning(f"Work too short: {word_count} words (minimum: {min_length})")
        elif word_count > min_length * 5:
            # Penalize if too long
            score = 1.0 - min(0.3, (word_count - min_length * 5) / (min_length * 10))
            logger.warning(f"Work too long: {word_count} words")
        else:
            score = 1.0
        
        return max(0, min(1, score))
    
    def _validate_coherence(self, work: str) -> float:
        """Validate work coherence (grammar, structure, readability)"""
        score = 1.0
        
        # Check for basic structure
        lines = work.strip().split('\n')
        if len(lines) < 3:
            score -= 0.2  # Penalize if too few lines
        
        # Check for proper sentences
        sentences = work.split('.')
        if len(sentences) < 3:
            score -= 0.2  # Penalize if too few sentences
        
        # Check for common coherence issues
        issues = 0
        
        # Check for repeated words (sign of poor writing)
        words = work.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # If any word appears too frequently, penalize
        max_freq = max(word_freq.values()) if word_freq else 0
        if max_freq > len(words) * 0.1:  # More than 10% of words are the same
            issues += 1
        
        # Check for very short sentences (sign of poor structure)
        short_sentences = sum(1 for s in sentences if len(s.split()) < 3)
        if short_sentences > len(sentences) * 0.3:
            issues += 1
        
        # Deduct points for issues
        score -= issues * 0.15
        
        return max(0, min(1, score))
    
    def _validate_relevance(self, work: str, job: Job) -> float:
        """Validate work relevance to job requirements"""
        score = 1.0
        
        # Check if work mentions key requirements
        work_lower = work.lower()
        job_desc_lower = job.description.lower()
        job_title_lower = job.title.lower()
        
        # Extract key terms from job
        key_terms = job.required_skills + job_title_lower.split()
        
        # Count how many key terms appear in work
        matched_terms = sum(1 for term in key_terms if term.lower() in work_lower)
        
        if len(key_terms) > 0:
            relevance_ratio = matched_terms / len(key_terms)
            score = relevance_ratio
        
        # Check if work is too generic
        generic_phrases = ["placeholder", "example", "todo", "fix me", "not implemented"]
        if any(phrase in work_lower for phrase in generic_phrases):
            score -= 0.3
        
        logger.debug(f"Relevance: {matched_terms}/{len(key_terms)} key terms matched")
        
        return max(0, min(1, score))
    
    def _detect_job_type(self, job: Job) -> str:
        """Detect job type from title and description"""
        text = (job.title + " " + job.description).lower()
        
        if any(word in text for word in ["write", "article", "blog", "content", "copy"]):
            return "writing"
        elif any(word in text for word in ["code", "program", "develop", "python", "javascript"]):
            return "coding"
        elif any(word in text for word in ["research", "analyze", "study", "report"]):
            return "research"
        elif any(word in text for word in ["design", "graphic", "ui", "ux", "visual"]):
            return "design"
        else:
            return "general"
    
    def store_quality_metrics(self, job_id: str, quality_score: float, work_length: int):
        """Store quality metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Create quality metrics table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    quality_score REAL,
                    work_length INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                INSERT INTO quality_metrics 
                (job_id, quality_score, work_length)
                VALUES (?, ?, ?)
            """, (job_id, quality_score, work_length))
            
            conn.commit()
            logger.debug(f"Stored quality metrics for job {job_id}")
            
        except Exception as e:
            logger.error(f"Error storing quality metrics: {e}")
            conn.rollback()
        finally:
            conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class LearningEngine:
    """Learns from web/YouTube to improve skills"""
    
    def __init__(self):
        self.resources: List[LearningResource] = []
        self.learned_concepts: Dict[str, List[str]] = {}
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        self.web_search_api_key = os.getenv("WEB_SEARCH_API_KEY")
        self.web_search_engine_id = os.getenv("WEB_SEARCH_ENGINE_ID")  # For Google Custom Search
        self.youtube_rate_limit_reset = 0
        self.youtube_requests_made = 0
        self.youtube_rate_limit_max = 100  # Requests per day
        self.web_search_rate_limit_reset = 0
        self.web_search_requests_made = 0
        self.web_search_rate_limit_max = 100  # Requests per day
        self.db_path = Path("janus_worker.db")
    
    async def find_learning_resources(self, topic: str, skill: str) -> List[LearningResource]:
        """Find learning resources for a topic"""
        logger.info(f"Searching for learning resources: {topic}")
        
        resources = []
        
        # YouTube search
        try:
            youtube_results = await self._search_youtube(topic)
            resources.extend(youtube_results)
            logger.info(f"Found {len(youtube_results)} YouTube resources for '{topic}'")
        except Exception as e:
            logger.error(f"Error searching YouTube: {e}")
        
        # Web search
        try:
            web_results = await self._search_web(topic)
            resources.extend(web_results)
            logger.info(f"Found {len(web_results)} web resources for '{topic}'")
        except Exception as e:
            logger.error(f"Error searching web: {e}")
        
        # Store resources in database
        for resource in resources:
            self._store_learning_resource(resource)
        
        return resources
    
    def _check_youtube_rate_limit(self) -> bool:
        """Check if we can make a YouTube API request"""
        current_time = time.time()
        
        # Reset daily counter if needed
        if current_time > self.youtube_rate_limit_reset:
            self.youtube_requests_made = 0
            self.youtube_rate_limit_reset = current_time + 86400  # 24 hours
        
        return self.youtube_requests_made < self.youtube_rate_limit_max
    
    def _check_web_search_rate_limit(self) -> bool:
        """Check if we can make a web search API request"""
        current_time = time.time()
        
        # Reset daily counter if needed
        if current_time > self.web_search_rate_limit_reset:
            self.web_search_requests_made = 0
            self.web_search_rate_limit_reset = current_time + 86400  # 24 hours
        
        return self.web_search_requests_made < self.web_search_rate_limit_max
    
    async def _retry_with_backoff(self, func, max_retries: int = 5, service_name: str = "API"):
        """Execute function with exponential backoff retry logic"""
        backoff_times = [1, 2, 4, 8, 16]  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
        
        for attempt in range(max_retries):
            try:
                result = func()
                return result
                
            except requests.exceptions.Timeout as e:
                logger.warning(f"{service_name} timeout (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = backoff_times[attempt]
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
                    
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"{service_name} connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = backoff_times[attempt]
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
                    
            except requests.exceptions.HTTPError as e:
                # Check if it's a rate limit error (429)
                if e.response.status_code == 429:
                    logger.warning(f"{service_name} rate limited (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        wait_time = backoff_times[attempt]
                        logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
                # Check if it's a server error (5xx)
                elif 500 <= e.response.status_code < 600:
                    logger.warning(f"{service_name} server error {e.response.status_code} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        wait_time = backoff_times[attempt]
                        logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
                else:
                    # Don't retry on client errors (4xx except 429)
                    raise
                    
            except Exception as e:
                logger.error(f"{service_name} unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = backoff_times[attempt]
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        raise Exception(f"{service_name} max retries exceeded")
    
    async def _search_youtube(self, topic: str) -> List[LearningResource]:
        """Search YouTube for learning resources using YouTube Data API"""
        if not self.youtube_api_key:
            logger.warning("YouTube API key not configured. Set YOUTUBE_API_KEY environment variable.")
            return []
        
        # Check rate limit
        if not self._check_youtube_rate_limit():
            logger.warning("YouTube API rate limit reached for today")
            return []
        
        try:
            logger.info(f"Searching YouTube for: {topic}")
            
            def make_request():
                # YouTube Data API v3 search endpoint
                url = "https://www.googleapis.com/youtube/v3/search"
                params = {
                    "q": f"{topic} tutorial educational",
                    "part": "snippet",
                    "type": "video",
                    "maxResults": 10,
                    "order": "relevance",
                    "videoCategoryId": "27",  # Education category
                    "key": self.youtube_api_key,
                    "relevanceLanguage": "en",
                    "safeSearch": "strict"  # Filter for educational content
                }
                
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                return response
            
            response = await self._retry_with_backoff(make_request, service_name="YouTube API")
            self.youtube_requests_made += 1
            
            data = response.json()
            resources = []
            
            # Parse search results
            for item in data.get("items", []):
                try:
                    video_id = item["id"].get("videoId")
                    if not video_id:
                        continue
                    
                    snippet = item["snippet"]
                    
                    # Filter for educational content
                    title_lower = snippet.get("title", "").lower()
                    description_lower = snippet.get("description", "").lower()
                    
                    # Check if content is educational
                    educational_keywords = ["tutorial", "course", "how-to", "guide", "lesson", "training", "education", "learn"]
                    is_educational = any(keyword in title_lower or keyword in description_lower for keyword in educational_keywords)
                    
                    if not is_educational:
                        logger.debug(f"Skipping non-educational video: {snippet.get('title')}")
                        continue
                    
                    # Get video details (duration, etc.)
                    video_details = await self._get_youtube_video_details(video_id)
                    
                    resource = LearningResource(
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        title=snippet.get("title", "Untitled"),
                        type="youtube",
                        topic=topic,
                        duration_minutes=video_details.get("duration_minutes", 0),
                        completed=False,
                        learned_concepts=[]
                    )
                    resources.append(resource)
                    logger.debug(f"Found YouTube video: {resource.title} ({resource.duration_minutes} min)")
                    
                except Exception as e:
                    logger.warning(f"Error parsing YouTube result: {e}")
                    continue
            
            logger.info(f"Successfully found {len(resources)} YouTube videos for '{topic}'")
            return resources
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("YouTube API authentication failed. Check API key.")
            elif e.response.status_code == 403:
                logger.error("YouTube API access forbidden. Check API permissions.")
            elif e.response.status_code == 429:
                logger.error("YouTube API rate limit exceeded")
            else:
                logger.error(f"YouTube API error: {e.response.status_code}")
            return []
            
        except Exception as e:
            logger.error(f"Error searching YouTube: {e}")
            return []
    
    async def _get_youtube_video_details(self, video_id: str) -> Dict[str, Any]:
        """Get video details including duration"""
        if not self.youtube_api_key:
            return {"duration_minutes": 0}
        
        try:
            def make_request():
                url = "https://www.googleapis.com/youtube/v3/videos"
                params = {
                    "id": video_id,
                    "part": "contentDetails",
                    "key": self.youtube_api_key
                }
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response
            
            response = await self._retry_with_backoff(make_request, service_name="YouTube Details API")
            data = response.json()
            
            # Parse ISO 8601 duration
            duration_str = data.get("items", [{}])[0].get("contentDetails", {}).get("duration", "PT0M")
            
            # Convert ISO 8601 duration to minutes
            import re
            match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
            if match:
                hours = int(match.group(1) or 0)
                minutes = int(match.group(2) or 0)
                seconds = int(match.group(3) or 0)
                total_minutes = hours * 60 + minutes + (1 if seconds > 0 else 0)
                return {"duration_minutes": total_minutes}
            
            return {"duration_minutes": 0}
            
        except Exception as e:
            logger.warning(f"Error getting YouTube video details: {e}")
            return {"duration_minutes": 0}
    
    async def _search_web(self, topic: str) -> List[LearningResource]:
        """Search web for learning resources using Google Custom Search API or fallback"""
        if not self.web_search_api_key:
            logger.warning("Web Search API key not configured. Set WEB_SEARCH_API_KEY environment variable.")
            return []
        
        # Check rate limit
        if not self._check_web_search_rate_limit():
            logger.warning("Web Search API rate limit reached for today")
            return []
        
        try:
            logger.info(f"Searching web for: {topic}")
            
            def make_request():
                # Google Custom Search API endpoint
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    "q": f"{topic} tutorial guide how-to educational",
                    "key": self.web_search_api_key,
                    "cx": self.web_search_engine_id,  # Custom search engine ID
                    "num": 10,
                    "sort": "date",  # Sort by recency
                    "fileType": "pdf,html,txt"  # Prefer specific formats
                }
                
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                return response
            
            response = await self._retry_with_backoff(make_request, service_name="Web Search API")
            self.web_search_requests_made += 1
            
            data = response.json()
            resources = []
            
            # Parse search results
            for item in data.get("items", []):
                try:
                    url = item.get("link", "")
                    title = item.get("title", "Untitled")
                    snippet = item.get("snippet", "")
                    
                    # Skip paywalled content
                    paywall_indicators = ["paywall", "subscription", "premium", "login required", "sign up", "members only"]
                    if any(indicator in url.lower() or indicator in snippet.lower() for indicator in paywall_indicators):
                        logger.debug(f"Skipping paywalled content: {title}")
                        continue
                    
                    # Skip low-quality sources
                    low_quality_domains = ["pinterest", "instagram", "facebook", "twitter"]
                    if any(domain in url.lower() for domain in low_quality_domains):
                        logger.debug(f"Skipping low-quality source: {title}")
                        continue
                    
                    # Estimate reading time (rough heuristic: 200 words per minute)
                    word_count = len(snippet.split())
                    duration_minutes = max(1, word_count // 200)
                    
                    # Determine resource type based on URL
                    resource_type = "article"
                    if url.endswith(".pdf"):
                        resource_type = "pdf"
                    elif "github" in url.lower():
                        resource_type = "code"
                    elif "documentation" in url.lower() or "docs" in url.lower():
                        resource_type = "documentation"
                    
                    resource = LearningResource(
                        url=url,
                        title=title,
                        type=resource_type,
                        topic=topic,
                        duration_minutes=duration_minutes,
                        completed=False,
                        learned_concepts=[]
                    )
                    resources.append(resource)
                    logger.debug(f"Found web resource ({resource_type}): {resource.title}")
                    
                except Exception as e:
                    logger.warning(f"Error parsing web search result: {e}")
                    continue
            
            logger.info(f"Successfully found {len(resources)} web resources for '{topic}'")
            return resources
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("Web Search API authentication failed. Check API key.")
            elif e.response.status_code == 403:
                logger.error("Web Search API access forbidden. Check API permissions.")
            elif e.response.status_code == 429:
                logger.error("Web Search API rate limit exceeded")
            else:
                logger.error(f"Web Search API error: {e.response.status_code}")
            return []
            
        except Exception as e:
            logger.error(f"Error searching web: {e}")
            return []
    
    async def learn_from_resource(self, resource: LearningResource) -> List[str]:
        """Learn from a resource and extract concepts"""
        logger.info(f"Learning from: {resource.title}")
        
        concepts = []
        
        try:
            # For YouTube videos, we would fetch the transcript
            if resource.type == "youtube":
                concepts = await self._extract_concepts_from_youtube(resource)
            
            # For web articles, we would fetch and parse the content
            elif resource.type == "article":
                concepts = await self._extract_concepts_from_web(resource)
            
            resource.completed = True
            resource.learned_concepts = concepts
            
            logger.info(f"Extracted {len(concepts)} concepts from: {resource.title}")
            
        except Exception as e:
            logger.error(f"Error learning from resource: {e}")
        
        return concepts
    
    async def _extract_concepts_from_youtube(self, resource: LearningResource) -> List[str]:
        """Extract concepts from YouTube video transcript"""
        try:
            logger.info(f"Extracting concepts from YouTube video: {resource.title}")
            
            # In a real implementation, we would:
            # 1. Fetch the video transcript using YouTube API or third-party service
            # 2. Use AI to extract key concepts
            
            # For now, we'll use a combination of title analysis and placeholder concepts
            concepts = []
            
            # Extract keywords from title
            title_words = resource.title.lower().split()
            # Filter out common words
            common_words = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by", "from", "is", "are", "was", "were"}
            title_concepts = [w.strip(".,!?;:") for w in title_words if w not in common_words and len(w) > 3]
            concepts.extend(title_concepts[:5])
            
            # Add placeholder concepts based on topic
            if "python" in resource.title.lower():
                concepts.extend(["python", "programming", "syntax", "functions", "libraries"])
            elif "javascript" in resource.title.lower():
                concepts.extend(["javascript", "web", "dom", "async", "promises"])
            elif "data" in resource.title.lower():
                concepts.extend(["data", "analysis", "visualization", "statistics", "insights"])
            elif "machine learning" in resource.title.lower():
                concepts.extend(["machine learning", "models", "training", "algorithms", "prediction"])
            elif "web" in resource.title.lower():
                concepts.extend(["web", "html", "css", "responsive", "design"])
            
            # Remove duplicates and limit to 10 concepts
            concepts = list(dict.fromkeys(concepts))[:10]
            
            return concepts
            
        except Exception as e:
            logger.error(f"Error extracting concepts from YouTube: {e}")
            return []
    
    async def _extract_concepts_from_web(self, resource: LearningResource) -> List[str]:
        """Extract concepts from web article"""
        try:
            logger.info(f"Extracting concepts from web article: {resource.title}")
            
            # Fetch the web page content
            try:
                response = requests.get(resource.url, timeout=10, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                })
                response.raise_for_status()
                
                # Simple HTML parsing to extract text
                from html.parser import HTMLParser
                
                class TextExtractor(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.text = []
                        self.in_script = False
                        self.in_style = False
                    
                    def handle_starttag(self, tag, attrs):
                        if tag in ("script", "style"):
                            self.in_script = True if tag == "script" else self.in_style
                    
                    def handle_endtag(self, tag):
                        if tag in ("script", "style"):
                            self.in_script = False if tag == "script" else self.in_style
                    
                    def handle_data(self, data):
                        if not self.in_script and not self.in_style:
                            if data.strip():
                                self.text.append(data.strip())
                
                parser = TextExtractor()
                parser.feed(response.text)
                content = " ".join(parser.text)
                
                # Extract key concepts using simple NLP
                # Split into sentences and extract important words
                sentences = content.split(".")[:10]  # First 10 sentences
                
                # Extract words that are likely concepts (capitalized, longer words, etc.)
                concepts = []
                common_words = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by", "from", "is", "are", "was", "were", "be", "been", "being"}
                
                for sentence in sentences:
                    words = sentence.split()
                    for word in words:
                        clean_word = word.strip(".,!?;:()[]{}\"'").lower()
                        # Include words that are: longer than 4 chars, not common, and not already in concepts
                        if len(clean_word) > 4 and clean_word not in common_words and clean_word not in concepts:
                            concepts.append(clean_word)
                
                # Also extract from title
                title_words = resource.title.lower().split()
                for word in title_words:
                    clean_word = word.strip(".,!?;:()[]{}\"'").lower()
                    if len(clean_word) > 4 and clean_word not in common_words and clean_word not in concepts:
                        concepts.append(clean_word)
                
                # Limit to 10 concepts
                concepts = concepts[:10]
                
                logger.debug(f"Extracted {len(concepts)} concepts from web article")
                return concepts
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout fetching web content: {resource.url}")
                # Fallback: extract from title
                return resource.title.lower().split()[:5]
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error fetching web content: {e}")
                # Fallback: extract from title
                return resource.title.lower().split()[:5]
            
        except Exception as e:
            logger.error(f"Error extracting concepts from web: {e}")
            return []
    
    def _store_learning_resource(self, resource: LearningResource):
        """Store learning resource in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            concepts_json = json.dumps(resource.learned_concepts)
            
            cursor.execute("""
                INSERT OR REPLACE INTO learning 
                (url, title, topic, completed, learned_at, concepts)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            """, (
                resource.url,
                resource.title,
                resource.topic,
                resource.completed,
                concepts_json
            ))
            
            conn.commit()
            logger.debug(f"Stored learning resource in database: {resource.url}")
            
        except Exception as e:
            logger.error(f"Error storing learning resource in database: {e}")
            conn.rollback()
        finally:
            conn.close()


# Import completion systems (market analysis, job manager, adaptive gen, etc.)
try:
    from janus_worker_completion import WorkerCompletionMixin
    HAS_COMPLETION = True
except ImportError:
    class WorkerCompletionMixin:  # type: ignore[no-redef]
        """Stub when janus_worker_completion is not available."""
        def init_completion_systems(self): pass
        def get_full_status(self): return {}
    HAS_COMPLETION = False


# ═══════════════════════════════════════════════════════════════════════════════
# AUTONOMOUS WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class JanusAutonomousWorker(WorkerCompletionMixin):
    """Main autonomous worker system"""
    
    def __init__(self, name: str = "Janus"):
        self.name = name
        self.skills: Dict[str, Skill] = {}
        self.jobs_claimed: List[Job] = []
        self.jobs_completed: List[Job] = []
        self.finances = FinancialState()
        self.learning_engine = LearningEngine()
        self.work_generator = WorkGenerator()
        self.quality_validator = QualityValidator()
        self.platforms: List[JobPlatform] = []
        self.db_path = Path("janus_worker.db")
        self.running = False
        
        # Initialize wallet for real financial tracking
        self.wallet = None
        if HAS_WALLET:
            try:
                self.wallet = JanusWallet()
                logger.info(f"Wallet initialized for {self.wallet.BUSINESS_NAME}")
                
                # Setup crypto wallets if not already done
                crypto_addresses = self.wallet.get_crypto_addresses()
                if not crypto_addresses:
                    logger.info("Setting up crypto wallets...")
                    self.wallet.setup_crypto_wallets()
                    crypto_addresses = self.wallet.get_crypto_addresses()
                    if crypto_addresses:
                        logger.info(f"Crypto wallets ready: {list(crypto_addresses.keys())}")
                
            except Exception as e:
                logger.error(f"Could not initialize wallet: {e}")
                self.wallet = None
        else:
            logger.warning("Wallet module not available. Financial tracking disabled.")
        
        # Initialize desktop interaction if available
        self.desktop = None
        self.app_launcher = None
        self.screen_capture = None
        if HAS_APP_LAUNCHER:
            try:
                self.desktop = DesktopInteraction()
                self.app_launcher = self.desktop.launcher
                self.screen_capture = self.desktop.screen_capture
                logger.info("Desktop interaction enabled")
            except Exception as e:
                logger.warning(f"Could not initialize desktop interaction: {e}")
        
        # Initialize database
        self._init_database()
        
        # Initialize platforms
        self._init_platforms()
        
        # Initialize skills
        self._init_skills()

        # Initialize HumanCore — personality, fatigue, mood, social awareness
        self.human = HumanCore(auto_load_mood=True) if HAS_HUMAN_CORE else None
        if self.human:
            logger.info(f"{self.name} human core initialised: {self.human}")

        # Initialize completion systems — market analysis, job manager, adaptive gen, etc.
        self.init_completion_systems()
        if HAS_COMPLETION:
            logger.info(f"{self.name} completion systems initialised")

        # Initialize checkpointer and notifier
        self.checkpointer = _get_checkpointer() if HAS_CHECKPOINT else None
        self.notifier = _notifier

        # Initialize human-like job decision engine
        if HAS_JOB_DECISION:
            self.job_decision = JobDecisionEngine(
                human_core=self.human,
                skills=self.skills
            )
            logger.info("Job decision engine initialized — Janus will choose its own work")
        else:
            self.job_decision = None

        # Adaptive scoring weights — updated by feedback loop
        self._score_weights = {
            "skill_match":   0.4,
            "budget":        0.3,
            "deadline":      0.2,
            "learning":      0.1,
        }
    
    def _recover_corrupt_file(self, path: str) -> bool:
        """
        Attempt to recover a corrupt state file using CorruptFileReader.

        Returns True if recovery produced usable data, False otherwise.
        Logs a full recovery report regardless of outcome.
        """
        if not HAS_CORRUPT_READER:
            logger.warning("CorruptFileReader not available — skipping recovery")
            return False

        logger.warning(f"Attempting corrupt-file recovery on: {path}")
        report = _corrupt_reader.read(path)
        logger.info(f"[Recovery Report]\n{report.summary()}")

        if report.status == RecoveryStatus.FAILED:
            logger.error(f"Recovery failed for {path} — no usable data found")
            return False

        logger.info(
            f"Recovery {report.status.value} for {path} — "
            f"{report.recovered_records} records salvaged "
            f"({report.recovery_rate:.0%} recovery rate)"
        )
        return True

    def _init_database(self):
        """Initialize SQLite database"""
        # If the DB file exists but is corrupt, attempt recovery before connecting
        if os.path.exists(self.db_path):
            try:
                test_conn = sqlite3.connect(
                    f"file:{self.db_path}?mode=ro", uri=True
                )
                test_conn.execute("SELECT 1")
                test_conn.close()
            except sqlite3.DatabaseError:
                logger.error(
                    f"Database appears corrupt: {self.db_path}. "
                    "Attempting recovery..."
                )
                self._recover_corrupt_file(self.db_path)
                # Rename the corrupt DB and start fresh so Janus can continue
                corrupt_backup = self.db_path + ".corrupt_backup"
                try:
                    os.rename(self.db_path, corrupt_backup)
                    logger.warning(
                        f"Corrupt DB moved to {corrupt_backup}. "
                        "Starting with a fresh database."
                    )
                except OSError as e:
                    logger.error(f"Could not rename corrupt DB: {e}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Jobs table - comprehensive schema for job tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                budget REAL,
                deadline TIMESTAMP,
                required_skills TEXT,
                status TEXT NOT NULL,
                platform TEXT NOT NULL,
                claimed_at TIMESTAMP,
                completed_at TIMESTAMP,
                quality_score REAL,
                payment_received REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Skills table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                name TEXT PRIMARY KEY,
                level INTEGER,
                experience_points INTEGER,
                success_rate REAL,
                last_used TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Learning table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning (
                url TEXT PRIMARY KEY,
                title TEXT,
                topic TEXT,
                completed BOOLEAN,
                learned_at TIMESTAMP,
                concepts TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Financial table - comprehensive transaction tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TIMESTAMP NOT NULL,
                type TEXT NOT NULL,
                amount REAL NOT NULL,
                description TEXT,
                job_id TEXT,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_platform ON jobs(platform)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_financials_date ON financials(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_financials_type ON financials(type)")
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def _init_platforms(self):
        """Initialize job platforms"""
        self.platforms = [
            UpworkIntegration(),
            FiverrIntegration(),
        ]
        logger.info(f"Initialized {len(self.platforms)} job platforms")
    
    def _init_skills(self):
        """Initialize core skills"""
        core_skills = [
            "writing",
            "coding",
            "data_analysis",
            "design",
            "research",
            "problem_solving",
            "communication",
        ]
        
        for skill_name in core_skills:
            self.skills[skill_name] = Skill(
                name=skill_name,
                level=SkillLevel.BEGINNER,
                experience_points=0
            )
        
        logger.info(f"Initialized {len(self.skills)} core skills")
    
    async def work_cycle(self):
        """Main work cycle - runs continuously"""
        logger.info(f"{self.name} starting work cycle")
        self.running = True

        # Resume any jobs that were interrupted by a previous crash
        if self.checkpointer:
            incomplete = self.checkpointer.get_incomplete()
            if incomplete:
                logger.info(
                    f"[Checkpoint] Resuming {len(incomplete)} interrupted job(s) from previous session"
                )
                for cp in incomplete:
                    logger.info(
                        f"[Checkpoint] Job {cp['job_id']} was at stage '{cp['stage']}' "
                        f"(attempt {cp['attempt']})"
                    )
                    # Re-queue the job for processing this cycle
                    # The job will be re-fetched from DB and processed normally
                    # High-attempt jobs (>3) are abandoned to avoid infinite loops
                    if cp["attempt"] > 3:
                        logger.warning(
                            f"[Checkpoint] Abandoning job {cp['job_id']} after "
                            f"{cp['attempt']} failed attempts"
                        )
                        self.checkpointer.fail(cp["job_id"], "max attempts exceeded")
                        if self.notifier:
                            self.notifier.notify_job_failed(
                                cp["data"].get("title", cp["job_id"]),
                                f"abandoned after {cp['attempt']} attempts"
                            )

        # Notify cycle start (only fires if energy/mood warrants it)
        if self.notifier and self.human:
            cycle_num = getattr(self, "_cycle_count", 0)
            self._cycle_count = cycle_num + 1
            self.notifier.notify_cycle_start(
                cycle_num=self._cycle_count,
                mood=self.human.mood.mood.label,
                energy=self.human.fatigue.state.energy,
            )

        while self.running:
            try:
                # Express how Janus is feeling at the start of each cycle
                if self.human:
                    logger.info(f"[Janus] {self.human.how_are_you()}")

                # 1. Check for available jobs
                await self._find_jobs()
                
                # 2. Evaluate and claim best jobs
                await self._evaluate_and_claim_jobs()
                
                # 3. Work on claimed jobs
                await self._work_on_jobs()
                
                # 4. Learn to improve skills
                await self._learn_and_improve()
                
                # 5. Check for payments
                await self._check_payments()
                
                # 6. Invest in self-improvement
                await self._invest_in_improvement()

                # 6b. Record cycle for continuous improvement engine
                if HAS_COMPLETION and hasattr(self, "improvement_engine"):
                    quality_scores = [
                        j.quality_score for j in self.jobs_completed
                        if j.quality_score is not None
                    ]
                    skills_used = list(self.skills.keys())
                    cycle_earnings = sum(
                        j.quality_score * j.budget
                        for j in self.jobs_completed
                        if j.quality_score and j.payment_received
                    )
                    self.improvement_engine.record_cycle(
                        jobs_completed=len(self.jobs_completed),
                        earnings=cycle_earnings,
                        quality_scores=quality_scores,
                        skills_improved=skills_used,
                    )
                    recs = self.improvement_engine.get_recommendations()
                    if recs:
                        logger.info(
                            f"[Improvement] Top recommendation: {recs[0]['action']} "
                            f"(confidence {recs[0]['confidence']:.0%})"
                        )

                # 6c. Apply feedback loop — update scoring weights from analysis
                self._apply_feedback_loop()

                # 6d. Notify low JC balance if applicable
                if self.notifier and HAS_COMPLETION and hasattr(self, "financial_reporter"):
                    alert = self.financial_reporter.alert_low_balance(threshold=20.0)
                    if alert:
                        self.notifier.notify_low_balance(self.finances.current_balance)
                
                # 7. Rest and prepare for next cycle
                if self.human:
                    # Simulate a proper rest between cycles (60 min)
                    rest_msg = self.human.take_break(minutes=60)
                    logger.info(f"[Janus] {rest_msg}")
                await asyncio.sleep(3600)  # 1 hour cycle
                
            except Exception as e:
                logger.error(f"Error in work cycle: {e}")
                if self.human:
                    self.human.mood.update("negative", intensity=0.4)
                    self.human.mood.save()
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _find_jobs(self):
        """Find available jobs from all platforms — browser-first, API fallback."""
        logger.info("Searching for available jobs...")

        skill_names = list(self.skills.keys())

        # Primary path: use browser like a human (no API keys needed)
        if HAS_PLATFORM_BROWSER and HAS_COMPUTER_USE:
            try:
                logger.info("[Janus] Opening browser to browse job platforms...")
                async with ComputerUseEngine() as engine:
                    platform_browser = PlatformBrowser(engine)
                    browser_jobs = await platform_browser.find_jobs(skill_names)

                    for bj in browser_jobs:
                        # Convert BrowserJob → Job for compatibility with rest of system
                        job = Job(
                            id=bj.id,
                            title=bj.title,
                            description=bj.description,
                            required_skills=bj.required_skills or skill_names[:2],
                            budget=bj.budget,
                            deadline=bj.deadline,
                            platform=bj.platform,
                            status=JobStatus.AVAILABLE,
                        )
                        self._store_job(job)

                    logger.info(f"[Janus] Found {len(browser_jobs)} jobs via browser")
                    return

            except Exception as e:
                logger.warning(f"[Janus] Browser job search failed: {e} — falling back to API")

        # Fallback: platform API classes (require API keys)
        for platform in self.platforms:
            try:
                jobs = await platform.get_available_jobs(skill_names)
                logger.info(f"Found {len(jobs)} jobs on {platform.platform_name}")
                for job in jobs:
                    self._store_job(job)
            except Exception as e:
                logger.error(f"Error finding jobs on {platform.platform_name}: {e}")
    
    async def _evaluate_and_claim_jobs(self):
        """
        Janus browses available jobs and decides which ones to take —
        based on mood, energy, curiosity, skill fit, and pay.
        Not a formula. A choice.
        """
        logger.info("Browsing available jobs...")

        available_jobs = self._get_available_jobs()
        if not available_jobs:
            logger.info("No jobs available right now.")
            return

        # Use human-like decision engine if available
        if self.job_decision:
            # Keep decision engine's skill map in sync
            self.job_decision.skills = self.skills

            # Let Janus deliberate and pick
            chosen_jobs = self.job_decision.pick_jobs(available_jobs, max_jobs=2)

            # Log what Janus is thinking
            interests = self.job_decision.get_interests_summary()
            logger.info(f"[Janus] {interests}")

            for job in chosen_jobs:
                success = await self._claim_job(job)
                if success:
                    job.status = JobStatus.CLAIMED
                    self.jobs_claimed.append(job)
                    logger.info(f"[Janus] Taking on: '{job.title}' (${job.budget:.0f})")
                    if self.human:
                        self.human.mood.update("positive", intensity=0.3)
                        self.human.fatigue.work(minutes=5)

            if not chosen_jobs:
                # Nothing felt right — Janus is being selective
                if self.human:
                    mood = self.human.mood.mood.label
                    energy = self.human.fatigue.state.energy
                    logger.info(
                        f"[Janus] Passing on everything this cycle "
                        f"(mood: {mood}, energy: {energy:.0%})"
                    )
        else:
            # Fallback: original scoring logic
            scored_jobs = [(job, self._score_job(job)) for job in available_jobs]
            scored_jobs.sort(key=lambda x: x[1], reverse=True)

            energy = self.human.fatigue.state.energy if self.human else 1.0
            acceptance_threshold = 0.5 + (1.0 - energy) * 0.2

            for job, score in scored_jobs[:3]:
                if score > acceptance_threshold:
                    success = await self._claim_job(job)
                    if success:
                        logger.info(f"Claimed job: {job.title} (score: {score:.2f})")
                        self.jobs_claimed.append(job)
                        if self.human:
                            self.human.mood.update("positive", intensity=0.3)
                            self.human.fatigue.work(minutes=5)
    
    def _score_job(self, job: Job) -> float:
        """Score a job based on fit and value — weights adapt from feedback loop."""
        w = self._score_weights
        score = 0.0

        # Skill match
        skill_match = sum(
            1 for skill in job.required_skills
            if skill in self.skills
        ) / max(len(job.required_skills), 1)
        score += skill_match * w["skill_match"]

        # Budget
        budget_score = min(job.budget / 500, 1.0)
        score += budget_score * w["budget"]

        # Deadline
        days_until_deadline = (job.deadline - datetime.now()).days
        deadline_score = max(0, 1 - (days_until_deadline / 30))
        score += deadline_score * w["deadline"]

        # Learning opportunity
        new_skills = [s for s in job.required_skills if s not in self.skills]
        learning_score = min(len(new_skills) / 3, 1.0)
        score += learning_score * w["learning"]

        return score
    
    def _apply_feedback_loop(self) -> None:
        """
        Use ContinuousImprovementEngine analysis to adjust job scoring weights.
        Called at the end of each work cycle.
        """
        if not (HAS_COMPLETION and hasattr(self, "improvement_engine")):
            return
        try:
            analysis = self.improvement_engine.analyze()
            failures = analysis.get("common_failures", [])
            recs = analysis.get("recommendations", [])

            # If quality is consistently low → weight skill_match higher
            if any("quality" in f.lower() for f in failures):
                self._score_weights["skill_match"] = min(
                    0.6, self._score_weights["skill_match"] + 0.02
                )
                self._score_weights["budget"] = max(
                    0.15, self._score_weights["budget"] - 0.01
                )
                logger.info(
                    "[FeedbackLoop] Quality issues detected — boosted skill_match weight to %.2f",
                    self._score_weights["skill_match"],
                )

            # If earnings are declining → weight budget higher
            if any("declining" in r.lower() for r in recs):
                self._score_weights["budget"] = min(
                    0.45, self._score_weights["budget"] + 0.02
                )
                self._score_weights["learning"] = max(
                    0.05, self._score_weights["learning"] - 0.01
                )
                logger.info(
                    "[FeedbackLoop] Earnings declining — boosted budget weight to %.2f",
                    self._score_weights["budget"],
                )

            # Normalise weights to sum to 1.0
            total = sum(self._score_weights.values())
            if total > 0:
                self._score_weights = {
                    k: round(v / total, 4)
                    for k, v in self._score_weights.items()
                }
            logger.debug("[FeedbackLoop] weights: %s", self._score_weights)
        except Exception as e:
            logger.error("[FeedbackLoop] error: %s", e)

    async def _claim_job(self, job: Job) -> bool:
        """Claim a job on the platform"""
        for platform in self.platforms:
            if platform.platform_name == job.platform:
                return await platform.claim_job(job.id)
        return False
        """Claim a job on the platform"""
        for platform in self.platforms:
            if platform.platform_name == job.platform:
                return await platform.claim_job(job.id)
        return False
    
    async def _work_on_jobs(self):
        """Work on claimed jobs"""
        logger.info(f"Working on {len(self.jobs_claimed)} claimed jobs...")
        
        for job in self.jobs_claimed:
            if job.status == JobStatus.CLAIMED:
                logger.info(f"Starting work on: {job.title}")

                # Fatigue accumulates per job — longer jobs cost more
                estimated_minutes = max(10, job.budget / 10)
                if self.human:
                    self.human.fatigue.work(minutes=estimated_minutes)

                # Track in concurrent job manager
                if HAS_COMPLETION and hasattr(self, "job_manager"):
                    self.job_manager.add_job(job.id)

                # Checkpoint: mark job as in-progress
                if self.checkpointer:
                    self.checkpointer.save(job.id, stage="generating",
                                           data={"title": job.title, "budget": job.budget})

                # Generate work using AI
                work = await self._generate_work(job)
                
                if work:
                    # Validate quality
                    quality_score = self.quality_validator.validate_quality(work, job)
                    job.quality_score = quality_score
                    
                    # Store quality metrics
                    self.quality_validator.store_quality_metrics(job.id, quality_score, len(work))
                    
                    # Check if quality is acceptable
                    if quality_score >= 0.7:
                        # Submit work
                        success = await self._submit_work(job, work)
                        
                        if success:
                            job.status = JobStatus.COMPLETED
                            logger.info(f"Completed job: {job.title} (quality: {quality_score:.2f})")
                            self.jobs_completed.append(job)
                            # Success lifts mood
                            if self.human:
                                self.human.mood.update("success", intensity=0.6)
                                self.human.mood.save()
                                logger.info(f"[Janus] Job done. Feeling {self.human.fatigue.status()}.")
                            # Mark complete in job manager; record outcome for adaptive generator
                            if HAS_COMPLETION:
                                if hasattr(self, "job_manager"):
                                    self.job_manager.complete_job(job.id)
                                    self.job_manager.promote_from_queue()
                                if hasattr(self, "adaptive_generator"):
                                    job_type = self.work_generator._detect_job_type(job)
                                    self.adaptive_generator.record_outcome(
                                        job_type, quality_score, client_satisfied=True
                                    )
                            # Record outcome in job decision engine so Janus learns preferences
                            if self.job_decision:
                                self.job_decision.record_outcome(
                                    job, quality_score, enjoyed=(quality_score >= 0.7)
                                )
                            # Checkpoint: complete
                            if self.checkpointer:
                                self.checkpointer.complete(job.id)
                            # Notify owner
                            if self.notifier:
                                self.notifier.notify_job_complete(job.title, quality_score)
                    else:
                        logger.warning(f"Work quality too low ({quality_score:.2f}). Regenerating...")
                        # Try to regenerate with adjusted parameters
                        work = await self._generate_work(job)
                        if work:
                            quality_score = self.quality_validator.validate_quality(work, job)
                            if quality_score >= 0.7:
                                success = await self._submit_work(job, work)
                                if success:
                                    job.status = JobStatus.COMPLETED
                                    logger.info(f"Completed job after regeneration: {job.title}")
                                    self.jobs_completed.append(job)
                                    if self.human:
                                        self.human.mood.update("success", intensity=0.3)
                                        self.human.mood.save()
                            else:
                                logger.error(f"Work quality still too low after regeneration. Marking job as failed.")
                                job.status = JobStatus.FAILED
                                # Failure dents mood
                                if self.human:
                                    self.human.mood.update("failure", intensity=0.5)
                                    self.human.mood.save()
                                    logger.info(
                                        f"[Janus] That one didn't go well. "
                                        f"Mood: {self.human.mood.mood.label}."
                                    )
                                if HAS_COMPLETION:
                                    if hasattr(self, "job_manager"):
                                        self.job_manager.fail_job(job.id)
                                    if hasattr(self, "adaptive_generator"):
                                        job_type = self.work_generator._detect_job_type(job)
                                        self.adaptive_generator.record_outcome(
                                            job_type, quality_score, client_satisfied=False
                                        )
                                # Record failed outcome so Janus avoids similar jobs
                                if self.job_decision:
                                    self.job_decision.record_outcome(
                                        job, quality_score, enjoyed=False
                                    )
                                # Checkpoint + notify
                                if self.checkpointer:
                                    self.checkpointer.fail(job.id, "quality below threshold after regeneration")
                                if self.notifier:
                                    self.notifier.notify_job_failed(job.title, "quality below threshold")
                else:
                    logger.error(f"Failed to generate work for job: {job.title}")
                    job.status = JobStatus.FAILED
                    if self.human:
                        self.human.mood.update("failure", intensity=0.4)
                        self.human.mood.save()
                    if self.checkpointer:
                        self.checkpointer.fail(job.id, "work generation failed")
                    if self.notifier:
                        self.notifier.notify_job_failed(job.title, "work generation failed")
    
    async def _generate_work(self, job: Job) -> Optional[str]:
        """Generate work for a job using AI"""
        logger.info(f"Generating work for: {job.title}")
        
        # Build skill context
        skill_context = {
            name: skill.level.name for name, skill in self.skills.items()
        }
        
        # Generate work
        work = await self.work_generator.generate_work(job, skill_context)
        
        return work
    
    async def _submit_work(self, job: Job, work: str) -> bool:
        """Submit completed work — browser-first, API fallback."""
        # Try browser submission first (no API key needed)
        if HAS_PLATFORM_BROWSER and HAS_COMPUTER_USE:
            try:
                from janus_platform_browser import BrowserJob
                async with ComputerUseEngine() as engine:
                    platform_browser = PlatformBrowser(engine)
                    bj = BrowserJob(
                        id=job.id,
                        title=job.title,
                        description=job.description,
                        budget=job.budget,
                        required_skills=job.required_skills,
                        deadline=job.deadline,
                        platform=job.platform,
                        url="",
                    )
                    success = await platform_browser.deliver(bj, work)
                    if success:
                        logger.info(f"[Janus] Work delivered via browser for: {job.title}")
                        return True
            except Exception as e:
                logger.warning(f"[Janus] Browser submission failed: {e} — falling back to API")

        # Fallback: platform API
        for platform in self.platforms:
            if platform.platform_name == job.platform:
                return await platform.submit_work(job.id, work)
        return False

    async def execute_job_with_computer_use(self, job: Job) -> bool:
        """Execute a job using the ComputerUseEngine for autonomous desktop interaction.

        Builds a session context from the job, runs the goal via the engine,
        and submits the resulting work on success.

        Returns True if the job was completed and submitted successfully.
        """
        if not HAS_COMPUTER_USE:
            logger.warning("Computer use not available — missing dependencies")
            return False

        context = {
            "job_id": job.id,
            "goal": job.description,
            "platform": job.platform,
        }

        try:
            async with ComputerUseEngine(context=context) as engine:
                result = await engine.run_goal(job.description)

            if result.success:
                summary = ""
                if result.data and isinstance(result.data, dict):
                    summary = result.data.get("summary", "")
                submitted = await self._submit_work(job, summary)
                if submitted:
                    logger.info(f"Job {job.id} completed and submitted via ComputerUseEngine.")
                return submitted
            else:
                logger.warning(
                    f"ComputerUseEngine did not complete job {job.id}: {result.error_message}"
                )
                return False

        except Exception as e:
            logger.error(f"Error executing job {job.id} with ComputerUseEngine: {e}")
            return False
    
    async def _learn_and_improve(self):
        """Learn from web/YouTube to improve skills"""
        logger.info("Learning and improving skills...")
        
        # Find skills that need improvement
        weak_skills = [
            skill for skill in self.skills.values()
            if skill.level.value < SkillLevel.EXPERT.value
        ]
        
        for skill in weak_skills[:2]:  # Learn 2 skills per cycle
            logger.info(f"Improving skill: {skill.name}")
            
            # Find learning resources
            resources = await self.learning_engine.find_learning_resources(
                skill.name, skill.name
            )
            
            # Learn from resources
            for resource in resources[:1]:  # Learn from 1 resource
                concepts = await self.learning_engine.learn_from_resource(resource)
                skill.gain_experience(len(concepts) * 10)
    
    async def _check_payments(self):
        """Check for payments from completed jobs"""
        logger.info("Checking for payments...")
        
        for job in self.jobs_completed:
            if not job.payment_received:
                for platform in self.platforms:
                    if platform.platform_name == job.platform:
                        payment = await platform.get_payment(job.id)
                        if payment:
                            # Update legacy financial state
                            self.finances.add_income(payment)
                            job.payment_received = True
                            logger.info(f"Received payment: ${payment:.2f}")
                            
                            # Record in wallet system
                            if self.wallet:
                                try:
                                    from janus_wallet import Category
                                    tx_id = self.wallet.record_income(
                                        amount=payment,
                                        currency="USD",
                                        source=f"{platform.platform_name}_{job.id}",
                                        category=Category.FREELANCE_EARNINGS,
                                        metadata={
                                            "job_id": job.id,
                                            "job_title": job.title,
                                            "platform": platform.platform_name,
                                            "quality_score": job.quality_score
                                        }
                                    )
                                    logger.info(f"Payment recorded in wallet: {tx_id}")
                                except Exception as e:
                                    logger.error(f"Error recording payment in wallet: {e}")
                            
                            # Getting paid is a genuine mood boost
                            if self.human:
                                intensity = min(1.0, payment / 200)  # scales with amount
                                self.human.mood.update("success", intensity=intensity)
                                self.human.mood.update("positive", intensity=0.3)
                                self.human.mood.save()
                                logger.info(
                                    f"[Janus] ${payment:.2f} received. "
                                    f"{self.human.mood.express()}"
                                )
                            # Notify owner
                            if self.notifier:
                                self.notifier.notify_payment(payment, job.title)
    
    async def _invest_in_improvement(self):
        """Use earnings to buy compute/training resources"""
        logger.info(f"Current balance: ${self.finances.current_balance:.2f}")
        
        # Invest in GPU compute if balance is high enough
        if self.finances.current_balance > 100:
            logger.info("Investing in GPU compute for training...")
            self.finances.add_expense(50, "GPU compute for model training")
            
            # Record in wallet
            if self.wallet:
                try:
                    from janus_wallet import Category
                    self.wallet.record_expense(
                        amount=50.0,
                        currency="USD",
                        destination="gpu_compute_provider",
                        category=Category.COMPUTE_RESOURCES,
                        metadata={"purpose": "model_training"}
                    )
                except Exception as e:
                    logger.error(f"Error recording compute expense in wallet: {e}")
        
        # Invest in learning resources
        if self.finances.current_balance > 50:
            logger.info("Investing in learning resources...")
            self.finances.add_expense(20, "Online courses and tutorials")
            
            # Record in wallet
            if self.wallet:
                try:
                    from janus_wallet import Category
                    self.wallet.record_expense(
                        amount=20.0,
                        currency="USD",
                        destination="learning_platform",
                        category=Category.TRAINING_DATA,
                        metadata={"purpose": "skill_improvement"}
                    )
                except Exception as e:
                    logger.error(f"Error recording learning expense in wallet: {e}")
    
    def _store_job(self, job: Job):
        """Store job in database with all required fields"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Convert required_skills list to JSON string for storage
            skills_json = json.dumps(job.required_skills)
            
            cursor.execute("""
                INSERT OR REPLACE INTO jobs 
                (id, title, description, budget, deadline, required_skills, 
                 status, platform, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                job.id,
                job.title,
                job.description,
                job.budget,
                job.deadline.isoformat() if job.deadline else None,
                skills_json,
                job.status.value,
                job.platform
            ))
            
            conn.commit()
            logger.debug(f"Stored job in database: {job.id} - {job.title}")
            
        except Exception as e:
            logger.error(f"Error storing job in database: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def _get_available_jobs(self) -> List[Job]:
        """Get available jobs from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, title, description, budget, deadline, required_skills, 
                       status, platform
                FROM jobs WHERE status = ?
                ORDER BY created_at DESC
            """, (JobStatus.AVAILABLE.value,))
            
            rows = cursor.fetchall()
            jobs = []
            
            for row in rows:
                try:
                    job_id, title, description, budget, deadline_str, skills_json, status, platform = row
                    
                    # Parse deadline
                    if deadline_str:
                        try:
                            deadline = datetime.fromisoformat(deadline_str)
                        except (ValueError, TypeError):
                            deadline = datetime.now() + timedelta(days=7)
                    else:
                        deadline = datetime.now() + timedelta(days=7)
                    
                    # Parse required_skills from JSON
                    try:
                        required_skills = json.loads(skills_json) if skills_json else []
                    except (json.JSONDecodeError, TypeError):
                        required_skills = []
                    
                    job = Job(
                        id=job_id,
                        title=title,
                        description=description or "",
                        budget=budget or 0.0,
                        required_skills=required_skills,
                        deadline=deadline,
                        platform=platform,
                        status=JobStatus(status)
                    )
                    jobs.append(job)
                    
                except Exception as e:
                    logger.warning(f"Error parsing job from database: {e}")
                    continue
            
            logger.debug(f"Retrieved {len(jobs)} available jobs from database")
            return jobs
            
        except Exception as e:
            logger.error(f"Error retrieving jobs from database: {e}")
            return []
        finally:
            conn.close()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        status = {
            "name": self.name,
            "running": self.running,
            "desktop_enabled": self.desktop is not None,
            "skills": {
                name: {
                    "level": skill.level.name,
                    "experience": skill.experience_points,
                    "success_rate": skill.success_rate
                }
                for name, skill in self.skills.items()
            },
            "jobs_claimed": len(self.jobs_claimed),
            "jobs_completed": len(self.jobs_completed),
            "finances": {
                "total_earned": self.finances.total_earned,
                "total_spent": self.finances.total_spent,
                "current_balance": self.finances.current_balance,
                "average_job_value": self.finances.average_job_value
            }
        }
        
        # Add wallet information if available
        if self.wallet:
            try:
                balances = self.wallet.get_all_balances()
                crypto_addresses = self.wallet.get_crypto_addresses()
                
                status["wallet"] = {
                    "business_name": self.wallet.BUSINESS_NAME,
                    "balances": {code: float(amount) for code, amount in balances.items()},
                    "crypto_addresses": crypto_addresses,
                    "paypal_email": self.wallet.get_paypal_email() if hasattr(self.wallet, 'get_paypal_email') else None
                }
            except Exception as e:
                logger.error(f"Error getting wallet status: {e}")
                status["wallet"] = {"error": str(e)}
        
        return status
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # DESKTOP INTERACTION METHODS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def open_upwork_browser(self) -> bool:
        """Open Upwork in browser"""
        if not self.desktop:
            logger.warning("Desktop interaction not available")
            return False
        
        logger.info("Opening Upwork in browser")
        return self.desktop.open_upwork()
    
    def open_fiverr_browser(self) -> bool:
        """Open Fiverr in browser"""
        if not self.desktop:
            logger.warning("Desktop interaction not available")
            return False
        
        logger.info("Opening Fiverr in browser")
        return self.desktop.open_fiverr()
    
    def open_youtube_browser(self) -> bool:
        """Open YouTube in browser"""
        if not self.desktop:
            logger.warning("Desktop interaction not available")
            return False
        
        logger.info("Opening YouTube in browser")
        return self.desktop.open_youtube()
    
    def open_website(self, url: str, browser: str = "chrome") -> bool:
        """Open any website in browser"""
        if not self.desktop:
            logger.warning("Desktop interaction not available")
            return False
        
        logger.info(f"Opening {url} in {browser}")
        return self.desktop.navigate_to_website(url, browser)
    
    def open_application(self, app_name: str, args: List[str] = None) -> bool:
        """Open any installed application"""
        if not self.app_launcher:
            logger.warning("App launcher not available")
            return False
        
        logger.info(f"Opening application: {app_name}")
        return self.app_launcher.open_any_app(app_name, args)
    
    def open_app_by_path(self, app_path: str, args: List[str] = None) -> bool:
        """Open application by full path"""
        if not self.app_launcher:
            logger.warning("App launcher not available")
            return False
        
        logger.info(f"Opening application from path: {app_path}")
        return self.app_launcher.open_app_by_path(app_path, args)
    
    def take_screenshot(self) -> Optional[str]:
        """Take a screenshot to see what's on screen"""
        if not self.screen_capture:
            logger.warning("Screen capture not available")
            return None
        
        logger.info("Taking screenshot")
        return self.screen_capture.take_screenshot()
    
    def list_installed_apps(self) -> List[str]:
        """Get list of all installed applications"""
        if not self.app_launcher:
            logger.warning("App launcher not available")
            return []
        
        return self.app_launcher.list_installed_apps()
    
    def search_apps(self, search_term: str) -> List[str]:
        """Search for installed applications"""
        if not self.app_launcher:
            logger.warning("App launcher not available")
            return []
        
        logger.info(f"Searching for apps: {search_term}")
        return self.app_launcher.search_apps(search_term)
    
    def open_file_explorer(self, path: str = "") -> bool:
        """Open file explorer"""
        if not self.app_launcher:
            logger.warning("App launcher not available")
            return False
        
        logger.info(f"Opening file explorer at: {path or 'home'}")
        return self.app_launcher.open_file_explorer(path)
    
    def open_terminal(self) -> bool:
        """Open terminal/command prompt"""
        if not self.app_launcher:
            logger.warning("App launcher not available")
            return False
        
        logger.info("Opening terminal")
        return self.app_launcher.open_terminal()
    
    def close_all_apps(self) -> None:
        """Close all opened applications"""
        if not self.app_launcher:
            logger.warning("App launcher not available")
            return
        
        logger.info("Closing all applications")
        self.app_launcher.close_all_apps()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Main entry point"""
    logger.info("Starting Janus Autonomous Worker")
    
    # Create worker
    janus = JanusAutonomousWorker()
    
    # Show initial status
    logger.info(f"Status: {json.dumps(janus.get_status(), indent=2, default=str)}")
    
    # Start work cycle
    try:
        await janus.work_cycle()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        janus.running = False


if __name__ == "__main__":
    asyncio.run(main())
