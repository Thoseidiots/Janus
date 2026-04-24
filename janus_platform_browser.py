"""
janus_platform_browser.py
==========================
Platform-specific browser flows for Upwork and Fiverr.

Janus uses the computer like a human — opens a browser, navigates to the
freelance platform, browses jobs, reads descriptions, applies, and submits
work. No API keys needed.

Each platform class knows the exact steps a human takes on that site:
  - Where the search bar is
  - How to filter by skill/category
  - How to read a job listing
  - How to write and submit a proposal
  - How to deliver completed work
  - How to check for new messages/payments
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("janus")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_credentials(platform: str) -> Dict[str, str]:
    """
    Load platform credentials.
    Priority: {platform}.env → Top_Mission.env → Top_Secret.env → env vars
    """
    cred_paths = [
        Path(f"{platform.lower()}.env"),                          # fiverr.env / upwork.env
        Path("Top_Mission.env"),
        Path("Top_Secret.env"),
        Path("Credentials.env"),
        Path("Credintials.env"),
        Path.home() / "Downloads" / "Janus-workspace" / f"{platform.lower()}.env",
        Path.home() / "Downloads" / "Janus-workspace" / "Top_Mission.env",
        Path.home() / "Downloads" / "Janus-workspace" / "Credentials.env",
    ]

    email = ""
    password = ""
    login_method = ""

    for cred_path in cred_paths:
        if cred_path.exists():
            try:
                text = cred_path.read_text(encoding="utf-8")
                for line in text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # Support both KEY=VALUE and KEY: VALUE formats
                    if "=" in line:
                        key, _, val = line.partition("=")
                    elif ":" in line:
                        key, _, val = line.partition(":")
                    else:
                        continue
                    key = key.strip().lower().replace("-", "_").replace(" ", "_")
                    val = val.strip()
                    if key in ("email", "upwork_email", "fiverr_email"):
                        email = val
                    elif key == "password":
                        password = val
                    elif key == "login_method":
                        login_method = val
                if email:
                    logger.debug(f"[Credentials] Loaded from {cred_path}")
                    break
            except Exception as e:
                logger.warning(f"[Credentials] Could not read {cred_path}: {e}")

    prefix = platform.upper()
    return {
        "username":     email or os.getenv(f"{prefix}_USERNAME", ""),
        "password":     password or os.getenv(f"{prefix}_PASSWORD", ""),
        "email":        email or os.getenv(f"{prefix}_EMAIL", ""),
        "login_method": login_method or os.getenv(f"{prefix}_LOGIN_METHOD", ""),
    }


@dataclass
class BrowserJob:
    """A job found by browsing a platform."""
    id: str
    title: str
    description: str
    budget: float
    required_skills: List[str]
    deadline: datetime
    platform: str
    url: str
    client_name: str = ""
    proposals_count: int = 0
    raw_text: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# UPWORK BROWSER FLOW
# ─────────────────────────────────────────────────────────────────────────────

class UpworkBrowser:
    """
    Drives Upwork like a human freelancer would.

    Flow:
      1. Open upwork.com
      2. Log in (if not already)
      3. Search for jobs matching skills
      4. Read each listing — extract title, budget, description
      5. Apply with a tailored proposal
      6. Check messages for responses
      7. Submit completed work via the contract page
    """

    BASE_URL = "https://www.upwork.com"
    SEARCH_URL = "https://www.upwork.com/nx/search/jobs"

    def __init__(self, engine: Any):
        """
        engine: ComputerUseEngine instance (already entered as context manager)
        """
        self._engine = engine
        self._browser = None
        self._logged_in = False

        # Import BrowserComputerUse lazily to avoid circular imports
        try:
            from janus_computer_use import BrowserComputerUse
            self._browser = BrowserComputerUse(engine, browser="chrome")
        except ImportError:
            logger.error("janus_computer_use not available")

    async def _ensure_logged_in(self) -> bool:
        """Log in to Upwork if not already logged in."""
        if self._logged_in:
            return True

        creds = _load_credentials("upwork")
        if not creds["email"] and not creds["username"]:
            logger.warning("[Upwork] No credentials found. Set UPWORK_EMAIL and UPWORK_PASSWORD env vars.")
            return False

        logger.info("[Upwork] Navigating to login page...")
        result = await self._browser.open(f"{self.BASE_URL}/ab/account-security/login")
        if not result.success:
            logger.error(f"[Upwork] Failed to open login page: {result.error_message}")
            return False

        await asyncio.sleep(2)  # wait for page to render

        username = creds["email"] or creds["username"]
        result = await self._browser.login(username, creds["password"])

        if result.success:
            self._logged_in = True
            logger.info("[Upwork] Logged in successfully")
            await asyncio.sleep(2)
        else:
            logger.error(f"[Upwork] Login failed: {result.error_message}")

        return self._logged_in

    async def search_jobs(self, skills: List[str], max_results: int = 10) -> List[BrowserJob]:
        """Search Upwork for jobs matching the given skills."""
        if not self._browser:
            return []

        await self._ensure_logged_in()

        query = " ".join(skills[:3])  # use top 3 skills as search query
        search_url = f"{self.SEARCH_URL}/?q={query.replace(' ', '%20')}&sort=recency"

        logger.info(f"[Upwork] Searching for: {query}")
        result = await self._browser.open(search_url)
        if not result.success:
            logger.error(f"[Upwork] Search failed: {result.error_message}")
            return []

        await asyncio.sleep(3)  # wait for results to load

        # Use the AI planner to extract job listings from the page
        jobs = await self._extract_job_listings("upwork", max_results)
        logger.info(f"[Upwork] Found {len(jobs)} jobs")
        return jobs

    async def apply_to_job(self, job: BrowserJob, proposal: str) -> bool:
        """Apply to a job with a proposal."""
        if not self._browser:
            return False

        logger.info(f"[Upwork] Applying to: {job.title}")

        # Open the job page
        result = await self._browser.open(job.url)
        if not result.success:
            logger.error(f"[Upwork] Failed to open job page: {result.error_message}")
            return False

        await asyncio.sleep(2)

        # Click "Apply Now" button
        screenshot = await self._engine.screen.capture()
        apply_btn = None
        for label in ("Apply Now", "Submit a Proposal", "Apply"):
            elements = await self._engine.vision.find_element(label, screenshot)
            if elements:
                apply_btn = elements[0]
                break

        if not apply_btn:
            # Try using the AI planner to find and click apply
            result = await self._engine.run_goal(
                f"Click the Apply Now or Submit Proposal button on this Upwork job page"
            )
            if not result.success:
                logger.error("[Upwork] Could not find Apply button")
                return False
        else:
            await self._engine.mouse.click(*apply_btn.center)
            await asyncio.sleep(2)

        # Fill in the cover letter / proposal
        result = await self._browser.apply_to_job(job.url, proposal)
        if result.success:
            logger.info(f"[Upwork] Applied to: {job.title}")
        else:
            logger.error(f"[Upwork] Application failed: {result.error_message}")

        return result.success

    async def submit_work(self, contract_id: str, work_content: str) -> bool:
        """Submit completed work on a contract."""
        if not self._browser:
            return False

        contract_url = f"{self.BASE_URL}/contracts/{contract_id}"
        logger.info(f"[Upwork] Submitting work for contract: {contract_id}")

        result = await self._browser.submit_work(contract_url, work_content)
        if result.success:
            logger.info("[Upwork] Work submitted successfully")
        else:
            logger.error(f"[Upwork] Work submission failed: {result.error_message}")

        return result.success

    async def check_messages(self) -> List[Dict[str, str]]:
        """Check Upwork inbox for new messages."""
        if not self._browser:
            return []

        await self._ensure_logged_in()

        result = await self._browser.open(f"{self.BASE_URL}/ab/messages")
        if not result.success:
            return []

        await asyncio.sleep(2)

        # Use AI planner to read messages
        result = await self._engine.run_goal(
            "Read the list of messages in the Upwork inbox and return the sender names and preview text"
        )

        messages = []
        if result.success and result.data:
            # Parse whatever the planner extracted
            if isinstance(result.data, list):
                messages = result.data
            elif isinstance(result.data, dict):
                messages = result.data.get("messages", [])

        return messages

    async def _extract_job_listings(self, platform: str, max_results: int) -> List[BrowserJob]:
        """Use AI planner + OCR to extract job listings from the current page."""
        result = await self._engine.run_goal(
            f"Extract all job listings visible on this {platform} search results page. "
            f"For each job, get: title, budget/rate, required skills, and the job URL. "
            f"Return as structured data."
        )

        jobs = []
        if result.success and result.data:
            raw_jobs = []
            if isinstance(result.data, list):
                raw_jobs = result.data
            elif isinstance(result.data, dict):
                raw_jobs = result.data.get("jobs", result.data.get("results", []))

            for i, raw in enumerate(raw_jobs[:max_results]):
                if not isinstance(raw, dict):
                    continue
                try:
                    job = BrowserJob(
                        id=f"{platform}_{i}_{datetime.now().timestamp():.0f}",
                        title=raw.get("title", "Untitled"),
                        description=raw.get("description", raw.get("title", "")),
                        budget=float(raw.get("budget", raw.get("rate", 0)) or 0),
                        required_skills=raw.get("skills", []),
                        deadline=datetime.now() + timedelta(days=7),
                        platform=platform,
                        url=raw.get("url", ""),
                        raw_text=str(raw),
                    )
                    jobs.append(job)
                except Exception as e:
                    logger.warning(f"[{platform}] Could not parse job {i}: {e}")

        # Fallback: if AI planner returned nothing useful, do OCR-based extraction
        if not jobs:
            jobs = await self._ocr_extract_jobs(platform, max_results)

        return jobs

    async def _ocr_extract_jobs(self, platform: str, max_results: int) -> List[BrowserJob]:
        """Fallback: extract jobs from page using raw OCR."""
        try:
            screenshot = await self._engine.screen.capture()
            words = await self._engine.screen.ocr(screenshot)
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return []

        # Group words into lines
        lines: Dict[int, List] = {}
        for word in words:
            row = word.bounding_box.y // 25
            lines.setdefault(row, []).append(word.text)

        line_texts = [" ".join(tokens) for tokens in sorted(lines.items())]

        jobs = []
        for i, line in enumerate(line_texts):
            line = line.strip()
            # Heuristic: job titles are usually 5-15 words, mixed case
            words_in_line = line.split()
            if 4 <= len(words_in_line) <= 20 and any(w[0].isupper() for w in words_in_line if w):
                job = BrowserJob(
                    id=f"{platform}_ocr_{i}_{datetime.now().timestamp():.0f}",
                    title=line,
                    description=line,
                    budget=0.0,
                    required_skills=[],
                    deadline=datetime.now() + timedelta(days=7),
                    platform=platform,
                    url="",
                    raw_text=line,
                )
                jobs.append(job)
                if len(jobs) >= max_results:
                    break

        return jobs


# ─────────────────────────────────────────────────────────────────────────────
# FIVERR BROWSER FLOW
# ─────────────────────────────────────────────────────────────────────────────

class FiverrBrowser:
    """
    Drives Fiverr like a human seller would.

    On Fiverr, Janus is a seller (offers gigs) rather than a buyer.
    Flow:
      1. Open fiverr.com
      2. Log in
      3. Check for new orders in the seller dashboard
      4. Read order requirements
      5. Deliver completed work
      6. Check for messages/revision requests
    """

    BASE_URL = "https://www.fiverr.com"

    def __init__(self, engine: Any):
        self._engine = engine
        self._browser = None
        self._logged_in = False

        try:
            from janus_computer_use import BrowserComputerUse
            self._browser = BrowserComputerUse(engine, browser="chrome")
        except ImportError:
            logger.error("janus_computer_use not available")

    async def _ensure_logged_in(self) -> bool:
        if self._logged_in:
            return True

        creds = _load_credentials("fiverr")
        if not creds["email"]:
            logger.warning("[Fiverr] No credentials found. Set FIVERR_EMAIL env var.")
            return False

        result = await self._browser.open(f"{self.BASE_URL}/login")
        if not result.success:
            return False

        await asyncio.sleep(2)

        login_method = creds.get("login_method", "").lower()

        if login_method == "google":
            # Click "Continue with Google" button
            logger.info("[Fiverr] Logging in via Google OAuth...")
            google_btn = await self._engine.vision.find_element("Continue with Google")
            if not google_btn:
                google_btn = await self._engine.vision.find_element("Sign in with Google")
            if google_btn:
                center = self._engine.vision.center_of(google_btn[0])
                await self._engine.mouse.click(*center)
                await asyncio.sleep(3)
                # Google will show email field — type the email
                email_field = await self._engine.vision.find_element("email")
                if email_field:
                    center = self._engine.vision.center_of(email_field[0])
                    await self._engine.mouse.click(*center)
                    await self._engine.keyboard.type_text(creds["email"])
                    await self._engine.keyboard.press_key("Return")
                    await asyncio.sleep(2)
                # Google may ask for password (Gmail password)
                # At this point the user may need to approve on their phone
                # Wait up to 30s for the redirect back to Fiverr
                for _ in range(6):
                    await asyncio.sleep(5)
                    screen = await self._engine.screen.capture()
                    ocr = await self._engine.screen.ocr(screen)
                    text = " ".join(w.text for w in ocr).lower()
                    if "fiverr" in text and ("dashboard" in text or "orders" in text or "gigs" in text):
                        self._logged_in = True
                        logger.info("[Fiverr] Google OAuth login successful")
                        return True
                logger.warning("[Fiverr] Google OAuth may require manual approval")
                self._logged_in = True  # Optimistically assume logged in
                return True
            else:
                logger.error("[Fiverr] Could not find Google login button")
                return False
        else:
            # Standard username/password login
            username = creds["username"] or creds["email"]
            result = await self._browser.login(username, creds.get("password", ""))
            if result.success:
                self._logged_in = True
                logger.info("[Fiverr] Logged in successfully")
                await asyncio.sleep(2)
            else:
                logger.error(f"[Fiverr] Login failed: {result.error_message}")

        return self._logged_in

    async def check_orders(self) -> List[BrowserJob]:
        """Check the seller dashboard for active orders."""
        if not self._browser:
            return []

        await self._ensure_logged_in()

        result = await self._browser.open(f"{self.BASE_URL}/seller_dashboard")
        if not result.success:
            return []

        await asyncio.sleep(2)

        result = await self._engine.run_goal(
            "Find all active orders in the Fiverr seller dashboard. "
            "For each order get: order ID, buyer name, gig title, deadline, and requirements."
        )

        orders = []
        if result.success and result.data:
            raw_orders = result.data if isinstance(result.data, list) else result.data.get("orders", [])
            for i, raw in enumerate(raw_orders):
                if not isinstance(raw, dict):
                    continue
                order = BrowserJob(
                    id=raw.get("order_id", f"fiverr_order_{i}"),
                    title=raw.get("gig_title", raw.get("title", "Fiverr Order")),
                    description=raw.get("requirements", raw.get("description", "")),
                    budget=float(raw.get("price", raw.get("budget", 0)) or 0),
                    required_skills=[],
                    deadline=datetime.now() + timedelta(days=int(raw.get("days_left", 3))),
                    platform="fiverr",
                    url=raw.get("url", ""),
                    client_name=raw.get("buyer_name", ""),
                )
                orders.append(order)

        logger.info(f"[Fiverr] Found {len(orders)} active orders")
        return orders

    async def deliver_order(self, order_id: str, work_content: str, files: List[str] = None) -> bool:
        """Deliver completed work for an order."""
        if not self._browser:
            return False

        order_url = f"{self.BASE_URL}/orders/{order_id}/deliver"
        logger.info(f"[Fiverr] Delivering order: {order_id}")

        result = await self._browser.open(order_url)
        if not result.success:
            return False

        await asyncio.sleep(2)

        # Use AI planner to fill in delivery form
        result = await self._engine.run_goal(
            f"Fill in the Fiverr order delivery form. "
            f"Write this in the delivery message field: {work_content[:500]}. "
            f"Then click the Deliver Now button."
        )

        if result.success:
            logger.info(f"[Fiverr] Order {order_id} delivered")
        else:
            logger.error(f"[Fiverr] Delivery failed: {result.error_message}")

        return result.success

    async def search_buyer_requests(self, skills: List[str]) -> List[BrowserJob]:
        """Browse Fiverr Buyer Requests for matching work."""
        if not self._browser:
            return []

        await self._ensure_logged_in()

        result = await self._browser.open(f"{self.BASE_URL}/seller_dashboard/buyer_requests")
        if not result.success:
            return []

        await asyncio.sleep(2)

        # Filter by relevant skills
        query = " ".join(skills[:2])
        result = await self._engine.run_goal(
            f"Find buyer requests related to: {query}. "
            f"Extract title, budget, description, and deadline for each."
        )

        jobs = []
        if result.success and result.data:
            raw = result.data if isinstance(result.data, list) else result.data.get("requests", [])
            for i, item in enumerate(raw):
                if not isinstance(item, dict):
                    continue
                job = BrowserJob(
                    id=f"fiverr_br_{i}_{datetime.now().timestamp():.0f}",
                    title=item.get("title", "Buyer Request"),
                    description=item.get("description", ""),
                    budget=float(item.get("budget", 0) or 0),
                    required_skills=skills,
                    deadline=datetime.now() + timedelta(days=7),
                    platform="fiverr",
                    url=item.get("url", ""),
                )
                jobs.append(job)

        logger.info(f"[Fiverr] Found {len(jobs)} buyer requests")
        return jobs


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED PLATFORM BROWSER — used by the autonomous worker
# ─────────────────────────────────────────────────────────────────────────────

class PlatformBrowser:
    """
    Single entry point for all platform browser interactions.
    The autonomous worker calls this instead of platform API classes.

    Usage:
        async with ComputerUseEngine() as engine:
            browser = PlatformBrowser(engine)
            jobs = await browser.find_jobs(["python", "ai", "writing"])
            for job in jobs:
                if janus_wants_this_job(job):
                    await browser.apply(job, proposal)
    """

    def __init__(self, engine: Any):
        self._engine = engine
        self.upwork = UpworkBrowser(engine)
        self.fiverr = FiverrBrowser(engine)

    async def find_jobs(self, skills: List[str], platforms: List[str] = None) -> List[BrowserJob]:
        """Find jobs across all platforms."""
        platforms = platforms or ["upwork", "fiverr"]
        all_jobs = []

        if "upwork" in platforms:
            try:
                jobs = await self.upwork.search_jobs(skills)
                all_jobs.extend(jobs)
            except Exception as e:
                logger.error(f"[PlatformBrowser] Upwork search failed: {e}")

        if "fiverr" in platforms:
            try:
                # On Fiverr, check both active orders and buyer requests
                orders = await self.fiverr.check_orders()
                requests = await self.fiverr.search_buyer_requests(skills)
                all_jobs.extend(orders)
                all_jobs.extend(requests)
            except Exception as e:
                logger.error(f"[PlatformBrowser] Fiverr search failed: {e}")

        logger.info(f"[PlatformBrowser] Total jobs found: {len(all_jobs)}")
        return all_jobs

    async def apply(self, job: BrowserJob, proposal: str) -> bool:
        """Apply to a job on its platform."""
        if job.platform == "upwork":
            return await self.upwork.apply_to_job(job, proposal)
        elif job.platform == "fiverr":
            # On Fiverr, send an offer on the buyer request
            result = await self._engine.run_goal(
                f"Send an offer on this Fiverr buyer request. "
                f"Write this proposal: {proposal[:300]}"
            )
            return result.success
        return False

    async def deliver(self, job: BrowserJob, work_content: str) -> bool:
        """Deliver completed work for a job."""
        if job.platform == "upwork":
            return await self.upwork.submit_work(job.id, work_content)
        elif job.platform == "fiverr":
            return await self.fiverr.deliver_order(job.id, work_content)
        return False

    async def check_all_messages(self) -> Dict[str, List]:
        """Check messages across all platforms."""
        messages = {}

        try:
            messages["upwork"] = await self.upwork.check_messages()
        except Exception as e:
            logger.error(f"[PlatformBrowser] Upwork messages failed: {e}")
            messages["upwork"] = []

        return messages
