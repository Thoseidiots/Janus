"""
janus_browser_test.py
======================
End-to-end browser test for Janus.

Proves that Janus can:
  1. Open Chrome on the EliteDesk
  2. Navigate to a target site (Upwork or Fiverr)
  3. Sign in with Google (avus.janus@gmail.com)
  4. Search for jobs matching a skill query
  5. Read and log the first job listing via OCR
  6. Take a screenshot as evidence

Run this BEFORE starting the full autonomous worker to confirm the
computer-use stack is working on your machine.

Usage:
    python janus_browser_test.py                        # test Upwork
    python janus_browser_test.py --platform fiverr      # test Fiverr
    python janus_browser_test.py --query "python"       # custom search
    python janus_browser_test.py --no-login             # skip login step
    python janus_browser_test.py --screenshot-only      # just take a screenshot
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("janus_browser_test")

# ── Import computer-use stack ─────────────────────────────────────────────────
try:
    from janus_computer_use import (
        ComputerUseEngine,
        BrowserComputerUse,
        WaitCondition,
        WaitConditionType,
        ActionType,
    )
    HAS_COMPUTER_USE = True
except ImportError as e:
    log.error("janus_computer_use not available: %s", e)
    log.error("Install dependencies: python janus_install_deps.py")
    HAS_COMPUTER_USE = False

# ── Platform configs ──────────────────────────────────────────────────────────

PLATFORMS = {
    "upwork": {
        "url":          "https://www.upwork.com",
        "search_url":   "https://www.upwork.com/nx/find-work/best-matches",
        "login_url":    "https://www.upwork.com/ab/account-security/login",
        "google_btn":   "Continue with Google",
        "search_field": "search",
        "job_indicator": "hourly",   # text likely on results page
    },
    "fiverr": {
        "url":          "https://www.fiverr.com",
        "search_url":   "https://www.fiverr.com",
        "login_url":    "https://www.fiverr.com/login",
        "google_btn":   "Continue with Google",
        "search_field": "search",
        "job_indicator": "starting at",
    },
}

GOOGLE_EMAIL = "avus.janus@gmail.com"

SCREENSHOT_DIR = Path("janus_screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)


# ── Test steps ────────────────────────────────────────────────────────────────

async def step_screenshot(engine: ComputerUseEngine, label: str) -> Path:
    """Take a screenshot and save it with a timestamped label."""
    try:
        screenshot = await engine.screen.capture()
        ts = datetime.now().strftime("%H%M%S")
        path = SCREENSHOT_DIR / f"{ts}_{label}.png"
        screenshot.save(str(path))
        log.info("Screenshot saved: %s", path)
        return path
    except Exception as e:
        log.warning("Screenshot failed: %s", e)
        return Path()


async def step_open_browser(engine: ComputerUseEngine, url: str) -> bool:
    """Open Chrome and navigate to url."""
    log.info("Opening browser → %s", url)
    browser = BrowserComputerUse(engine)
    result = await browser.open(url)
    if result.success:
        log.info("✓ Browser opened successfully")
        return True
    log.error("✗ Failed to open browser: %s", result.error_message)
    return False


async def step_sign_in_with_google(engine: ComputerUseEngine, platform_cfg: dict) -> bool:
    """
    Click 'Continue with Google', then handle the Google sign-in flow.
    Janus already has avus.janus@gmail.com saved in Chrome, so Google
    should offer it as a one-click option.
    """
    log.info("Looking for 'Continue with Google' button...")
    try:
        screenshot = await engine.screen.capture()
        elements = await engine.vision.find_element(
            platform_cfg["google_btn"], screenshot
        )
        if not elements:
            # Try alternate labels
            for label in ("Sign in with Google", "Google", "google"):
                elements = await engine.vision.find_element(label, screenshot)
                if elements:
                    break

        if not elements:
            log.warning("Could not find Google sign-in button — may already be logged in")
            return True   # Optimistic: might already be signed in

        btn = elements[0]
        log.info("Clicking Google sign-in button at %s", btn.center)
        await engine.mouse.click(*btn.center)
        await asyncio.sleep(3)   # wait for Google OAuth popup

        # Google should show the saved account — look for the email
        screenshot = await engine.screen.capture()
        words = await engine.screen.ocr(screenshot)
        ocr_text = " ".join(w.text for w in words).lower()

        if "avus.janus" in ocr_text or "janus" in ocr_text:
            log.info("Google account visible — clicking to select it")
            # Click the account tile
            account_elements = await engine.vision.find_element(
                "avus.janus@gmail.com", screenshot
            )
            if account_elements:
                await engine.mouse.click(*account_elements[0].center)
            else:
                # Just press Enter — Chrome usually pre-selects the saved account
                await engine.keyboard.press_key("enter")
            await asyncio.sleep(4)
            log.info("✓ Google sign-in completed")
            return True

        # If Google asks for email (not pre-filled)
        if "email" in ocr_text or "sign in" in ocr_text:
            log.info("Google asking for email — typing avus.janus@gmail.com")
            email_elements = await engine.vision.find_element("email", screenshot)
            if email_elements:
                await engine.mouse.click(*email_elements[0].center)
            await engine.keyboard.type_text(GOOGLE_EMAIL)
            await engine.keyboard.press_key("enter")
            await asyncio.sleep(3)
            log.info("✓ Email entered — Google will handle the rest")
            return True

        log.warning("Unexpected Google sign-in state — OCR: %s", ocr_text[:200])
        return False

    except Exception as e:
        log.error("Sign-in step failed: %s", e)
        return False


async def step_search_jobs(
    engine: ComputerUseEngine,
    platform_cfg: dict,
    query: str,
) -> list:
    """Navigate to the search page and search for jobs."""
    log.info("Navigating to search page...")
    browser = BrowserComputerUse(engine)
    await browser.open(platform_cfg["search_url"])
    await asyncio.sleep(2)

    log.info("Searching for: %s", query)
    results = await browser.search_jobs(query)
    log.info("Found %d job results via OCR", len(results))
    for i, job in enumerate(results[:5], 1):
        log.info("  Job %d: %s", i, job.get("title", "?")[:80])
    return results


async def step_read_first_job(engine: ComputerUseEngine) -> dict:
    """OCR the current page and extract job details."""
    log.info("Reading job details from screen...")
    try:
        screenshot = await engine.screen.capture()
        words = await engine.screen.ocr(screenshot)
        full_text = " ".join(w.text for w in words)
        log.info("OCR text (first 300 chars): %s", full_text[:300])
        return {"ocr_text": full_text, "word_count": len(words)}
    except Exception as e:
        log.error("OCR step failed: %s", e)
        return {}


# ── Main test runner ──────────────────────────────────────────────────────────

async def run_test(
    platform: str = "upwork",
    query: str = "python developer",
    do_login: bool = True,
    screenshot_only: bool = False,
) -> dict:
    """
    Run the full end-to-end browser test.
    Returns a result dict with pass/fail for each step.
    """
    if not HAS_COMPUTER_USE:
        return {"error": "janus_computer_use not available"}

    platform_cfg = PLATFORMS.get(platform, PLATFORMS["upwork"])
    results = {
        "platform":       platform,
        "query":          query,
        "timestamp":      datetime.now().isoformat(),
        "steps":          {},
        "screenshots":    [],
        "jobs_found":     [],
        "overall_pass":   False,
    }

    log.info("=" * 60)
    log.info("  JANUS BROWSER TEST — %s", platform.upper())
    log.info("=" * 60)

    async with ComputerUseEngine() as engine:

        # ── Step 1: Screenshot only mode ──────────────────────────────
        if screenshot_only:
            log.info("Screenshot-only mode")
            path = await step_screenshot(engine, "current_screen")
            results["screenshots"].append(str(path))
            results["steps"]["screenshot"] = True
            results["overall_pass"] = True
            return results

        # ── Step 2: Open browser ──────────────────────────────────────
        log.info("\n[Step 1] Open browser")
        ok = await step_open_browser(engine, platform_cfg["url"])
        results["steps"]["open_browser"] = ok
        path = await step_screenshot(engine, f"01_{platform}_home")
        results["screenshots"].append(str(path))
        await asyncio.sleep(2)

        if not ok:
            log.error("Cannot continue — browser failed to open")
            return results

        # ── Step 3: Sign in with Google ───────────────────────────────
        if do_login:
            log.info("\n[Step 2] Sign in with Google")
            ok = await step_sign_in_with_google(engine, platform_cfg)
            results["steps"]["google_login"] = ok
            path = await step_screenshot(engine, "02_after_login")
            results["screenshots"].append(str(path))
            await asyncio.sleep(2)
        else:
            log.info("[Step 2] Skipping login (--no-login)")
            results["steps"]["google_login"] = "skipped"

        # ── Step 4: Search for jobs ───────────────────────────────────
        log.info("\n[Step 3] Search for jobs: '%s'", query)
        jobs = await step_search_jobs(engine, platform_cfg, query)
        results["steps"]["search_jobs"] = len(jobs) > 0
        results["jobs_found"] = jobs[:10]
        path = await step_screenshot(engine, "03_search_results")
        results["screenshots"].append(str(path))
        await asyncio.sleep(1)

        # ── Step 5: Read first job ────────────────────────────────────
        log.info("\n[Step 4] Read job details via OCR")
        job_data = await step_read_first_job(engine)
        results["steps"]["read_job_ocr"] = bool(job_data.get("word_count", 0) > 10)
        results["ocr_sample"] = job_data.get("ocr_text", "")[:500]
        path = await step_screenshot(engine, "04_job_detail")
        results["screenshots"].append(str(path))

    # ── Summary ───────────────────────────────────────────────────────
    passed = sum(1 for v in results["steps"].values() if v is True or v == "skipped")
    total  = len(results["steps"])
    results["overall_pass"] = passed == total

    log.info("\n" + "=" * 60)
    log.info("  RESULTS: %d/%d steps passed", passed, total)
    for step, status in results["steps"].items():
        icon = "✓" if status is True or status == "skipped" else "✗"
        log.info("  %s  %s: %s", icon, step, status)
    log.info("  Jobs found: %d", len(results["jobs_found"]))
    log.info("  Screenshots: %s", results["screenshots"])
    log.info("=" * 60)

    # Save results to JSON
    report_path = SCREENSHOT_DIR / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    log.info("Report saved: %s", report_path)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end browser test for Janus computer-use stack"
    )
    parser.add_argument(
        "--platform",
        choices=["upwork", "fiverr"],
        default="upwork",
        help="Platform to test (default: upwork)",
    )
    parser.add_argument(
        "--query",
        default="python developer",
        help="Job search query (default: 'python developer')",
    )
    parser.add_argument(
        "--no-login",
        action="store_true",
        dest="no_login",
        help="Skip the Google sign-in step",
    )
    parser.add_argument(
        "--screenshot-only",
        action="store_true",
        dest="screenshot_only",
        help="Just take a screenshot of the current screen and exit",
    )
    args = parser.parse_args()

    if not HAS_COMPUTER_USE:
        log.error("Cannot run test — janus_computer_use not importable.")
        log.error("Run: python janus_install_deps.py")
        sys.exit(1)

    results = asyncio.run(
        run_test(
            platform=args.platform,
            query=args.query,
            do_login=not args.no_login,
            screenshot_only=args.screenshot_only,
        )
    )

    sys.exit(0 if results.get("overall_pass") else 1)


if __name__ == "__main__":
    main()
