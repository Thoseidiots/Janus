"""
Browser Automation for Financial Tasks - No APIs
Uses Selenium/Playwright to automate browser interactions
No direct API calls - everything through UI automation
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger("janus_automation")

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False
    logger.warning("Selenium not installed - browser automation disabled")


class BrowserAutomationAgent:
    """
    Automate financial tasks through browser UI.
    No API keys needed - simulates human interaction.
    """
    
    def __init__(self):
        self.driver = None
        self.task_log = []
        self.screenshots = []
        logger.info("Browser Automation Agent initialized")
        
        if HAS_SELENIUM:
            logger.info("Selenium available - browser automation enabled")
        else:
            logger.warning("Selenium not available - install: pip install selenium")
    
    def start_browser(self, headless: bool = False) -> bool:
        """Start Chrome browser."""
        
        if not HAS_SELENIUM:
            logger.error("Selenium not available")
            return False
        
        try:
            options = Options()
            if headless:
                options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--start-maximized")
            
            self.driver = webdriver.Chrome(options=options)
            logger.info("Browser started")
            return True
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            return False
    
    def navigate_to(self, url: str) -> bool:
        """Navigate to URL."""
        if not self.driver:
            return False
        
        try:
            self.driver.get(url)
            self.log_action("navigate", {"url": url})
            return True
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return False
    
    def find_element(self, selector: str, by=By.CSS_SELECTOR, timeout: int = 10):
        """Find element by selector."""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
            return element
        except:
            return None
    
    def click(self, selector: str, by=By.CSS_SELECTOR) -> bool:
        """Click element."""
        try:
            element = self.find_element(selector, by)
            if element:
                element.click()
                self.log_action("click", {"selector": selector})
                return True
        except Exception as e:
            logger.error(f"Click failed: {e}")
        return False
    
    def type_text(self, selector: str, text: str, by=By.CSS_SELECTOR) -> bool:
        """Type text into element."""
        try:
            element = self.find_element(selector, by)
            if element:
                element.clear()
                element.send_keys(text)
                self.log_action("type", {"selector": selector, "text": f"***{len(text)} chars***"})
                return True
        except Exception as e:
            logger.error(f"Type failed: {e}")
        return False
    
    def get_text(self, selector: str, by=By.CSS_SELECTOR) -> Optional[str]:
        """Get text from element."""
        try:
            element = self.find_element(selector, by)
            if element:
                return element.text
        except Exception as e:
            logger.error(f"Get text failed: {e}")
        return None
    
    def take_screenshot(self, filename: str = None) -> str:
        """Take screenshot."""
        if not self.driver:
            return ""
        
        if filename is None:
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        try:
            self.driver.save_screenshot(filename)
            self.screenshots.append(filename)
            self.log_action("screenshot", {"filename": filename})
            return filename
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
        return ""
    
    def get_page_source(self) -> str:
        """Get page HTML."""
        if not self.driver:
            return ""
        return self.driver.page_source
    
    def get_current_url(self) -> str:
        """Get current URL."""
        if not self.driver:
            return ""
        return self.driver.current_url
    
    def wait_for_element(self, selector: str, by=By.CSS_SELECTOR, timeout: int = 10) -> bool:
        """Wait for element to be present."""
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
            return True
        except:
            return False
    
    def scroll(self, direction: str = "down", amount: int = 500) -> bool:
        """Scroll page."""
        try:
            if direction == "down":
                self.driver.execute_script(f"window.scrollBy(0, {amount});")
            elif direction == "up":
                self.driver.execute_script(f"window.scrollBy(0, -{amount});")
            elif direction == "right":
                self.driver.execute_script(f"window.scrollBy({amount}, 0);")
            else:
                self.driver.execute_script(f"window.scrollBy(-{amount}, 0);")
            
            self.log_action("scroll", {"direction": direction, "amount": amount})
            return True
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
        return False
    
    def close(self):
        """Close browser."""
        if self.driver:
            self.driver.quit()
            logger.info("Browser closed")
    
    def log_action(self, action_type: str, details: Dict):
        """Log an action."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": action_type,
            "details": details
        }
        self.task_log.append(entry)
        logger.debug(f"Action: {action_type} - {details}")
    
    def get_task_log(self) -> List[Dict]:
        """Get action log."""
        return self.task_log
    
    def save_task_log(self, filename: str = "automation_log.json"):
        """Save task log to file."""
        with open(filename, 'w') as f:
            json.dump(self.task_log, f, indent=2)
        logger.info(f"Task log saved to {filename}")


class FinancialAutomationWorkflow:
    """
    Pre-built workflows for common financial tasks.
    All done through browser automation - no APIs.
    """
    
    def __init__(self, browser_agent: BrowserAutomationAgent):
        self.agent = browser_agent
    
    def check_bank_balance(self, bank_url: str, username_selector: str, 
                          password_selector: str, username: str, password: str) -> Dict:
        """
        Automated login and balance check.
        Scrapes data from UI instead of using API.
        """
        result = {
            "task": "check_bank_balance",
            "status": "started",
            "steps": []
        }
        
        # Navigate to bank
        if not self.agent.navigate_to(bank_url):
            result["status"] = "failed"
            result["error"] = "Navigation failed"
            return result
        
        result["steps"].append({"step": "navigate", "status": "success"})
        
        # Login
        if self.agent.type_text(username_selector, username):
            result["steps"].append({"step": "enter_username", "status": "success"})
        
        if self.agent.type_text(password_selector, password):
            result["steps"].append({"step": "enter_password", "status": "success"})
        
        # Submit login (look for login button)
        if self.agent.click("button[type='submit']"):
            result["steps"].append({"step": "submit_login", "status": "success"})
        
        # Wait for dashboard
        if self.agent.wait_for_element(".balance, [data-balance]", timeout=10):
            result["steps"].append({"step": "wait_dashboard", "status": "success"})
            
            # Extract balance (depends on bank HTML structure)
            balance_text = self.agent.get_text(".balance, [data-balance]")
            result["balance"] = balance_text
            result["status"] = "success"
        else:
            result["status"] = "failed"
            result["error"] = "Dashboard not loaded"
        
        return result
    
    def monitor_stock_price(self, stock_url: str, stock_symbol: str) -> Dict:
        """
        Monitor stock price by scraping from free websites.
        Examples: yahoo.com, investing.com, tradingview.com
        """
        result = {
            "task": "monitor_stock",
            "symbol": stock_symbol,
            "status": "started"
        }
        
        if not self.agent.navigate_to(stock_url):
            result["status"] = "failed"
            return result
        
        # Take screenshot for visual verification
        screenshot = self.agent.take_screenshot(f"stock_{stock_symbol}.png")
        result["screenshot"] = screenshot
        
        # Try to extract price from common selectors
        price_selectors = [
            ".price",
            "[data-price]",
            ".current-price",
            "[class*='price']"
        ]
        
        price = None
        for selector in price_selectors:
            price = self.agent.get_text(selector)
            if price:
                break
        
        if price:
            result["current_price"] = price
            result["status"] = "success"
        else:
            result["status"] = "partial"
            result["note"] = "Price not found with standard selectors"
        
        return result
    
    def transfer_between_accounts(self, transfer_url: str, from_account: str,
                                 to_account: str, amount: str) -> Dict:
        """
        Automate bank transfer through UI.
        Requires pre-login or persistent session.
        """
        result = {
            "task": "transfer_money",
            "status": "started",
            "steps": []
        }
        
        # Navigate to transfer page
        if not self.agent.navigate_to(transfer_url):
            result["status"] = "failed"
            return result
        
        # Select from account
        if self.agent.click(f"option[value='{from_account}'], select"):
            result["steps"].append({"step": "select_from_account", "status": "success"})
        
        # Enter to account
        if self.agent.type_text("input[name*='to'], input[name*='recipient']", to_account):
            result["steps"].append({"step": "enter_to_account", "status": "success"})
        
        # Enter amount
        if self.agent.type_text("input[name*='amount'], input[type='number']", amount):
            result["steps"].append({"step": "enter_amount", "status": "success"})
        
        # Review & Confirm (take screenshot before confirming)
        self.agent.take_screenshot("transfer_review.png")
        
        if self.agent.click("button[name*='confirm'], button[type='submit']"):
            result["steps"].append({"step": "confirm_transfer", "status": "success"})
            result["status"] = "completed"
        else:
            result["status"] = "pending_confirmation"
        
        return result
    
    def track_expenses(self, expense_tracker_url: str) -> Dict:
        """
        Scrape expense tracking website or automate expense entry.
        """
        result = {
            "task": "track_expenses",
            "status": "started"
        }
        
        if not self.agent.navigate_to(expense_tracker_url):
            result["status"] = "failed"
            return result
        
        # Take screenshot of current expenses
        self.agent.take_screenshot("expenses_dashboard.png")
        
        # Try to extract expense data from table
        try:
            # This is simplified - real implementation would parse HTML more carefully
            page_source = self.agent.get_page_source()
            result["data_captured"] = True
            result["status"] = "success"
        except:
            result["status"] = "failed"
        
        return result


if __name__ == "__main__":
    print("Browser Automation for Financial Tasks")
    print("=" * 50)
    
    if HAS_SELENIUM:
        print("✓ Selenium available")
        print("\nFeatures:")
        print("  • Automate login/logout")
        print("  • Fill forms")
        print("  • Extract data from websites")
        print("  • Monitor prices")
        print("  • Transfer money (through UI)")
        print("  • Track expenses")
        print("\nNo API keys needed - everything through browser automation!")
    else:
        print("✗ Selenium not installed")
        print("  Install: pip install selenium")
        print("  Download ChromeDriver: https://chromedriver.chromium.org/")
