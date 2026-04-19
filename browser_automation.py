"""
Browser Automation for Janus
Specialized browser control for web navigation, form filling, and interaction
"""

from typing import Optional, Dict, List, Any
import time
from dataclasses import dataclass

try:
    from os_human_interface import JanusOS
    from window_manager import WindowManager
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False


@dataclass
class BrowserElement:
    """Represents a browser UI element"""
    element_type: str  # 'button', 'input', 'link', 'dropdown'
    text: str
    position: tuple  # (x, y)
    size: tuple  # (width, height)
    confidence: float


class BrowserAutomation:
    """
    Specialized browser automation for Janus
    Works with Chrome, Firefox, Edge
    """
    
    def __init__(self):
        self.janus_os = JanusOS() if COMPONENTS_AVAILABLE else None
        self.window_manager = WindowManager() if COMPONENTS_AVAILABLE else None
        self.current_browser = None
        
    def find_browser(self) -> Optional[str]:
        """Find and activate browser window"""
        if not self.window_manager:
            return None
        
        browsers = ['Chrome', 'Firefox', 'Edge', 'Brave', 'Opera']
        
        for browser in browsers:
            window = self.window_manager.find_window(title_contains=browser)
            if window:
                self.window_manager.activate_window(window.hwnd)
                self.current_browser = browser
                time.sleep(0.5)
                return browser
        
        return None
    
    def open_browser(self, browser: str = 'chrome') -> bool:
        """Open browser application"""
        if not self.janus_os:
            return False
        
        try:
            # Press Win+R
            self.janus_os.press_key(0x5B)  # Win key
            time.sleep(0.1)
            self.janus_os.press_key(0x52)  # R key
            time.sleep(0.5)
            
            # Type browser name
            self.janus_os.type_string(browser)
            time.sleep(0.3)
            
            # Press Enter
            self.janus_os.press_key(0x0D)
            time.sleep(2)
            
            self.current_browser = browser
            return True
        except Exception as e:
            print(f"Error opening browser: {e}")
            return False
    
    def navigate_to_url(self, url: str) -> bool:
        """Navigate to URL"""
        if not self.janus_os:
            return False
        
        try:
            # Ensure browser is active
            if not self.find_browser():
                self.open_browser()
            
            # Focus address bar (Ctrl+L)
            self.janus_os.press_key(0x11)  # Ctrl
            time.sleep(0.05)
            self.janus_os.press_key(0x4C)  # L
            time.sleep(0.3)
            
            # Type URL
            self.janus_os.type_string(url)
            time.sleep(0.3)
            
            # Press Enter
            self.janus_os.press_key(0x0D)
            time.sleep(2)
            
            return True
        except Exception as e:
            print(f"Error navigating to URL: {e}")
            return False
    
    def click_link(self, link_text: str, approximate_position: tuple = None) -> bool:
        """Click a link by text or position"""
        if not self.janus_os:
            return False
        
        try:
            if approximate_position:
                x, y = approximate_position
                self.janus_os.click(x, y)
                time.sleep(1)
                return True
            else:
                # Use Ctrl+F to find text
                return self.find_and_click_text(link_text)
        except Exception as e:
            print(f"Error clicking link: {e}")
            return False
    
    def find_and_click_text(self, text: str) -> bool:
        """Find text on page and click it"""
        if not self.janus_os:
            return False
        
        try:
            # Open find dialog (Ctrl+F)
            self.janus_os.press_key(0x11)  # Ctrl
            time.sleep(0.05)
            self.janus_os.press_key(0x46)  # F
            time.sleep(0.5)
            
            # Type search text
            self.janus_os.type_string(text)
            time.sleep(0.5)
            
            # Close find dialog (Esc)
            self.janus_os.press_key(0x1B)
            time.sleep(0.3)
            
            # Click at center of screen (where found text should be highlighted)
            screen_w, screen_h = self.janus_os.get_screen_size()
            self.janus_os.click(screen_w // 2, screen_h // 2)
            time.sleep(1)
            
            return True
        except Exception as e:
            print(f"Error finding and clicking text: {e}")
            return False
    
    def fill_form_field(self, field_position: tuple, text: str) -> bool:
        """Fill a form field at given position"""
        if not self.janus_os:
            return False
        
        try:
            x, y = field_position
            
            # Click field
            self.janus_os.click(x, y)
            time.sleep(0.3)
            
            # Clear existing text (Ctrl+A, Delete)
            self.janus_os.press_key(0x11)  # Ctrl
            time.sleep(0.05)
            self.janus_os.press_key(0x41)  # A
            time.sleep(0.1)
            self.janus_os.press_key(0x2E)  # Delete
            time.sleep(0.2)
            
            # Type new text
            self.janus_os.type_string(text)
            time.sleep(0.3)
            
            return True
        except Exception as e:
            print(f"Error filling form field: {e}")
            return False
    
    def submit_form(self, submit_button_position: tuple = None) -> bool:
        """Submit form by clicking button or pressing Enter"""
        if not self.janus_os:
            return False
        
        try:
            if submit_button_position:
                x, y = submit_button_position
                self.janus_os.click(x, y)
            else:
                # Press Enter
                self.janus_os.press_key(0x0D)
            
            time.sleep(2)
            return True
        except Exception as e:
            print(f"Error submitting form: {e}")
            return False
    
    def scroll_page(self, direction: str = 'down', amount: int = 3) -> bool:
        """Scroll page up or down"""
        if not self.janus_os:
            return False
        
        try:
            key_code = 0x22 if direction == 'down' else 0x21  # Page Down / Page Up
            
            for _ in range(amount):
                self.janus_os.press_key(key_code)
                time.sleep(0.3)
            
            return True
        except Exception as e:
            print(f"Error scrolling page: {e}")
            return False
    
    def go_back(self) -> bool:
        """Navigate back"""
        if not self.janus_os:
            return False
        
        try:
            # Alt+Left Arrow
            self.janus_os.press_key(0x12)  # Alt
            time.sleep(0.05)
            self.janus_os.press_key(0x25)  # Left Arrow
            time.sleep(1)
            return True
        except Exception as e:
            print(f"Error going back: {e}")
            return False
    
    def go_forward(self) -> bool:
        """Navigate forward"""
        if not self.janus_os:
            return False
        
        try:
            # Alt+Right Arrow
            self.janus_os.press_key(0x12)  # Alt
            time.sleep(0.05)
            self.janus_os.press_key(0x27)  # Right Arrow
            time.sleep(1)
            return True
        except Exception as e:
            print(f"Error going forward: {e}")
            return False
    
    def refresh_page(self) -> bool:
        """Refresh current page"""
        if not self.janus_os:
            return False
        
        try:
            # F5
            self.janus_os.press_key(0x74)
            time.sleep(2)
            return True
        except Exception as e:
            print(f"Error refreshing page: {e}")
            return False
    
    def new_tab(self) -> bool:
        """Open new tab"""
        if not self.janus_os:
            return False
        
        try:
            # Ctrl+T
            self.janus_os.press_key(0x11)  # Ctrl
            time.sleep(0.05)
            self.janus_os.press_key(0x54)  # T
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"Error opening new tab: {e}")
            return False
    
    def close_tab(self) -> bool:
        """Close current tab"""
        if not self.janus_os:
            return False
        
        try:
            # Ctrl+W
            self.janus_os.press_key(0x11)  # Ctrl
            time.sleep(0.05)
            self.janus_os.press_key(0x57)  # W
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"Error closing tab: {e}")
            return False
    
    def switch_tab(self, direction: str = 'next') -> bool:
        """Switch to next or previous tab"""
        if not self.janus_os:
            return False
        
        try:
            # Ctrl+Tab (next) or Ctrl+Shift+Tab (previous)
            self.janus_os.press_key(0x11)  # Ctrl
            time.sleep(0.05)
            
            if direction == 'previous':
                self.janus_os.press_key(0x10)  # Shift
                time.sleep(0.05)
            
            self.janus_os.press_key(0x09)  # Tab
            time.sleep(0.3)
            return True
        except Exception as e:
            print(f"Error switching tab: {e}")
            return False
    
    def download_file(self, download_link_position: tuple) -> bool:
        """Click download link"""
        if not self.janus_os:
            return False
        
        try:
            x, y = download_link_position
            self.janus_os.click(x, y)
            time.sleep(2)
            return True
        except Exception as e:
            print(f"Error downloading file: {e}")
            return False
    
    def take_screenshot(self) -> Optional[bytes]:
        """Take screenshot of browser window"""
        if not self.janus_os:
            return None
        
        try:
            raw_bytes, w, h = self.janus_os.capture_screen()
            return raw_bytes
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            return None


# High-level browser tasks
class BrowserTasks:
    """High-level browser task automation"""
    
    def __init__(self):
        self.browser = BrowserAutomation()
    
    def search_google(self, query: str) -> bool:
        """Search on Google"""
        if not self.browser.navigate_to_url("https://www.google.com"):
            return False
        
        time.sleep(2)
        
        # Click search box (approximate center)
        screen_w, screen_h = self.browser.janus_os.get_screen_size()
        self.browser.fill_form_field((screen_w // 2, screen_h // 2), query)
        
        return self.browser.submit_form()
    
    def login_form(self, username_pos: tuple, password_pos: tuple,
                   username: str, password: str, 
                   submit_pos: tuple = None) -> bool:
        """Fill and submit login form"""
        if not self.browser.fill_form_field(username_pos, username):
            return False
        
        time.sleep(0.5)
        
        if not self.browser.fill_form_field(password_pos, password):
            return False
        
        time.sleep(0.5)
        
        return self.browser.submit_form(submit_pos)
    
    def read_article(self, url: str, scroll_times: int = 5) -> bool:
        """Navigate to article and scroll through it"""
        if not self.browser.navigate_to_url(url):
            return False
        
        time.sleep(3)
        
        return self.browser.scroll_page('down', scroll_times)


if __name__ == "__main__":
    # Test browser automation
    browser = BrowserAutomation()
    
    print("Browser Automation Test")
    print("=" * 60)
    
    print("\n1. Finding browser...")
    found = browser.find_browser()
    if found:
        print(f"   ✓ Found: {found}")
    else:
        print("   Opening browser...")
        browser.open_browser()
    
    print("\n2. Testing navigation...")
    if browser.navigate_to_url("https://www.example.com"):
        print("   ✓ Navigated to example.com")
    
    print("\n" + "=" * 60)
    print("Browser automation ready!")
