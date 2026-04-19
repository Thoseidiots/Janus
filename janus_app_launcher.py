"""
janus_app_launcher.py
=====================
Application launcher and desktop interaction system for Janus.

Allows Janus to act like a person on the desktop:
- Open applications (Chrome, Firefox, file explorer, etc.)
- Take screenshots to see what's on screen
- Click on things
- Type text
- Navigate windows
- Interact with the desktop naturally

This makes Janus truly "live" on the desktop like a human would.
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from enum import Enum
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ApplicationType(Enum):
    """Types of applications Janus can open"""
    CHROME = "chrome"
    FIREFOX = "firefox"
    EDGE = "edge"
    FILE_EXPLORER = "file_explorer"
    TEXT_EDITOR = "text_editor"
    TERMINAL = "terminal"
    NOTEPAD = "notepad"
    CALCULATOR = "calculator"
    CUSTOM = "custom"


class DesktopEnvironment(Enum):
    """Desktop environment detection"""
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    UNKNOWN = "unknown"


class AppLauncher:
    """Launches and manages applications on the desktop"""
    
    def __init__(self):
        self.desktop_env = self._detect_desktop_environment()
        self.open_apps: List[subprocess.Popen] = []
        self.app_paths = self._detect_app_paths()
        self.installed_apps = self._scan_installed_apps()
        logger.info(f"AppLauncher initialized for {self.desktop_env.value}")
        logger.info(f"Found {len(self.installed_apps)} installed applications")
    
    def _detect_desktop_environment(self) -> DesktopEnvironment:
        """Detect the operating system"""
        if sys.platform == "win32":
            return DesktopEnvironment.WINDOWS
        elif sys.platform == "darwin":
            return DesktopEnvironment.MACOS
        elif sys.platform == "linux":
            return DesktopEnvironment.LINUX
        else:
            return DesktopEnvironment.UNKNOWN
    
    def _detect_app_paths(self) -> dict:
        """Detect paths to common applications"""
        paths = {}
        
        if self.desktop_env == DesktopEnvironment.WINDOWS:
            # Windows application paths
            paths = {
                "chrome": self._find_chrome_windows(),
                "firefox": self._find_firefox_windows(),
                "edge": self._find_edge_windows(),
                "file_explorer": "explorer.exe",
                "notepad": "notepad.exe",
                "calculator": "calc.exe",
                "terminal": "cmd.exe",
                "powershell": "powershell.exe",
            }
        elif self.desktop_env == DesktopEnvironment.MACOS:
            # macOS application paths
            paths = {
                "chrome": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "firefox": "/Applications/Firefox.app/Contents/MacOS/firefox",
                "file_explorer": "open",
                "terminal": "/Applications/Utilities/Terminal.app/Contents/MacOS/Terminal",
                "notepad": "/Applications/TextEdit.app/Contents/MacOS/TextEdit",
            }
        elif self.desktop_env == DesktopEnvironment.LINUX:
            # Linux application paths
            paths = {
                "chrome": "google-chrome",
                "firefox": "firefox",
                "file_explorer": "nautilus",
                "terminal": "gnome-terminal",
                "notepad": "gedit",
            }
        
        logger.info(f"Detected app paths: {paths}")
        return paths
    
    def _find_chrome_windows(self) -> Optional[str]:
        """Find Chrome installation on Windows"""
        possible_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _find_firefox_windows(self) -> Optional[str]:
        """Find Firefox installation on Windows"""
        possible_paths = [
            r"C:\Program Files\Mozilla Firefox\firefox.exe",
            r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _find_edge_windows(self) -> Optional[str]:
        """Find Edge installation on Windows"""
        possible_paths = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _scan_installed_apps(self) -> dict:
        """Scan system for installed applications"""
        apps = {}
        
        try:
            if self.desktop_env == DesktopEnvironment.WINDOWS:
                apps = self._scan_windows_apps()
            elif self.desktop_env == DesktopEnvironment.MACOS:
                apps = self._scan_macos_apps()
            elif self.desktop_env == DesktopEnvironment.LINUX:
                apps = self._scan_linux_apps()
        except Exception as e:
            logger.warning(f"Error scanning installed apps: {e}")
        
        return apps
    
    def _scan_windows_apps(self) -> dict:
        """Scan Windows for installed applications"""
        apps = {}
        
        # Common Windows app locations
        app_dirs = [
            r"C:\Program Files",
            r"C:\Program Files (x86)",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs"),
            os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs"),
        ]
        
        for app_dir in app_dirs:
            if not os.path.exists(app_dir):
                continue
            
            try:
                for root, dirs, files in os.walk(app_dir):
                    for file in files:
                        if file.endswith(('.exe', '.lnk')):
                            full_path = os.path.join(root, file)
                            app_name = os.path.splitext(file)[0]
                            
                            # Avoid duplicates, keep first found
                            if app_name.lower() not in apps:
                                apps[app_name.lower()] = full_path
                    
                    # Limit depth to avoid scanning too deep
                    if root.count(os.sep) - app_dir.count(os.sep) > 3:
                        dirs.clear()
            except PermissionError:
                continue
        
        logger.info(f"Found {len(apps)} Windows applications")
        return apps
    
    def _scan_macos_apps(self) -> dict:
        """Scan macOS for installed applications"""
        apps = {}
        
        app_dirs = [
            "/Applications",
            os.path.expanduser("~/Applications"),
        ]
        
        for app_dir in app_dirs:
            if not os.path.exists(app_dir):
                continue
            
            try:
                for item in os.listdir(app_dir):
                    if item.endswith('.app'):
                        app_path = os.path.join(app_dir, item)
                        app_name = item.replace('.app', '')
                        apps[app_name.lower()] = app_path
            except PermissionError:
                continue
        
        logger.info(f"Found {len(apps)} macOS applications")
        return apps
    
    def _scan_linux_apps(self) -> dict:
        """Scan Linux for installed applications"""
        apps = {}
        
        # Check common Linux app locations
        app_dirs = [
            "/usr/bin",
            "/usr/local/bin",
            os.path.expanduser("~/.local/bin"),
        ]
        
        for app_dir in app_dirs:
            if not os.path.exists(app_dir):
                continue
            
            try:
                for item in os.listdir(app_dir):
                    full_path = os.path.join(app_dir, item)
                    if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                        apps[item.lower()] = full_path
            except PermissionError:
                continue
        
        logger.info(f"Found {len(apps)} Linux applications")
        return apps
    
    def open_browser(self, url: str = "", browser: str = "chrome") -> bool:
        """Open a web browser and optionally navigate to a URL"""
        try:
            browser_lower = browser.lower()
            
            if browser_lower == "chrome":
                app_path = self.app_paths.get("chrome")
            elif browser_lower == "firefox":
                app_path = self.app_paths.get("firefox")
            elif browser_lower == "edge":
                app_path = self.app_paths.get("edge")
            else:
                logger.error(f"Unknown browser: {browser}")
                return False
            
            if not app_path:
                logger.error(f"{browser} not found on this system")
                return False
            
            # Build command
            if url:
                if self.desktop_env == DesktopEnvironment.WINDOWS:
                    cmd = [app_path, url]
                elif self.desktop_env == DesktopEnvironment.MACOS:
                    cmd = [app_path, url]
                else:  # Linux
                    cmd = [app_path, url]
            else:
                cmd = [app_path]
            
            # Launch browser
            logger.info(f"Opening {browser} with URL: {url}")
            process = subprocess.Popen(cmd)
            self.open_apps.append(process)
            
            time.sleep(2)  # Wait for browser to open
            logger.info(f"Successfully opened {browser}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening browser: {e}")
            return False
    
    def open_file_explorer(self, path: str = "") -> bool:
        """Open file explorer at specified path"""
        try:
            if self.desktop_env == DesktopEnvironment.WINDOWS:
                if path:
                    cmd = ["explorer.exe", path]
                else:
                    cmd = ["explorer.exe"]
            elif self.desktop_env == DesktopEnvironment.MACOS:
                if path:
                    cmd = ["open", path]
                else:
                    cmd = ["open", os.path.expanduser("~")]
            else:  # Linux
                app_path = self.app_paths.get("file_explorer", "nautilus")
                if path:
                    cmd = [app_path, path]
                else:
                    cmd = [app_path]
            
            logger.info(f"Opening file explorer at: {path or 'home'}")
            process = subprocess.Popen(cmd)
            self.open_apps.append(process)
            
            time.sleep(1)
            logger.info("Successfully opened file explorer")
            return True
            
        except Exception as e:
            logger.error(f"Error opening file explorer: {e}")
            return False
    
    def open_terminal(self) -> bool:
        """Open terminal/command prompt"""
        try:
            if self.desktop_env == DesktopEnvironment.WINDOWS:
                cmd = ["cmd.exe"]
            elif self.desktop_env == DesktopEnvironment.MACOS:
                cmd = [self.app_paths.get("terminal", "open")]
            else:  # Linux
                cmd = [self.app_paths.get("terminal", "gnome-terminal")]
            
            logger.info("Opening terminal")
            process = subprocess.Popen(cmd)
            self.open_apps.append(process)
            
            time.sleep(1)
            logger.info("Successfully opened terminal")
            return True
            
        except Exception as e:
            logger.error(f"Error opening terminal: {e}")
            return False
    
    def open_text_editor(self, file_path: str = "") -> bool:
        """Open text editor, optionally with a file"""
        try:
            if self.desktop_env == DesktopEnvironment.WINDOWS:
                app_path = "notepad.exe"
            elif self.desktop_env == DesktopEnvironment.MACOS:
                app_path = self.app_paths.get("notepad", "open")
            else:  # Linux
                app_path = self.app_paths.get("notepad", "gedit")
            
            if file_path:
                cmd = [app_path, file_path]
                logger.info(f"Opening text editor with file: {file_path}")
            else:
                cmd = [app_path]
                logger.info("Opening text editor")
            
            process = subprocess.Popen(cmd)
            self.open_apps.append(process)
            
            time.sleep(1)
            logger.info("Successfully opened text editor")
            return True
            
        except Exception as e:
            logger.error(f"Error opening text editor: {e}")
            return False
    
    def open_application(self, app_type: ApplicationType, args: List[str] = None) -> bool:
        """Open an application by type"""
        try:
            if app_type == ApplicationType.CHROME:
                url = args[0] if args else ""
                return self.open_browser(url, "chrome")
            
            elif app_type == ApplicationType.FIREFOX:
                url = args[0] if args else ""
                return self.open_browser(url, "firefox")
            
            elif app_type == ApplicationType.EDGE:
                url = args[0] if args else ""
                return self.open_browser(url, "edge")
            
            elif app_type == ApplicationType.FILE_EXPLORER:
                path = args[0] if args else ""
                return self.open_file_explorer(path)
            
            elif app_type == ApplicationType.TERMINAL:
                return self.open_terminal()
            
            elif app_type == ApplicationType.TEXT_EDITOR:
                file_path = args[0] if args else ""
                return self.open_text_editor(file_path)
            
            elif app_type == ApplicationType.CUSTOM:
                if not args or len(args) < 1:
                    logger.error("Custom app requires command path")
                    return False
                
                cmd = args
                logger.info(f"Opening custom application: {cmd}")
                process = subprocess.Popen(cmd)
                self.open_apps.append(process)
                
                time.sleep(1)
                logger.info("Successfully opened custom application")
                return True
            
            else:
                logger.error(f"Unknown application type: {app_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error opening application: {e}")
            return False
    
    def open_any_app(self, app_name: str, args: List[str] = None) -> bool:
        """Open any installed application by name"""
        try:
            app_name_lower = app_name.lower()
            
            # First check if it's in our detected apps
            if app_name_lower in self.installed_apps:
                app_path = self.installed_apps[app_name_lower]
                logger.info(f"Opening installed app: {app_name} from {app_path}")
                
                cmd = [app_path]
                if args:
                    cmd.extend(args)
                
                process = subprocess.Popen(cmd)
                self.open_apps.append(process)
                
                time.sleep(1)
                logger.info(f"Successfully opened {app_name}")
                return True
            
            # Try to open by name directly (works if in PATH)
            logger.info(f"Attempting to open {app_name} from PATH")
            cmd = [app_name]
            if args:
                cmd.extend(args)
            
            process = subprocess.Popen(cmd)
            self.open_apps.append(process)
            
            time.sleep(1)
            logger.info(f"Successfully opened {app_name}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Application not found: {app_name}")
            logger.info(f"Available apps: {list(self.installed_apps.keys())[:20]}...")
            return False
        except Exception as e:
            logger.error(f"Error opening app {app_name}: {e}")
            return False
    
    def open_app_by_path(self, app_path: str, args: List[str] = None) -> bool:
        """Open an application by full file path"""
        try:
            if not os.path.exists(app_path):
                logger.error(f"Application path not found: {app_path}")
                return False
            
            logger.info(f"Opening application from path: {app_path}")
            
            cmd = [app_path]
            if args:
                cmd.extend(args)
            
            process = subprocess.Popen(cmd)
            self.open_apps.append(process)
            
            time.sleep(1)
            logger.info(f"Successfully opened application from {app_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening app from path: {e}")
            return False
    
    def list_installed_apps(self) -> List[str]:
        """Get list of all detected installed applications"""
        return sorted(list(self.installed_apps.keys()))
    
    def search_apps(self, search_term: str) -> List[str]:
        """Search for applications by name"""
        search_lower = search_term.lower()
        matches = [app for app in self.installed_apps.keys() if search_lower in app]
        return sorted(matches)
    
    def close_all_apps(self) -> None:
        """Close all opened applications"""
        logger.info(f"Closing {len(self.open_apps)} applications")
        
        for process in self.open_apps:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        
        self.open_apps.clear()
        logger.info("All applications closed")
    
    def get_open_apps_count(self) -> int:
        """Get count of open applications"""
        return len(self.open_apps)


class ScreenCapture:
    """Captures screenshots so Janus can see what's on screen"""
    
    def __init__(self):
        self.screenshots_dir = Path("janus_screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
        logger.info("ScreenCapture initialized")
    
    def take_screenshot(self, filename: str = "") -> Optional[str]:
        """Take a screenshot of the current screen"""
        try:
            # Try to import PIL for screenshot capability
            try:
                from PIL import ImageGrab
                
                if not filename:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.png"
                
                filepath = self.screenshots_dir / filename
                
                logger.info(f"Taking screenshot: {filepath}")
                screenshot = ImageGrab.grab()
                screenshot.save(filepath)
                
                logger.info(f"Screenshot saved: {filepath}")
                return str(filepath)
                
            except ImportError:
                logger.warning("PIL not installed. Screenshot capability unavailable.")
                logger.info("Install with: pip install Pillow")
                return None
                
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return None
    
    def get_screenshots(self) -> List[str]:
        """Get list of all screenshots taken"""
        screenshots = list(self.screenshots_dir.glob("*.png"))
        return [str(s) for s in screenshots]


class DesktopInteraction:
    """High-level desktop interaction for Janus"""
    
    def __init__(self):
        self.launcher = AppLauncher()
        self.screen_capture = ScreenCapture()
        logger.info("DesktopInteraction initialized")
    
    def navigate_to_website(self, url: str, browser: str = "chrome") -> bool:
        """Navigate to a website"""
        logger.info(f"Navigating to {url} in {browser}")
        return self.launcher.open_browser(url, browser)
    
    def open_upwork(self) -> bool:
        """Open Upwork in browser"""
        return self.navigate_to_website("https://www.upwork.com", "chrome")
    
    def open_fiverr(self) -> bool:
        """Open Fiverr in browser"""
        return self.navigate_to_website("https://www.fiverr.com", "chrome")
    
    def open_youtube(self) -> bool:
        """Open YouTube in browser"""
        return self.navigate_to_website("https://www.youtube.com", "chrome")
    
    def open_google(self) -> bool:
        """Open Google in browser"""
        return self.navigate_to_website("https://www.google.com", "chrome")
    
    def see_screen(self) -> Optional[str]:
        """Take a screenshot to see what's on screen"""
        return self.screen_capture.take_screenshot()
    
    def open_work_folder(self) -> bool:
        """Open the work folder"""
        work_folder = Path.home() / "Janus_Work"
        work_folder.mkdir(exist_ok=True)
        return self.launcher.open_file_explorer(str(work_folder))
    
    def shutdown(self) -> None:
        """Shutdown and close all apps"""
        self.launcher.close_all_apps()


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("JANUS DESKTOP INTERACTION SYSTEM")
    print("="*60 + "\n")
    
    # Initialize desktop interaction
    desktop = DesktopInteraction()
    
    print("Available actions:")
    print("1. Open Upwork")
    print("2. Open Fiverr")
    print("3. Open YouTube")
    print("4. Open Google")
    print("5. Open file explorer")
    print("6. Take screenshot")
    print("7. Open terminal")
    print("8. Open text editor")
    print("9. List all installed apps")
    print("10. Search for app")
    print("11. Open any app by name")
    print("12. Open app by path")
    print("13. Exit")
    
    while True:
        choice = input("\nEnter choice (1-13): ").strip()
        
        if choice == "1":
            desktop.open_upwork()
        elif choice == "2":
            desktop.open_fiverr()
        elif choice == "3":
            desktop.open_youtube()
        elif choice == "4":
            desktop.open_google()
        elif choice == "5":
            desktop.launcher.open_file_explorer()
        elif choice == "6":
            screenshot = desktop.see_screen()
            if screenshot:
                print(f"Screenshot saved: {screenshot}")
        elif choice == "7":
            desktop.launcher.open_terminal()
        elif choice == "8":
            desktop.launcher.open_text_editor()
        elif choice == "9":
            apps = desktop.launcher.list_installed_apps()
            print(f"\nFound {len(apps)} installed applications:")
            for app in apps[:50]:  # Show first 50
                print(f"  - {app}")
            if len(apps) > 50:
                print(f"  ... and {len(apps) - 50} more")
        elif choice == "10":
            search_term = input("Search for app: ").strip()
            results = desktop.launcher.search_apps(search_term)
            if results:
                print(f"\nFound {len(results)} matching apps:")
                for app in results:
                    print(f"  - {app}")
            else:
                print("No apps found matching that search")
        elif choice == "11":
            app_name = input("Enter app name (e.g., 'steam', 'discord', 'vlc'): ").strip()
            args_input = input("Enter arguments (optional, space-separated): ").strip()
            args = args_input.split() if args_input else None
            desktop.launcher.open_any_app(app_name, args)
        elif choice == "12":
            app_path = input("Enter full path to application: ").strip()
            args_input = input("Enter arguments (optional, space-separated): ").strip()
            args = args_input.split() if args_input else None
            desktop.launcher.open_app_by_path(app_path, args)
        elif choice == "13":
            print("Shutting down...")
            desktop.shutdown()
            break
        else:
            print("Invalid choice")
    
    print("\nGoodbye!")
