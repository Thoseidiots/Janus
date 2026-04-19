"""
Window Manager for Janus
Provides human-level window management capabilities on Windows
"""

import ctypes
import ctypes.wintypes as wintypes
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time

# Windows Constants
SW_MAXIMIZE = 3
SW_MINIMIZE = 6
SW_RESTORE = 9
SW_SHOW = 5
SW_HIDE = 0
GW_HWNDNEXT = 2
GWL_STYLE = -16
WS_VISIBLE = 0x10000000
WS_MINIMIZE = 0x20000000

@dataclass
class WindowInfo:
    """Information about a window"""
    hwnd: int
    title: str
    class_name: str
    is_visible: bool
    is_minimized: bool
    rect: Tuple[int, int, int, int]  # (left, top, right, bottom)
    process_id: int
    
    @property
    def width(self):
        return self.rect[2] - self.rect[0]
    
    @property
    def height(self):
        return self.rect[3] - self.rect[1]
    
    @property
    def position(self):
        return (self.rect[0], self.rect[1])
    
    @property
    def size(self):
        return (self.width, self.height)


class WindowManager:
    """Manages windows like a human would"""
    
    def __init__(self):
        self.user32 = ctypes.windll.user32
        self.kernel32 = ctypes.windll.kernel32
        
    def get_all_windows(self) -> List[WindowInfo]:
        """Get all visible windows"""
        windows = []
        
        def enum_callback(hwnd, lparam):
            if self.user32.IsWindowVisible(hwnd):
                title = self._get_window_title(hwnd)
                if title:  # Only include windows with titles
                    windows.append(self._get_window_info(hwnd))
            return True
        
        EnumWindowsProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int)
        )
        
        self.user32.EnumWindows(EnumWindowsProc(enum_callback), 0)
        return windows
    
    def get_active_window(self) -> Optional[WindowInfo]:
        """Get currently active window"""
        hwnd = self.user32.GetForegroundWindow()
        if hwnd:
            return self._get_window_info(hwnd)
        return None
    
    def find_window(self, title_contains: str = None, 
                   class_name: str = None) -> Optional[WindowInfo]:
        """Find window by title or class name"""
        windows = self.get_all_windows()
        
        for window in windows:
            if title_contains and title_contains.lower() in window.title.lower():
                return window
            if class_name and class_name == window.class_name:
                return window
        
        return None
    
    def activate_window(self, hwnd: int) -> bool:
        """Bring window to foreground"""
        try:
            # Restore if minimized
            if self.user32.IsIconic(hwnd):
                self.user32.ShowWindow(hwnd, SW_RESTORE)
                time.sleep(0.1)
            
            # Bring to foreground
            self.user32.SetForegroundWindow(hwnd)
            time.sleep(0.1)
            return True
        except Exception as e:
            print(f"Error activating window: {e}")
            return False
    
    def switch_to_window(self, title_contains: str) -> bool:
        """Switch to window by title (like Alt+Tab)"""
        window = self.find_window(title_contains=title_contains)
        if window:
            return self.activate_window(window.hwnd)
        return False
    
    def minimize_window(self, hwnd: int) -> bool:
        """Minimize window"""
        try:
            self.user32.ShowWindow(hwnd, SW_MINIMIZE)
            return True
        except:
            return False
    
    def maximize_window(self, hwnd: int) -> bool:
        """Maximize window"""
        try:
            self.user32.ShowWindow(hwnd, SW_MAXIMIZE)
            return True
        except:
            return False
    
    def restore_window(self, hwnd: int) -> bool:
        """Restore window to normal size"""
        try:
            self.user32.ShowWindow(hwnd, SW_RESTORE)
            return True
        except:
            return False
    
    def close_window(self, hwnd: int) -> bool:
        """Close window gracefully"""
        try:
            WM_CLOSE = 0x0010
            self.user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
            return True
        except:
            return False
    
    def move_window(self, hwnd: int, x: int, y: int, 
                   width: int = None, height: int = None) -> bool:
        """Move and optionally resize window"""
        try:
            if width is None or height is None:
                # Get current size
                rect = wintypes.RECT()
                self.user32.GetWindowRect(hwnd, ctypes.byref(rect))
                width = width or (rect.right - rect.left)
                height = height or (rect.bottom - rect.top)
            
            self.user32.MoveWindow(hwnd, x, y, width, height, True)
            return True
        except:
            return False
    
    def resize_window(self, hwnd: int, width: int, height: int) -> bool:
        """Resize window"""
        try:
            rect = wintypes.RECT()
            self.user32.GetWindowRect(hwnd, ctypes.byref(rect))
            return self.move_window(hwnd, rect.left, rect.top, width, height)
        except:
            return False
    
    def tile_windows_horizontal(self, hwnds: List[int]) -> bool:
        """Tile windows horizontally"""
        if not hwnds:
            return False
        
        # Get screen size
        screen_width = self.user32.GetSystemMetrics(0)
        screen_height = self.user32.GetSystemMetrics(1)
        
        # Calculate tile dimensions
        tile_height = screen_height // len(hwnds)
        
        for i, hwnd in enumerate(hwnds):
            y = i * tile_height
            self.move_window(hwnd, 0, y, screen_width, tile_height)
        
        return True
    
    def tile_windows_vertical(self, hwnds: List[int]) -> bool:
        """Tile windows vertically"""
        if not hwnds:
            return False
        
        # Get screen size
        screen_width = self.user32.GetSystemMetrics(0)
        screen_height = self.user32.GetSystemMetrics(1)
        
        # Calculate tile dimensions
        tile_width = screen_width // len(hwnds)
        
        for i, hwnd in enumerate(hwnds):
            x = i * tile_width
            self.move_window(hwnd, x, 0, tile_width, screen_height)
        
        return True
    
    def snap_window_left(self, hwnd: int) -> bool:
        """Snap window to left half of screen"""
        screen_width = self.user32.GetSystemMetrics(0)
        screen_height = self.user32.GetSystemMetrics(1)
        return self.move_window(hwnd, 0, 0, screen_width // 2, screen_height)
    
    def snap_window_right(self, hwnd: int) -> bool:
        """Snap window to right half of screen"""
        screen_width = self.user32.GetSystemMetrics(0)
        screen_height = self.user32.GetSystemMetrics(1)
        return self.move_window(hwnd, screen_width // 2, 0, 
                               screen_width // 2, screen_height)
    
    def get_window_at_position(self, x: int, y: int) -> Optional[WindowInfo]:
        """Get window at screen coordinates"""
        point = wintypes.POINT(x, y)
        hwnd = self.user32.WindowFromPoint(point)
        if hwnd:
            return self._get_window_info(hwnd)
        return None
    
    def _get_window_info(self, hwnd: int) -> WindowInfo:
        """Get detailed window information"""
        title = self._get_window_title(hwnd)
        class_name = self._get_window_class(hwnd)
        
        rect = wintypes.RECT()
        self.user32.GetWindowRect(hwnd, ctypes.byref(rect))
        
        process_id = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        
        is_visible = bool(self.user32.IsWindowVisible(hwnd))
        is_minimized = bool(self.user32.IsIconic(hwnd))
        
        return WindowInfo(
            hwnd=hwnd,
            title=title,
            class_name=class_name,
            is_visible=is_visible,
            is_minimized=is_minimized,
            rect=(rect.left, rect.top, rect.right, rect.bottom),
            process_id=process_id.value
        )
    
    def _get_window_title(self, hwnd: int) -> str:
        """Get window title"""
        length = self.user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return ""
        
        buffer = ctypes.create_unicode_buffer(length + 1)
        self.user32.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value
    
    def _get_window_class(self, hwnd: int) -> str:
        """Get window class name"""
        buffer = ctypes.create_unicode_buffer(256)
        self.user32.GetClassNameW(hwnd, buffer, 256)
        return buffer.value


# Convenience functions
_manager = None

def get_manager() -> WindowManager:
    """Get shared WindowManager instance"""
    global _manager
    if _manager is None:
        _manager = WindowManager()
    return _manager


if __name__ == "__main__":
    # Test window management
    wm = WindowManager()
    
    print("All visible windows:")
    print("-" * 60)
    for window in wm.get_all_windows():
        print(f"  {window.title[:50]}")
        print(f"    Position: {window.position}, Size: {window.size}")
        print(f"    Minimized: {window.is_minimized}")
    
    print("\nActive window:")
    print("-" * 60)
    active = wm.get_active_window()
    if active:
        print(f"  {active.title}")
        print(f"  Position: {active.position}, Size: {active.size}")
    
    print("\nWindow management ready!")
