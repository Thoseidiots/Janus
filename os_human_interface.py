import ctypes
import ctypes.wintypes as wintypes
import time
import subprocess
import struct

# Windows Constants
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_ABSOLUTE = 0x8000
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_RETURN = 0x0D

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class INPUT_I(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]

class INPUT(ctypes.Structure):
    _anonymous_ = ("ii",)
    _fields_ = [("type", wintypes.DWORD), ("ii", INPUT_I)]

class JanusOS:
    def __init__(self):
        self.user32 = ctypes.windll.user32
        self.gdi32 = ctypes.windll.gdi32
        self.kernel32 = ctypes.windll.kernel32
        
    def get_screen_size(self):
        return self.user32.GetSystemMetrics(0), self.user32.GetSystemMetrics(1)

    def move_mouse(self, x, y):
        w, h = self.get_screen_size()
        nx = int(x * 65535 / w)
        ny = int(y * 65535 / h)
        inp = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(nx, ny, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, 0, None))
        self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def click(self, x, y, button='left'):
        self.move_mouse(x, y)
        down = MOUSEEVENTF_LEFTDOWN if button == 'left' else MOUSEEVENTF_RIGHTDOWN
        up = MOUSEEVENTF_LEFTUP if button == 'left' else MOUSEEVENTF_RIGHTUP
        self.user32.SendInput(1, ctypes.byref(INPUT(0, INPUT_I(mi=MOUSEINPUT(0,0,0,down,0,None)))), ctypes.sizeof(INPUT))
        time.sleep(0.05)
        self.user32.SendInput(1, ctypes.byref(INPUT(0, INPUT_I(mi=MOUSEINPUT(0,0,0,up,0,None)))), ctypes.sizeof(INPUT))

    def press_key(self, vk_code):
        self.user32.keybd_event(vk_code, 0, 0, 0)
        self.user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)

    def type_string(self, text):
        for char in text:
            vk = self.user32.VkKeyScanW(ord(char))
            if vk == -1: continue
            self.user32.keybd_event(vk & 0xFF, 0, 0, 0)
            self.user32.keybd_event(vk & 0xFF, 0, KEYEVENTF_KEYUP, 0)

    def capture_screen(self):
        w, h = self.get_screen_size()
        hwnd = self.user32.GetDesktopWindow()
        hdc_screen = self.user32.GetDC(hwnd)
        hdc_mem = self.gdi32.CreateCompatibleDC(hdc_screen)
        hbmp = self.gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
        self.gdi32.SelectObject(hdc_mem, hbmp)
        self.gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, 0, 0, 0x00CC0020)
        
        bmi = struct.pack('<LllHHLLllLL', 40, w, -h, 1, 32, 0, w*h*4, 0, 0, 0, 0)
        buf = (ctypes.c_ubyte * (w * h * 4))()
        self.gdi32.GetDIBits(hdc_mem, hbmp, 0, h, buf, bmi, 0)
        
        self.gdi32.DeleteObject(hbmp)
        self.gdi32.DeleteDC(hdc_mem)
        self.user32.ReleaseDC(hwnd, hdc_screen)
        return bytes(buf), w, h

    def get_clipboard(self):
        if not self.user32.OpenClipboard(None): return ""
        h_data = self.user32.GetClipboardData(13) # CF_UNICODETEXT
        if not h_data:
            self.user32.CloseClipboard()
            return ""
        ptr = self.kernel32.GlobalLock(h_data)
        text = ctypes.c_wchar_p(ptr).value
        self.kernel32.GlobalUnlock(h_data)
        self.user32.CloseClipboard()
        return text
