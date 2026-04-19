# Janus Computer Use

Capability layer that gives the Janus autonomous worker full human-like desktop control on Windows. It closes the loop between Janus's existing screenshot capability and full input control: see the screen → understand what is shown → decide what to do → act with mouse and keyboard → observe the result → repeat.

---

## Prerequisites

### 1. Tesseract OCR engine

Tesseract must be installed separately before the Python packages. Run the following command in an elevated PowerShell or Windows Terminal:

```powershell
winget install UB-Mannheim.TesseractOCR
```

The `pytesseract` wrapper auto-detects the default install path (`C:\Program Files\Tesseract-OCR\tesseract.exe`). If you install Tesseract to a custom location, set the path explicitly in your code:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\custom\path\tesseract.exe"
```

### 2. Python packages

Install all required Python packages with:

```bash
pip install -r requirements_computer_use.txt
```

Or install them individually:

```bash
pip install pyautogui>=0.9.54 pytesseract>=0.3.10 Pillow>=10.0.0 pygetwindow>=0.0.9 opencv-python>=4.8.0 imagehash>=4.3.1
```

---

## System Requirements

- **OS**: Windows 10 or Windows 11 (win32 API required)
- **Python**: 3.10 or later
- **Display**: Primary display must be active (not headless) for mouse/keyboard control

---

## Quick Start

```python
import asyncio
from janus_computer_use import ComputerUseEngine

async def main():
    async with ComputerUseEngine(context={"goal": "Open Notepad and type Hello"}) as engine:
        result = await engine.run_goal("Open Notepad and type Hello")
        print("Success:", result.success)

asyncio.run(main())
```

---

## Running Tests

```bash
pip install pytest pytest-asyncio hypothesis
pytest tests/ --tb=short
```

---

## Safety Notes

- pyautogui's **FAILSAFE** is enabled by default: moving the mouse to the top-left corner of the screen raises a `FailSafeException`, which `ComputerUseEngine` catches and converts to a failed `ActionResult`.
- Destructive actions (delete, format, uninstall, etc.) are paused automatically unless explicitly pre-approved via `action.pre_approved = True`.
- The engine detects stuck states (3 consecutive actions with no screen change) and stops automatically.
