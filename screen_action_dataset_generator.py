"""
screen_action_dataset_generator.py
====================================
Generates rich screen action training data for Avus fine-tuning.

Covers:
- App launching (Win+R, Start menu, taskbar)
- Typing text into fields
- Multi-step tasks (open app → click → type → submit)
- Browser navigation
- File operations
- Window management

Format: <|startoftext|>{screen_description} [ACT_START]{action_json}[ACT_END]<|endoftext|>
"""

import json
import random
import os
from typing import List, Dict, Any
from pathlib import Path

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ri(lo, hi): return random.randint(lo, hi)
def _rc(lst):    return random.choice(lst)
def _rf(lo, hi): return round(random.uniform(lo, hi), 2)

APPS = [
    "Notepad", "Chrome", "Firefox", "VS Code", "File Explorer",
    "Discord", "Slack", "Terminal", "PowerShell", "Paint",
    "Calculator", "Task Manager", "Settings", "Word", "Excel",
]

WEBSITES = [
    "google.com", "github.com", "upwork.com", "fiverr.com",
    "stackoverflow.com", "youtube.com", "reddit.com",
]

BUTTONS = [
    "Submit", "Cancel", "Save", "Login", "Search", "Next", "Delete",
    "OK", "Apply", "Close", "Open", "New", "Edit", "Upload", "Download",
    "Sign In", "Sign Up", "Continue", "Back", "Confirm", "Send",
]

TEXT_INPUTS = [
    "hello world", "test message", "search query", "username",
    "my project", "important note", "task description",
]

# ── Single action generators ──────────────────────────────────────────────────

def gen_click(n: int) -> List[str]:
    """Click a button on screen."""
    out = []
    for _ in range(n):
        x, y = _ri(10, 1910), _ri(10, 1070)
        btn = _rc(BUTTONS)
        app = _rc(APPS)
        action = {"type": "click", "x": x, "y": y, "button": "left"}
        screen = f"{app} is open. A '{btn}' button is visible at ({x}, {y})."
        out.append(f"<|startoftext|>{screen} [ACT_START]{json.dumps(action)}[ACT_END]<|endoftext|>")
    return out


def gen_type(n: int) -> List[str]:
    """Type text into a field."""
    out = []
    for _ in range(n):
        x, y = _ri(100, 1800), _ri(100, 900)
        text = _rc(TEXT_INPUTS)
        app = _rc(APPS)
        field = _rc(["text field", "input box", "search bar", "text area", "address bar"])
        action = {"type": "type", "text": text}
        screen = f"{app} is open. A {field} is active at ({x}, {y}). Type: '{text}'."
        out.append(f"<|startoftext|>{screen} [ACT_START]{json.dumps(action)}[ACT_END]<|endoftext|>")
    return out


def gen_hotkey(n: int) -> List[str]:
    """Press a keyboard shortcut."""
    hotkeys = [
        (["ctrl", "c"], "copy selected text"),
        (["ctrl", "v"], "paste from clipboard"),
        (["ctrl", "s"], "save the file"),
        (["ctrl", "z"], "undo last action"),
        (["ctrl", "a"], "select all text"),
        (["ctrl", "n"], "open new window"),
        (["ctrl", "w"], "close current tab"),
        (["alt", "F4"], "close the application"),
        (["win", "r"], "open Run dialog"),
        (["ctrl", "t"], "open new tab"),
        (["F5"], "refresh the page"),
        (["enter"], "confirm the action"),
        (["escape"], "cancel or close dialog"),
        (["tab"], "move to next field"),
    ]
    out = []
    for _ in range(n):
        keys, desc = _rc(hotkeys)
        app = _rc(APPS)
        action = {"type": "hotkey", "keys": keys}
        screen = f"{app} is open. Need to {desc}."
        out.append(f"<|startoftext|>{screen} [ACT_START]{json.dumps(action)}[ACT_END]<|endoftext|>")
    return out


def gen_launch_app(n: int) -> List[str]:
    """Launch an application."""
    out = []
    for _ in range(n):
        app = _rc(APPS)
        # Win+R approach
        action = {"type": "hotkey", "keys": ["win", "r"]}
        screen = f"Windows desktop is visible. Need to open {app}."
        out.append(f"<|startoftext|>{screen} [ACT_START]{json.dumps(action)}[ACT_END]<|endoftext|>")

        # Type in run dialog
        action2 = {"type": "type", "text": app.lower().replace(" ", "")}
        screen2 = f"Run dialog is open. Need to launch {app}."
        out.append(f"<|startoftext|>{screen2} [ACT_START]{json.dumps(action2)}[ACT_END]<|endoftext|>")
    return out


def gen_scroll(n: int) -> List[str]:
    """Scroll the page."""
    out = []
    for _ in range(n):
        x, y = _ri(200, 1700), _ri(200, 800)
        direction = _rc(["up", "down"])
        amount = _ri(3, 10)
        app = _rc(APPS)
        action = {"type": "scroll", "x": x, "y": y, "direction": direction, "amount": amount}
        screen = f"{app} is open. Need to scroll {direction} to see more content."
        out.append(f"<|startoftext|>{screen} [ACT_START]{json.dumps(action)}[ACT_END]<|endoftext|>")
    return out


def gen_navigate_browser(n: int) -> List[str]:
    """Navigate to a URL."""
    out = []
    for _ in range(n):
        site = _rc(WEBSITES)
        url = f"https://www.{site}"
        # Click address bar
        action1 = {"type": "click", "x": 700, "y": 45, "button": "left"}
        screen1 = f"Chrome is open. Need to navigate to {url}. Address bar is at top."
        out.append(f"<|startoftext|>{screen1} [ACT_START]{json.dumps(action1)}[ACT_END]<|endoftext|>")

        # Type URL
        action2 = {"type": "type", "text": url}
        screen2 = f"Chrome address bar is selected. Need to type {url}."
        out.append(f"<|startoftext|>{screen2} [ACT_START]{json.dumps(action2)}[ACT_END]<|endoftext|>")

        # Press Enter
        action3 = {"type": "hotkey", "keys": ["enter"]}
        screen3 = f"Chrome address bar shows '{url}'. Need to navigate."
        out.append(f"<|startoftext|>{screen3} [ACT_START]{json.dumps(action3)}[ACT_END]<|endoftext|>")
    return out


# ── Multi-step task generators ────────────────────────────────────────────────

def gen_open_notepad_and_type(n: int) -> List[str]:
    """Full multi-step: open Notepad and type text."""
    out = []
    texts = ["hello", "hello world", "test", "meeting notes", "todo list"]
    for _ in range(n):
        text = _rc(texts)

        # Step 1: Win+R
        a1 = {"type": "hotkey", "keys": ["win", "r"]}
        out.append(f"<|startoftext|>Windows desktop. Goal: open Notepad and type '{text}'. Step 1: open Run dialog. [ACT_START]{json.dumps(a1)}[ACT_END]<|endoftext|>")

        # Step 2: type notepad
        a2 = {"type": "type", "text": "notepad"}
        out.append(f"<|startoftext|>Run dialog is open. Goal: open Notepad. Type 'notepad' in the field. [ACT_START]{json.dumps(a2)}[ACT_END]<|endoftext|>")

        # Step 3: press enter
        a3 = {"type": "hotkey", "keys": ["enter"]}
        out.append(f"<|startoftext|>Run dialog shows 'notepad'. Press Enter to launch. [ACT_START]{json.dumps(a3)}[ACT_END]<|endoftext|>")

        # Step 4: click text area
        a4 = {"type": "click", "x": 640, "y": 400, "button": "left"}
        out.append(f"<|startoftext|>Notepad is open. Click the text area to focus it. [ACT_START]{json.dumps(a4)}[ACT_END]<|endoftext|>")

        # Step 5: type text
        a5 = {"type": "type", "text": text}
        out.append(f"<|startoftext|>Notepad text area is active. Type '{text}'. [ACT_START]{json.dumps(a5)}[ACT_END]<|endoftext|>")

    return out


def gen_browser_search(n: int) -> List[str]:
    """Full multi-step: open browser and search."""
    queries = ["python tutorial", "how to code", "weather today", "news"]
    out = []
    for _ in range(n):
        query = _rc(queries)

        a1 = {"type": "hotkey", "keys": ["win", "r"]}
        out.append(f"<|startoftext|>Desktop. Goal: search for '{query}' in browser. Open Run dialog first. [ACT_START]{json.dumps(a1)}[ACT_END]<|endoftext|>")

        a2 = {"type": "type", "text": "chrome"}
        out.append(f"<|startoftext|>Run dialog open. Launch Chrome browser. [ACT_START]{json.dumps(a2)}[ACT_END]<|endoftext|>")

        a3 = {"type": "hotkey", "keys": ["enter"]}
        out.append(f"<|startoftext|>Run dialog shows 'chrome'. Press Enter. [ACT_START]{json.dumps(a3)}[ACT_END]<|endoftext|>")

        a4 = {"type": "click", "x": 700, "y": 45, "button": "left"}
        out.append(f"<|startoftext|>Chrome is open. Click address bar to search for '{query}'. [ACT_START]{json.dumps(a4)}[ACT_END]<|endoftext|>")

        a5 = {"type": "type", "text": query}
        out.append(f"<|startoftext|>Chrome address bar selected. Type search query '{query}'. [ACT_START]{json.dumps(a5)}[ACT_END]<|endoftext|>")

        a6 = {"type": "hotkey", "keys": ["enter"]}
        out.append(f"<|startoftext|>Chrome address bar shows '{query}'. Press Enter to search. [ACT_START]{json.dumps(a6)}[ACT_END]<|endoftext|>")

    return out


# ── Main generator ────────────────────────────────────────────────────────────

def generate_screen_action_dataset(
    total: int = 100_000,
    output_path: str = "training_data/screen_actions.jsonl",
) -> int:
    """
    Generate a rich screen action dataset and save to JSONL.

    Returns the number of examples generated.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    per_type = total // 10
    all_examples = []

    print(f"Generating {total:,} screen action examples...")

    all_examples += gen_click(per_type * 2)
    print(f"  clicks: {per_type * 2:,}")

    all_examples += gen_type(per_type)
    print(f"  type: {per_type:,}")

    all_examples += gen_hotkey(per_type)
    print(f"  hotkeys: {per_type:,}")

    all_examples += gen_launch_app(per_type // 2)
    print(f"  launch app: {per_type // 2 * 2:,}")  # 2 steps each

    all_examples += gen_scroll(per_type // 2)
    print(f"  scroll: {per_type // 2:,}")

    all_examples += gen_navigate_browser(per_type // 3)
    print(f"  browser nav: {per_type // 3 * 3:,}")  # 3 steps each

    all_examples += gen_open_notepad_and_type(per_type // 5)
    print(f"  open+type: {per_type // 5 * 5:,}")  # 5 steps each

    all_examples += gen_browser_search(per_type // 6)
    print(f"  browser search: {per_type // 6 * 6:,}")  # 6 steps each

    random.shuffle(all_examples)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps({"text": ex}) + "\n")

    print(f"\nSaved {len(all_examples):,} examples to {output_path}")
    return len(all_examples)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=100_000)
    parser.add_argument("--output", type=str, default="training_data/screen_actions.jsonl")
    args = parser.parse_args()
    n = generate_screen_action_dataset(args.total, args.output)
    print(f"Done: {n:,} examples")
