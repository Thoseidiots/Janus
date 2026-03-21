# “””
screen_action_dataset_generator.py

Synthetic (screen_description, action_json) dataset generator for training
Avus to interpret screen state and output desktop control actions.

No real screenshots needed. Generates thousands of described UI states paired
with the correct JSON action — same philosophy as Grade3DGeneration.

Output format (matches skill_executor action types):
<|startoftext|>{screen_description} [ACT_START] {action_json} [ACT_END]<|endoftext|>

Action types produced:
click, double_click, right_click, type, key, press_enter, scroll, wait, mouse_move

Usage:
gen = ScreenActionDataset()
pairs = gen.generate_dataset(samples=10_000)

# pairs -> List[Tuple[str, str]]  (description, json_string)

“””

import json
import random
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────────────────────

# Vocabulary pools

# ─────────────────────────────────────────────────────────────────────────────

APPS = [
“Chrome”, “Firefox”, “Notepad”, “VS Code”, “File Explorer”,
“Discord”, “Slack”, “Terminal”, “Paint”, “Excel”,
“Word”, “Outlook”, “Task Manager”, “Settings”, “Calculator”,
]

WEBSITES = [
“Google”, “YouTube”, “GitHub”, “Fiverr”, “Reddit”,
“Twitter”, “LinkedIn”, “Gmail”, “Wikipedia”, “Stack Overflow”,
]

BUTTON_LABELS = [
“Submit”, “Cancel”, “OK”, “Close”, “Save”,
“Login”, “Sign Up”, “Search”, “Next”, “Back”,
“Download”, “Upload”, “Delete”, “Edit”, “Confirm”,
“Apply”, “Reset”, “Send”, “Continue”, “Finish”,
]

INPUT_FIELDS = [
“username”, “password”, “email”, “search bar”, “message box”,
“title field”, “description field”, “URL bar”, “file name field”,
“phone number field”, “zip code field”, “comment box”,
]

MENU_ITEMS = [
“File > Save”, “File > Open”, “File > New”, “Edit > Copy”,
“Edit > Paste”, “Edit > Undo”, “View > Zoom In”, “View > Zoom Out”,
“Tools > Settings”, “Help > About”,
]

KEYBOARD_ACTIONS = [
(“press Ctrl+C”, 0x43, “copy”),
(“press Ctrl+V”, 0x56, “paste”),
(“press Ctrl+Z”, 0x5A, “undo”),
(“press Ctrl+S”, 0x53, “save”),
(“press Escape”,  0x1B, “dismiss”),
(“press Tab”,     0x09, “tab”),
(“press Delete”,  0x2E, “delete selected”),
(“press Ctrl+A”,  0x41, “select all”),
(“press Ctrl+F”,  0x46, “open find”),
(“press F5”,      0x74, “refresh”),
]

SCROLL_CONTEXTS = [
“a long webpage”, “a list of search results”, “a file directory”,
“a chat history”, “a code file”, “a settings panel”,
“a dropdown menu”, “an image gallery”, “a log output”,
]

WAIT_REASONS = [
“a loading spinner is visible”, “a progress bar is at 60%”,
“a page is still loading”, “a download is in progress”,
“an animation is playing”, “a video is buffering”,
]

DRAG_TARGETS = [
“a file icon”, “a window title bar”, “a slider handle”,
“a list item”, “an image thumbnail”,
]

TEXT_TO_TYPE = [
“Hello, world!”, “search query”, “my username”, “a short note”,
“the file name”, “an email address”, “a URL”, “a password”,
“Yes”, “No”, “Done”, “1234”, “test input”,
]

# ─────────────────────────────────────────────────────────────────────────────

# Coordinate helpers

# ─────────────────────────────────────────────────────────────────────────────

# Common screen resolutions to sample from

RESOLUTIONS = [
(1920, 1080), (1366, 768), (2560, 1440), (1280, 720), (1600, 900)
]

def _res():
return random.choice(RESOLUTIONS)

def _coord(w, h):
return random.randint(10, w - 10), random.randint(10, h - 10)

def _rc(lst):  return random.choice(lst)
def _ri(a, b): return random.randint(a, b)
def _rf(a, b): return round(random.uniform(a, b), 2)

# ─────────────────────────────────────────────────────────────────────────────

# ScreenActionDataset

# ─────────────────────────────────────────────────────────────────────────────

class ScreenActionDataset:
“””
Generates synthetic (screen_description, action_json) pairs.

```
Each sample is a realistic description of what Janus 'sees' on screen,
paired with the correct action to take — formatted as JSON matching
skill_executor's action dispatch.
"""

def __init__(self):
    self._generators = [
        self._click_button,
        self._double_click_item,
        self._right_click_item,
        self._type_into_field,
        self._press_keyboard_shortcut,
        self._press_enter_to_confirm,
        self._scroll_page,
        self._wait_for_load,
        self._click_menu_item,
        self._move_to_element,
        self._click_link,
        self._dismiss_dialog,
        self._select_all_and_type,
        self._click_tab,
        self._scroll_to_find,
    ]

# ── public ────────────────────────────────────────────────────────────────

def generate_dataset(self, samples: int = 10_000, seed: int = 42
                     ) -> List[Tuple[str, str]]:
    """
    Returns a list of (screen_description, action_json_string) tuples.
    """
    random.seed(seed)
    out = []
    for _ in range(samples):
        fn = _rc(self._generators)
        desc, action = fn()
        out.append((desc, json.dumps(action)))
    return out

# ── scenario generators ───────────────────────────────────────────────────

def _click_button(self):
    w, h   = _res()
    app    = _rc(APPS)
    label  = _rc(BUTTON_LABELS)
    x, y   = _coord(w, h)
    desc = (f"The screen shows {app}. "
            f"There is a '{label}' button visible at approximately ({x}, {y}). "
            f"The task requires clicking it to proceed.")
    action = {"type": "click", "x": x, "y": y, "button": "left"}
    return desc, action

def _double_click_item(self):
    w, h  = _res()
    app   = _rc(["File Explorer", "Desktop", "VS Code", "Excel"])
    item  = _rc(["a folder", "a file", "a shortcut icon",
                  "a spreadsheet cell", "a text file"])
    x, y  = _coord(w, h)
    desc = (f"{app} is open. "
            f"There is {item} at position ({x}, {y}). "
            f"It needs to be opened with a double-click.")
    action = {"type": "double_click", "x": x, "y": y}
    return desc, action

def _right_click_item(self):
    w, h   = _res()
    target = _rc(["a file icon", "a folder", "the desktop background",
                   "a taskbar icon", "an image"])
    x, y   = _coord(w, h)
    desc = (f"The screen shows the desktop or file manager. "
            f"There is {target} at ({x}, {y}). "
            f"A context menu is needed — right-click it.")
    action = {"type": "right_click", "x": x, "y": y}
    return desc, action

def _type_into_field(self):
    w, h  = _res()
    field = _rc(INPUT_FIELDS)
    text  = _rc(TEXT_TO_TYPE)
    x, y  = _coord(w, h)
    app   = _rc(APPS + WEBSITES)
    desc = (f"{app} is open with a {field} visible and focused at ({x}, {y}). "
            f"The field is empty and waiting for input. "
            f"Type: \"{text}\".")
    action = {"type": "type", "text": text, "x": x, "y": y}
    return desc, action

def _press_keyboard_shortcut(self):
    w, h          = _res()
    label, vk, purpose = _rc(KEYBOARD_ACTIONS)
    app           = _rc(APPS)
    desc = (f"{app} is in focus. "
            f"The current task requires {purpose}. "
            f"Use the keyboard: {label}.")
    action = {"type": "key", "vk_code": vk, "label": label}
    return desc, action

def _press_enter_to_confirm(self):
    w, h  = _res()
    ctx   = _rc(["a dialog box is open asking to confirm",
                  "a search field has been filled in",
                  "a form is complete and ready to submit",
                  "a rename field is active with new text entered",
                  "a terminal command has been typed"])
    desc = (f"The screen shows {ctx}. "
            f"Press Enter to confirm or execute.")
    action = {"type": "press_enter"}
    return desc, action

def _scroll_page(self):
    direction = _rc(["down", "up"])
    amount    = _ri(2, 8)
    ctx       = _rc(SCROLL_CONTEXTS)
    goal      = _rc(["find a specific item", "read more content",
                      "reach the bottom", "find a button",
                      "see earlier messages"])
    desc = (f"The screen is showing {ctx}. "
            f"Need to scroll {direction} {amount} times to {goal}.")
    action = {"type": "scroll", "direction": direction, "amount": amount}
    return desc, action

def _wait_for_load(self):
    duration = _rf(0.5, 3.0)
    reason   = _rc(WAIT_REASONS)
    desc = (f"The screen shows {reason}. "
            f"Wait {duration} seconds before taking the next action.")
    action = {"type": "wait", "duration": duration}
    return desc, action

def _click_menu_item(self):
    w, h  = _res()
    menu  = _rc(MENU_ITEMS)
    app   = _rc(APPS)
    x, y  = _coord(w, h)
    desc = (f"{app} is open with the menu bar visible. "
            f"The menu item '{menu}' is highlighted at approximately ({x}, {y}). "
            f"Click it.")
    action = {"type": "click", "x": x, "y": y, "button": "left",
              "context": f"menu:{menu}"}
    return desc, action

def _move_to_element(self):
    w, h   = _res()
    target = _rc(DRAG_TARGETS)
    x, y   = _coord(w, h)
    desc = (f"The cursor needs to move to {target} at ({x}, {y}) "
            f"before performing the next action.")
    action = {"type": "mouse_move", "x": x, "y": y}
    return desc, action

def _click_link(self):
    w, h = _res()
    site = _rc(WEBSITES)
    link = _rc(["a blue hyperlink", "a navigation item", "a search result",
                 "a 'Read more' link", "a profile link", "an article title"])
    x, y = _coord(w, h)
    desc = (f"A {site} page is open in the browser. "
            f"There is {link} visible at ({x}, {y}). "
            f"Click it to navigate.")
    action = {"type": "click", "x": x, "y": y, "button": "left",
              "context": f"link:{site}"}
    return desc, action

def _dismiss_dialog(self):
    w, h    = _res()
    dialog  = _rc(["an error dialog", "a confirmation popup",
                    "a save prompt", "an update notification",
                    "a cookie consent banner"])
    btn     = _rc(["OK", "Close", "Dismiss", "Cancel", "No Thanks"])
    x, y    = _coord(w, h)
    desc = (f"The screen has {dialog} overlaying the main window. "
            f"A '{btn}' button is visible at ({x}, {y}). "
            f"Dismiss it.")
    action = {"type": "click", "x": x, "y": y, "button": "left",
              "context": f"dialog_dismiss:{btn}"}
    return desc, action

def _select_all_and_type(self):
    w, h  = _res()
    field = _rc(INPUT_FIELDS)
    text  = _rc(TEXT_TO_TYPE)
    x, y  = _coord(w, h)
    app   = _rc(APPS)
    desc = (f"{app} shows a {field} at ({x}, {y}) with existing text. "
            f"Select all existing content and replace it with: \"{text}\".")
    # Two-action scenario encoded as a sequence note in context
    action = {"type": "key", "vk_code": 0x41, "label": "Ctrl+A",
              "then": {"type": "type", "text": text}}
    return desc, action

def _click_tab(self):
    w, h = _res()
    tab  = _rc(["Settings", "Profile", "Home", "Notifications",
                 "Messages", "Dashboard", "Analytics", "Help"])
    x, y = _coord(w, h)
    app  = _rc(APPS + WEBSITES)
    desc = (f"{app} is open. "
            f"There is a '{tab}' tab in the navigation bar at ({x}, {y}). "
            f"Click it to switch views.")
    action = {"type": "click", "x": x, "y": y, "button": "left",
              "context": f"tab:{tab}"}
    return desc, action

def _scroll_to_find(self):
    direction = _rc(["down", "up"])
    amount    = _ri(3, 10)
    target    = _rc(BUTTON_LABELS + INPUT_FIELDS)
    ctx       = _rc(SCROLL_CONTEXTS)
    desc = (f"The screen shows {ctx}. "
            f"The '{target}' element is not yet visible. "
            f"Scroll {direction} {amount} times to locate it.")
    action = {"type": "scroll", "direction": direction, "amount": amount,
              "target": target}
    return desc, action
```

# ─────────────────────────────────────────────────────────────────────────────

# Quick test

# ─────────────────────────────────────────────────────────────────────────────

if **name** == “**main**”:
gen   = ScreenActionDataset()
pairs = gen.generate_dataset(samples=10_000)

```
print(f"Generated {len(pairs)} samples\n")
print("── Sample outputs ──────────────────────────────────────")
for i in random.sample(range(len(pairs)), 6):
    desc, action = pairs[i]
    print(f"\n[{i}] SCREEN:\n  {desc}")
    print(f"     ACTION:\n  {action}")

# Verify every action type present
types = set()
for _, a in pairs:
    types.add(json.loads(a).get("type"))
print(f"\nAction types covered: {sorted(types)}")
```