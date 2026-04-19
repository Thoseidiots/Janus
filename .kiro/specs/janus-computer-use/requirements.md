# Requirements Document

## Introduction

Janus Computer Use is the capability layer that allows Janus — the autonomous AI worker running on Windows — to interact with any desktop application or website exactly as a human would. Rather than relying on APIs, Janus will move the mouse, click buttons, type text, read what is on screen, and navigate any UI it encounters. This makes Janus truly platform-agnostic: it can use Upwork directly in a browser, fill in forms, play games, operate downloaded software, and perform any task a human operator could perform at a keyboard and monitor.

The system builds on Janus's existing ability to take screenshots and launch applications, adding a full input-output loop: see the screen → understand what is shown → decide what to do → act with mouse and keyboard → observe the result → repeat.

## Glossary

- **Janus**: The autonomous AI worker system that operates on a Windows desktop
- **Computer_Use_Engine**: The top-level module that coordinates all computer-use capabilities
- **Mouse_Controller**: The sub-system responsible for moving the cursor and issuing click events
- **Keyboard_Controller**: The sub-system responsible for typing text and sending key combinations
- **Screen_Reader**: The sub-system that captures the screen and extracts text via OCR
- **Visual_Detector**: The sub-system that locates UI elements (buttons, fields, icons) by analysing a screenshot
- **Window_Manager**: The sub-system that lists, focuses, resizes, and repositions application windows
- **Action_Planner**: The AI component that decides which action to take next given the current screen state
- **UI_Element**: A discrete interactive widget on screen (button, text field, checkbox, link, icon, etc.)
- **Screen_Region**: A rectangular area of the screen defined by (x, y, width, height) in pixels
- **OCR**: Optical Character Recognition — converting pixel content to machine-readable text
- **Key_Combination**: Two or more keys pressed simultaneously (e.g. Ctrl+C, Alt+F4, Win+D)
- **Scroll_Direction**: One of UP, DOWN, LEFT, RIGHT
- **Drag_Operation**: A mouse-down at a source coordinate followed by mouse-up at a destination coordinate
- **Wait_Condition**: A predicate on screen state that must become true before the next action proceeds
- **Action_Result**: The outcome of a single computer-use action, including success flag and any extracted data

## Requirements

---

### Requirement 1: Mouse Control

**User Story:** As Janus, I want to move the mouse cursor and issue click events anywhere on the screen, so that I can interact with any UI element in any application.

#### Acceptance Criteria

1. THE Mouse_Controller SHALL move the cursor to any (x, y) coordinate within the primary display bounds
2. WHEN a left-click action is requested, THE Mouse_Controller SHALL press and release the left mouse button at the specified coordinate
3. WHEN a right-click action is requested, THE Mouse_Controller SHALL press and release the right mouse button at the specified coordinate
4. WHEN a double-click action is requested, THE Mouse_Controller SHALL issue two left-click events at the specified coordinate within 200 ms
5. WHEN a click coordinate falls outside the primary display bounds, THE Mouse_Controller SHALL log the error and return a failed Action_Result without performing any click
6. THE Mouse_Controller SHALL complete any single click or move operation within 100 ms of the request being issued
7. WHEN human-like movement is requested, THE Mouse_Controller SHALL interpolate cursor movement along a smooth path rather than teleporting instantly

---

### Requirement 2: Keyboard Input

**User Story:** As Janus, I want to type text and send key combinations to the focused application, so that I can fill in forms, trigger shortcuts, and navigate UIs with the keyboard.

#### Acceptance Criteria

1. WHEN a type-text action is requested, THE Keyboard_Controller SHALL send each character in the provided string as a key-down/key-up event pair to the currently focused window
2. WHEN a Key_Combination is requested (e.g. Ctrl+C, Alt+Tab, Win+D), THE Keyboard_Controller SHALL press all modifier keys before the primary key and release them in reverse order
3. WHEN a single special key is requested (Enter, Tab, Escape, Backspace, arrow keys, function keys), THE Keyboard_Controller SHALL send the correct virtual key code for that key
4. WHEN a type-text action contains non-ASCII characters, THE Keyboard_Controller SHALL use the appropriate Unicode input method to deliver those characters correctly
5. IF the target window loses focus during a typing sequence, THE Keyboard_Controller SHALL pause, attempt to re-focus the window, and resume from the interrupted position
6. THE Keyboard_Controller SHALL support a configurable typing speed (characters per second) to simulate human-paced input when required
7. WHEN a typing action completes, THE Keyboard_Controller SHALL return an Action_Result indicating the number of characters successfully delivered

---

### Requirement 3: Screen Capture and OCR

**User Story:** As Janus, I want to capture the screen and extract all readable text from it, so that I can understand what is currently displayed in any application.

#### Acceptance Criteria

1. WHEN a screen-capture action is requested, THE Screen_Reader SHALL capture a full-resolution screenshot of the primary display and return it as an image object
2. WHEN a Screen_Region is specified, THE Screen_Reader SHALL capture only that region rather than the full display
3. WHEN OCR is requested on a captured image, THE Screen_Reader SHALL extract all visible text and return it with bounding-box coordinates for each word
4. WHEN OCR is requested, THE Screen_Reader SHALL complete text extraction within 3 seconds for a full 1080p screenshot
5. IF the screen capture fails (e.g. display driver error), THE Screen_Reader SHALL retry once after 500 ms and return a failed Action_Result if the retry also fails
6. THE Screen_Reader SHALL support English text extraction at minimum; WHERE additional language packs are installed, THE Screen_Reader SHALL use them automatically
7. WHEN OCR returns text, THE Screen_Reader SHALL include a confidence score (0.0–1.0) for each extracted word

---

### Requirement 4: Visual Element Detection

**User Story:** As Janus, I want to locate buttons, text fields, icons, and other UI elements by looking at the screen, so that I can click the right target without needing to know pixel coordinates in advance.

#### Acceptance Criteria

1. WHEN asked to find a UI_Element by label or description, THE Visual_Detector SHALL analyse the current screenshot and return the bounding-box coordinates of the best matching element
2. WHEN multiple matching elements are found, THE Visual_Detector SHALL return them ranked by confidence score, highest first
3. WHEN no matching element is found, THE Visual_Detector SHALL return an empty result set and log a warning rather than raising an exception
4. THE Visual_Detector SHALL detect common UI element types: buttons, text input fields, checkboxes, radio buttons, dropdown menus, links, icons, and scrollbars
5. WHEN a template image is provided, THE Visual_Detector SHALL use template matching to locate that exact image on screen and return its position
6. WHEN element detection is requested, THE Visual_Detector SHALL complete the search within 2 seconds for a full 1080p screenshot
7. WHEN a UI_Element is located, THE Visual_Detector SHALL return the centre coordinate of the element's bounding box as the recommended click target

---

### Requirement 5: Wait for UI Elements

**User Story:** As Janus, I want to wait for specific UI elements or text to appear on screen before proceeding, so that I can handle loading states, animations, and asynchronous UI updates reliably.

#### Acceptance Criteria

1. WHEN a Wait_Condition is specified, THE Computer_Use_Engine SHALL poll the screen at a configurable interval (default 500 ms) until the condition is satisfied or a timeout is reached
2. WHEN the Wait_Condition is satisfied before the timeout, THE Computer_Use_Engine SHALL return a successful Action_Result immediately
3. WHEN the timeout is reached before the Wait_Condition is satisfied, THE Computer_Use_Engine SHALL return a failed Action_Result with a descriptive timeout message
4. THE Computer_Use_Engine SHALL support the following Wait_Condition types: element-visible, element-gone, text-present, text-gone, and image-present
5. WHEN a wait operation is active, THE Computer_Use_Engine SHALL not block other non-UI operations running concurrently
6. THE Computer_Use_Engine SHALL accept a configurable timeout value per wait call, with a default of 30 seconds and a maximum of 300 seconds

---

### Requirement 6: Scroll and Drag

**User Story:** As Janus, I want to scroll within windows and drag elements from one position to another, so that I can navigate long pages and interact with drag-and-drop interfaces.

#### Acceptance Criteria

1. WHEN a scroll action is requested, THE Mouse_Controller SHALL send scroll wheel events at the specified coordinate in the specified Scroll_Direction
2. WHEN a scroll amount is specified in lines, THE Mouse_Controller SHALL convert lines to the platform's native scroll unit and deliver the correct number of scroll events
3. WHEN a Drag_Operation is requested, THE Mouse_Controller SHALL press the left button at the source coordinate, move to the destination coordinate along a smooth path, and release the button
4. WHEN a Drag_Operation destination is outside the display bounds, THE Mouse_Controller SHALL clamp the destination to the nearest valid coordinate and log a warning
5. THE Mouse_Controller SHALL complete a Drag_Operation within 2 seconds for any drag distance up to the full screen diagonal
6. WHEN a horizontal scroll is requested, THE Mouse_Controller SHALL send horizontal scroll events rather than vertical scroll events

---

### Requirement 7: Window Management

**User Story:** As Janus, I want to list open windows, bring a specific window to the foreground, and resize or reposition windows, so that I can organise the desktop and ensure the correct application is active before interacting with it.

#### Acceptance Criteria

1. WHEN a list-windows action is requested, THE Window_Manager SHALL return a list of all visible top-level windows, each with: window handle, title, process name, and bounding rectangle
2. WHEN a focus-window action is requested, THE Window_Manager SHALL bring the specified window to the foreground and give it keyboard focus
3. WHEN a resize-window action is requested, THE Window_Manager SHALL set the window's dimensions to the specified width and height in pixels
4. WHEN a move-window action is requested, THE Window_Manager SHALL reposition the window's top-left corner to the specified (x, y) coordinate
5. WHEN a minimise-window action is requested, THE Window_Manager SHALL minimise the specified window to the taskbar
6. WHEN a maximise-window action is requested, THE Window_Manager SHALL maximise the specified window to fill the primary display
7. IF a window handle is no longer valid (window was closed), THE Window_Manager SHALL return a failed Action_Result with a descriptive message rather than raising an unhandled exception
8. WHEN a window title pattern is provided instead of a handle, THE Window_Manager SHALL find the first window whose title matches the pattern (case-insensitive substring match) and operate on it

---

### Requirement 8: Action Planning and Screen Understanding

**User Story:** As Janus, I want an AI-driven planner that looks at the current screen and decides what action to take next to accomplish a given goal, so that I can navigate complex multi-step UI flows without hard-coded scripts.

#### Acceptance Criteria

1. WHEN a high-level goal is provided, THE Action_Planner SHALL capture the current screen, analyse its content, and produce a ranked list of candidate next actions
2. WHEN producing candidate actions, THE Action_Planner SHALL include for each: action type, target coordinate or element description, parameters, and a confidence score
3. WHEN the top-ranked action is executed and the screen changes, THE Action_Planner SHALL re-capture the screen and re-evaluate progress toward the goal
4. WHEN the goal is determined to be achieved (success condition met on screen), THE Action_Planner SHALL return a completed Action_Result with a summary of steps taken
5. WHEN the Action_Planner has attempted more than a configurable maximum number of steps (default 50) without achieving the goal, THE Action_Planner SHALL return a failed Action_Result with the last known screen state
6. WHEN an action fails, THE Action_Planner SHALL record the failure, choose an alternative action, and continue rather than aborting the entire goal
7. THE Action_Planner SHALL maintain a step history (action taken, screenshot before, screenshot after, outcome) for the duration of a goal execution session

---

### Requirement 9: Error Recovery and Safety

**User Story:** As Janus, I want the computer-use system to recover from unexpected UI states and avoid destructive actions, so that it can operate reliably without human supervision.

#### Acceptance Criteria

1. WHEN an action produces an unexpected screen state (e.g. an error dialog appears), THE Computer_Use_Engine SHALL detect the dialog, attempt to dismiss it, and retry the previous action
2. WHEN a confirmation dialog appears before a potentially destructive action (delete, format, uninstall), THE Computer_Use_Engine SHALL pause and log the dialog text before proceeding, unless the action was explicitly pre-approved
3. WHEN the Computer_Use_Engine detects it has navigated to an unintended application or website, THE Computer_Use_Engine SHALL stop further actions and return a failed Action_Result with the current screen state
4. IF three consecutive actions fail to change the screen state, THE Computer_Use_Engine SHALL treat this as a stuck state, capture a screenshot, and return a failed Action_Result
5. THE Computer_Use_Engine SHALL log every action taken (type, target, timestamp, outcome) to a persistent action log
6. WHEN an action log entry is written, THE Computer_Use_Engine SHALL include the screenshot taken immediately before the action as a base64-encoded thumbnail

---

### Requirement 10: Integration with Janus Autonomous Worker

**User Story:** As Janus, I want the computer-use capabilities to be accessible from the existing autonomous worker and automation platform, so that I can use any website or desktop app as part of my job execution workflow.

#### Acceptance Criteria

1. THE Computer_Use_Engine SHALL expose a Python API that the existing `janus_autonomous_worker.py` and `janus_automation_platform.py` modules can import and call without modification to their core logic
2. WHEN the Computer_Use_Engine is imported, THE Computer_Use_Engine SHALL check for required system dependencies (pyautogui, pytesseract, Pillow, pygetwindow) and raise a descriptive ImportError listing any missing packages if they are absent
3. WHEN Janus needs to interact with the Upwork website directly (no API key available), THE Computer_Use_Engine SHALL provide a `BrowserComputerUse` helper that opens a browser, navigates to the URL, and exposes high-level actions (login, search_jobs, apply_to_job, submit_work)
4. WHEN a computer-use session starts, THE Computer_Use_Engine SHALL accept an optional `context` dictionary carrying Janus's current job, credentials reference, and goal description, and make this context available to the Action_Planner
5. THE Computer_Use_Engine SHALL be usable as an async context manager so that resources (screenshot threads, OCR engine) are properly initialised on entry and released on exit
6. WHEN the Computer_Use_Engine is running, THE Computer_Use_Engine SHALL emit structured log events compatible with the existing Janus monitoring system so that computer-use actions appear in the same log stream as other Janus activities
