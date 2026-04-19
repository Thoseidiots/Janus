# Implementation Plan: Janus Computer Use

## Overview

Implement `janus_computer_use.py` — a Python module that gives Janus full human-like desktop control on Windows. The module provides mouse control, keyboard input, OCR-based screen reading, visual element detection, window management, an AI action planner, a browser helper, and structured logging. All components are async-compatible and integrate with the existing `janus_autonomous_worker.py` and `janus_automation_platform.py` without modifying their core logic.

Implementation language: **Python** (asyncio, dataclasses, pytest-hypothesis for property tests).

## Tasks

- [x] 1. Project setup and dependency scaffolding
  - Create `janus_computer_use.py` with module docstring and top-level imports
  - Add `_check_dependencies()` function that imports each required package and raises a descriptive `ImportError` listing all missing packages if any are absent
  - Add `requirements_computer_use.txt` listing pinned versions: `pyautogui>=0.9.54`, `pytesseract>=0.3.10`, `Pillow>=10.0.0`, `pygetwindow>=0.0.9`, `opencv-python>=4.8.0`, `imagehash>=4.3.1`, `pytest-hypothesis>=6.0.0`
  - Add a `README_computer_use.md` section with Tesseract install instructions (`winget install UB-Mannheim.TesseractOCR`) and the pip install command
  - Call `_check_dependencies()` at module import time so missing packages are caught immediately
  - _Requirements: 10.1, 10.2_

- [x] 2. Data models — enums and dataclasses
  - [x] 2.1 Implement all enums and dataclasses in `janus_computer_use.py`
    - Implement `ActionType`, `ScrollDirection`, `WaitConditionType` enums
    - Implement `ScreenRegion`, `UIElement`, `OCRWord`, `WaitCondition`, `Action`, `ActionResult`, `CandidateAction`, `StepRecord`, `WindowInfo`, `ActionLogEntry` dataclasses exactly as specified in the design document
    - Ensure `ActionResult.timestamp` defaults to `datetime.datetime.utcnow()`
    - _Requirements: 1.1–1.7, 2.1–2.7, 3.1–3.7, 4.1–4.7, 6.1–6.6, 7.1–7.8, 8.1–8.7, 9.1–9.6_

  - [x] 2.2 Write property test — UIElement center equals bounding box center (Property 10)
    - **Property 10: UIElement center equals bounding box center**
    - **Validates: Requirements 4.7**
    - Use `@given(st.integers(), st.integers(), st.integers(min_value=1), st.integers(min_value=1))` to generate arbitrary bounding boxes
    - Assert `element.center == (bb.x + bb.width // 2, bb.y + bb.height // 2)` for every generated element

- [x] 3. MouseController implementation
  - [x] 3.1 Implement `MouseController` class
    - On `__init__`, call `pyautogui.size()` to store `(screen_width, screen_height)`
    - Implement `_validate_coords(x, y)` — returns `False` and logs error if `x < 0`, `y < 0`, `x >= screen_width`, or `y >= screen_height`
    - Implement `async move(x, y, human_like=False)` — use `asyncio.to_thread`; when `human_like=True` use `pyautogui.moveTo` with `duration` and `tween=pyautogui.easeInOutQuad`; reject out-of-bounds coords
    - Implement `async click(x, y, button="left")` — validate coords, then `pyautogui.click`
    - Implement `async double_click(x, y)` — validate coords, then `pyautogui.doubleClick`
    - Implement `async right_click(x, y)` — validate coords, then `pyautogui.rightClick`
    - Implement `async scroll(x, y, direction, amount=3)` — map `ScrollDirection` to pyautogui scroll calls; support horizontal scroll via `pyautogui.hscroll`
    - Implement `async drag(src_x, src_y, dst_x, dst_y)` — clamp destination to screen bounds (log warning if clamped), then `pyautogui.dragTo`
    - Catch `pyautogui.FailSafeException` in all methods and return a failed `ActionResult`
    - _Requirements: 1.1–1.7, 6.1–6.6_

  - [x] 3.2 Write property test — out-of-bounds coordinates are always rejected (Property 1)
    - **Property 1: Out-of-bounds coordinates are always rejected**
    - **Validates: Requirements 1.5**
    - Use `@given` with coordinates outside `[0, screen_width) × [0, screen_height)` (mock `pyautogui.size()` to return `(1920, 1080)`)
    - Assert `result.success == False` and that no pyautogui mouse call was made

  - [x] 3.3 Write property test — human-like movement visits intermediate points (Property 2)
    - **Property 2: Human-like movement visits intermediate points**
    - **Validates: Requirements 1.7**
    - Mock `pyautogui.moveTo` to capture the `duration` argument; assert `duration > 0` when `human_like=True` and distance > 1 pixel

  - [x] 3.4 Write property test — drag destination clamping stays within bounds (Property 11)
    - **Property 11: Drag destination clamping stays within bounds**
    - **Validates: Requirements 6.4**
    - Use `@given` with out-of-bounds drag destinations; assert clamped coords satisfy `0 <= x < 1920` and `0 <= y < 1080`

- [x] 4. KeyboardController implementation
  - [x] 4.1 Implement `KeyboardController` class
    - `__init__(self, typing_speed_cps=30.0)` — store speed; compute `interval = 1.0 / typing_speed_cps`
    - Implement `async type_text(text)` — iterate characters, call `pyautogui.typewrite` per character with `interval`; handle non-ASCII via `pyautogui.write` with Unicode fallback using `pyperclip` paste for characters outside ASCII range; return `ActionResult` with `chars_delivered=len(text)`
    - Implement `async press_key(key)` — `pyautogui.press(key)` via `asyncio.to_thread`
    - Implement `async hotkey(*keys)` — `pyautogui.hotkey(*keys)` via `asyncio.to_thread`
    - Implement `async key_combination(modifiers, key)` — press modifiers in order, press key, release key, release modifiers in reverse order using `pyautogui.keyDown`/`keyUp`
    - _Requirements: 2.1–2.7_

  - [x] 4.2 Write property test — typing delivers all characters (Property 4)
    - **Property 4: Typing delivers all characters**
    - **Validates: Requirements 2.1, 2.4, 2.7**
    - Use `@given(st.text(min_size=1))` with mocked `pyautogui`; assert `result.chars_delivered == len(text)` and `result.success == True`

  - [x] 4.3 Write property test — configurable typing speed sets inter-key delay (Property 5)
    - **Property 5: Configurable typing speed sets inter-key delay**
    - **Validates: Requirements 2.6**
    - Use `@given(st.floats(min_value=0.1, max_value=200.0))` for speed; assert the `interval` passed to pyautogui equals `1/speed` within 10% tolerance

  - [x] 4.4 Write property test — key combination ordering invariant (Property 3)
    - **Property 3: Key combination ordering invariant**
    - **Validates: Requirements 2.2**
    - Use `@given` with lists of modifier keys and a primary key; capture `keyDown`/`keyUp` call order; assert all modifiers are pressed before the primary key and released after in reverse order

- [x] 5. ScreenReader implementation
  - [x] 5.1 Implement `ScreenReader` class
    - Implement `async capture(region=None)` — use `asyncio.to_thread(PIL.ImageGrab.grab, bbox)` where `bbox = (region.x, region.y, region.x+region.width, region.y+region.height)` if region is given, else `None`; retry once after 500 ms on failure; return `PIL.Image`
    - Implement `async ocr(image)` — call `asyncio.to_thread(pytesseract.image_to_data, image, output_type=Output.DICT)`; parse result into `List[OCRWord]` with `confidence` clamped to `[0.0, 1.0]` (divide raw 0–100 score by 100); filter out words with empty text
    - Implement `async capture_and_ocr(region=None)` — compose `capture` then `ocr`
    - _Requirements: 3.1–3.7_

  - [x] 5.2 Write property test — region capture returns correctly sized image (Property 6)
    - **Property 6: Region capture returns correctly sized image**
    - **Validates: Requirements 3.2**
    - Use `@given(st.integers(0,1900), st.integers(0,1060), st.integers(1,100), st.integers(1,100))` for region; mock `PIL.ImageGrab.grab` to return a correctly sized image; assert `img.width == region.width` and `img.height == region.height`

  - [x] 5.3 Write property test — OCR confidence scores are always in [0.0, 1.0] (Property 7)
    - **Property 7: OCR confidence scores are always in [0.0, 1.0]**
    - **Validates: Requirements 3.7**
    - Use `@given(st.lists(st.integers(-200, 200)))` for raw tesseract confidence values; assert every `OCRWord.confidence` in the result is in `[0.0, 1.0]`

- [x] 6. VisualDetector implementation
  - [x] 6.1 Implement `VisualDetector` class
    - Implement `async find_template(template, screenshot)` — use `asyncio.to_thread(cv2.matchTemplate, ...)` with `cv2.TM_CCOEFF_NORMED`; return `UIElement` with bounding box at the best match location, or `None` if max correlation < 0.7
    - Implement `async find_element(label, screenshot)` — run OCR on screenshot, find words matching `label` (case-insensitive), build `UIElement` list sorted by confidence descending; also attempt template matching if label looks like a file path
    - Implement `async find_by_type(element_type, screenshot)` — use OCR heuristics and contour detection via `cv2` to locate common element types (button, input, checkbox, etc.); return list sorted by confidence descending
    - Implement `center_of(element)` — return `(element.bounding_box.x + element.bounding_box.width // 2, element.bounding_box.y + element.bounding_box.height // 2)`
    - Ensure `find_element` returns empty list (not exception) when no match found, and logs a warning
    - _Requirements: 4.1–4.7_

  - [x] 6.2 Write property test — template matching returns correct position (Property 8)
    - **Property 8: Template matching returns correct position**
    - **Validates: Requirements 4.5**
    - Use `@given(st.integers(0,1800), st.integers(0,980), st.integers(10,100), st.integers(10,100))` for template position and size; embed template into a synthetic screenshot; assert returned bounding box top-left is within 5 pixels of the known position

  - [x] 6.3 Write property test — search results are sorted by confidence descending (Property 9)
    - **Property 9: Search results are sorted by confidence descending**
    - **Validates: Requirements 4.2**
    - Use `@given(st.lists(st.floats(0,1), min_size=2))` for confidence values; construct mock OCR results; assert `result[i].confidence >= result[i+1].confidence` for all i

- [x] 7. WindowManager implementation
  - [x] 7.1 Implement `WindowManager` class
    - Implement `async list_windows()` — use `asyncio.to_thread(pygetwindow.getAllWindows)`; map each window to `WindowInfo` with `handle`, `title`, `process_name` (via `psutil.Process(win32process.GetWindowThreadProcessId(hwnd)[1]).name()` with fallback to empty string), and `bounding_box`
    - Implement `async focus(handle_or_title)` — resolve window by handle or case-insensitive substring title match; call `win.activate()` and `win.restore()` if minimised; return failed `ActionResult` (no exception) if window not found
    - Implement `async resize(handle_or_title, width, height)` — resolve window; call `win.resizeTo(width, height)`; return failed `ActionResult` if not found
    - Implement `async move(handle_or_title, x, y)` — resolve window; call `win.moveTo(x, y)`; return failed `ActionResult` if not found
    - Implement `async minimise(handle_or_title)` — resolve window; call `win.minimize()`; return failed `ActionResult` if not found
    - Implement `async maximise(handle_or_title)` — resolve window; call `win.maximize()`; return failed `ActionResult` if not found
    - Implement `_resolve_window(handle_or_title)` — if int, find by handle; if str, find first window where `pattern.lower() in title.lower()`; return `None` if not found
    - _Requirements: 7.1–7.8_

  - [x] 7.2 Write property test — window list contains all required fields (Property 12)
    - **Property 12: Window list contains all required fields**
    - **Validates: Requirements 7.1**
    - Use `@given(st.lists(st.fixed_dictionaries({...}), min_size=1))` for mock window data; assert every `WindowInfo` has non-None `handle`, `title`, `process_name`, and `bounding_box`

  - [x] 7.3 Write property test — invalid window handles return failed ActionResult without exception (Property 13)
    - **Property 13: Invalid window handles return failed ActionResult without exception**
    - **Validates: Requirements 7.7**
    - Use `@given(st.integers())` for arbitrary handle values with empty window list; assert all five operations return `ActionResult(success=False)` and raise no exception

  - [x] 7.4 Write property test — title pattern matching is case-insensitive substring (Property 14)
    - **Property 14: Title pattern matching is case-insensitive substring**
    - **Validates: Requirements 7.8**
    - Use `@given(st.text(), st.text())` for title T and pattern P; assert `_resolve_window(P)` finds the window iff `P.lower() in T.lower()`

- [x] 8. ActionLogger implementation
  - [x] 8.1 Implement `ActionLogger` class
    - `__init__(self, log_path="janus_computer_use.log")` — open/create a JSON-lines log file; set up `logging.getLogger("janus")` structured handler
    - Implement `log(entry: ActionLogEntry)` — append JSON-serialised entry to log file; emit structured log event via `logger.info("computer_use_action", extra={...})` with fields `event_type`, `action_type`, `target`, `outcome`, `timestamp`
    - Implement `_make_thumbnail(image: PIL.Image) -> str` — resize to 160×90, convert to JPEG bytes, base64-encode, return as string
    - Ensure every log entry includes a valid base64-encoded `screenshot_thumbnail`
    - _Requirements: 9.5, 9.6, 10.6_

  - [x] 8.2 Write property test — every action produces a log entry with all required fields (Property 19)
    - **Property 19: Every action produces a log entry with all required fields**
    - **Validates: Requirements 9.5, 9.6**
    - Use `@given(st.sampled_from(ActionType), st.text(), st.booleans())` for action type, target, and success; assert the written log entry has non-None `action_type`, `target`, `timestamp`, `outcome`, and a valid base64 `screenshot_thumbnail`

  - [x] 8.3 Write property test — emitted log events contain required structure fields (Property 22)
    - **Property 22: Emitted log events contain required structure fields**
    - **Validates: Requirements 10.6**
    - Capture `logger.info` calls via a mock handler; assert every emitted `extra` dict contains `event_type`, `action_type`, `target`, `outcome`, and `timestamp`

- [x] 9. ComputerUseEngine — core coordinator
  - [x] 9.1 Implement `ComputerUseEngine` class skeleton and async context manager
    - `__init__(self, context=None)` — store context dict; instantiate `MouseController`, `KeyboardController`, `ScreenReader`, `VisualDetector`, `WindowManager`, `ActionLogger`; initialise stuck-state hash buffer `_hash_buffer = deque(maxlen=3)`
    - Implement `async __aenter__` — call `_check_dependencies()`; initialise `ActionPlanner`; return `self`
    - Implement `async __aexit__` — flush `ActionLogger`; release any held resources
    - Expose `@property` accessors: `mouse`, `keyboard`, `screen`, `vision`, `windows`, `planner`
    - _Requirements: 10.4, 10.5_

  - [x] 9.2 Implement `execute_action` with safety guards and timeout
    - Implement `async execute_action(action)` — wrap entire execution in `asyncio.wait_for(..., timeout=30.0)`
    - Before executing: run destructive-action OCR scan; if destructive keywords found and `action.pre_approved == False`, return paused `ActionResult`
    - Dispatch to the correct sub-system based on `action.action_type`
    - After executing: run error-dialog OCR scan; if error dialog detected, attempt `Escape` or click "OK"/"Close", retry action once; if dialog reappears, return failed `ActionResult`
    - Log every action via `ActionLogger` with before-screenshot thumbnail
    - _Requirements: 9.1, 9.2, 9.5, 9.6_

  - [x] 9.3 Implement stuck-state detection
    - After each action, capture screenshot and compute `imagehash.phash(screenshot)`
    - Append hash to `_hash_buffer`; if all 3 hashes in buffer have pairwise distance ≤ 5, return failed `ActionResult` with `error_message="Stuck state detected after 3 consecutive no-change actions"`
    - _Requirements: 9.4_

  - [x] 9.4 Implement `wait_for` method
    - `async wait_for(condition)` — poll at `condition.poll_interval_seconds`; check condition type against current screen via OCR or template matching; return success when satisfied; return failed `ActionResult` on timeout
    - Clamp timeout to maximum 300 seconds
    - _Requirements: 5.1–5.6_

  - [x] 9.5 Write property test — stuck state detected after exactly 3 consecutive no-change screens (Property 18)
    - **Property 18: Stuck state detected after exactly 3 consecutive no-change screens**
    - **Validates: Requirements 9.4**
    - Use `@given(st.integers(min_value=3, max_value=10))` for N; mock screenshot to return identical image N times; assert stuck state is returned after exactly the 3rd consecutive no-change action, not before

  - [x] 9.6 Write property test — missing dependency ImportError lists all missing packages (Property 20)
    - **Property 20: Missing dependency ImportError lists all missing packages**
    - **Validates: Requirements 10.2**
    - Use `@given(st.lists(st.sampled_from(["pyautogui","pytesseract","PIL","pygetwindow","cv2"]), min_size=1, unique=True))` for missing package subsets; mock `__import__` to raise `ImportError` for those packages; assert the raised `ImportError` message contains every missing package name

  - [x] 9.7 Write property test — session context is accessible to ActionPlanner (Property 21)
    - **Property 21: Session context is accessible to ActionPlanner**
    - **Validates: Requirements 10.4**
    - Use `@given(st.dictionaries(st.text(), st.text()))` for context dicts; assert `engine.planner._context` equals the dict passed to `ComputerUseEngine.__init__`

- [x] 10. Checkpoint — core components complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. ActionPlanner implementation
  - [x] 11.1 Implement `ActionPlanner` class and perceive→reason→act loop
    - `__init__(self, engine, brain)` — store engine reference, brain reference, and context from `engine._context`; initialise `_history: List[StepRecord] = []`
    - Implement `async plan_next(goal, screenshot, history)` — run OCR and `find_element` on screenshot; build structured prompt (goal + OCR text + element list + last 5 history entries); call `await asyncio.to_thread(brain.ask, prompt)`; parse JSON response into `List[CandidateAction]`; on parse failure, re-prompt with simpler format; return list sorted by confidence descending
    - Implement `async run(goal, max_steps=50)` — loop: capture screenshot, call `plan_next`, select top action, execute via `engine.execute_action`, record `StepRecord` in `_history`, check stuck state, check goal achieved (look for success keywords in OCR), check `len(_history) >= max_steps`; return appropriate `ActionResult`
    - Ensure `_history` contains exactly N entries after N steps
    - _Requirements: 8.1–8.7_

  - [x] 11.2 Implement `run_goal` on `ComputerUseEngine`
    - `async run_goal(goal, max_steps=50)` — delegate to `self.planner.run(goal, max_steps)`
    - _Requirements: 8.1–8.7_

  - [x] 11.3 Write property test — planner step history length equals steps taken (Property 15)
    - **Property 15: Planner step history length equals steps taken**
    - **Validates: Requirements 8.7**
    - Use `@given(st.integers(min_value=1, max_value=50))` for N steps; mock engine to never achieve goal; assert `len(planner._history) == N` after N steps

  - [x] 11.4 Write property test — planner stops at max_steps (Property 16)
    - **Property 16: Planner stops at max_steps**
    - **Validates: Requirements 8.5**
    - Use `@given(st.integers(min_value=1, max_value=20))` for max_steps M; mock engine so goal is never achieved; assert planner executes exactly M steps and returns `ActionResult(success=False)`

  - [x] 11.5 Write property test — candidate actions contain all required fields (Property 17)
    - **Property 17: Candidate actions contain all required fields**
    - **Validates: Requirements 8.2**
    - Use `@given` with arbitrary JSON-like planner responses; assert every `CandidateAction` has non-None `action.action_type`, `action.params`, `confidence` in `[0.0, 1.0]`, and `rationale`

- [x] 12. BrowserComputerUse implementation
  - [x] 12.1 Implement `BrowserComputerUse` class
    - `__init__(self, engine, browser="chrome")` — store engine; store `_expected_domain = None`
    - Implement `async open(url)` — extract domain from URL; store as `_expected_domain`; use `engine.keyboard.hotkey("ctrl", "l")` to focus address bar, type URL, press Enter; wait for page load via `engine.wait_for(WaitCondition(TEXT_PRESENT, ""))`
    - Implement `async login(username, password)` — find username field via `engine.vision.find_element("username")` or `"email"`; click and type; find password field; click and type; find and click submit button
    - Implement `async search_jobs(query)` — navigate to search, type query, parse results via OCR, return list of dicts
    - Implement `async apply_to_job(job_url, cover_letter)` — open URL, find application form, fill cover letter field, submit
    - Implement `async submit_work(submission_url, content)` — open URL, find submission field, type content, submit
    - After each navigation action, read address bar via OCR and compare domain; return failed `ActionResult` if domain mismatch
    - _Requirements: 10.3_

  - [x] 12.2 Write unit tests for BrowserComputerUse
    - Test `open()` sets `_expected_domain` correctly
    - Test domain mismatch detection returns failed `ActionResult`
    - Test `login()` finds and fills username/password fields
    - Mock all engine calls; no real browser required
    - _Requirements: 10.3_

- [x] 13. Integration with janus_autonomous_worker.py
  - [x] 13.1 Add `execute_job_with_computer_use` method to `JanusAutonomousWorker`
    - In `janus_autonomous_worker.py`, add import: `from janus_computer_use import ComputerUseEngine` inside a `try/except ImportError` block (sets `HAS_COMPUTER_USE = True/False`)
    - Add `async execute_job_with_computer_use(self, job)` method that builds a context dict from `job.id`, `job.description`, `job.platform`; uses `async with ComputerUseEngine(context=context) as engine`; calls `await engine.run_goal(job.description)`; calls `self.submit_work` on success
    - Update `UpworkIntegration.get_available_jobs` to fall back to `BrowserComputerUse` when `self.api_key is None`: import `ComputerUseEngine` and `BrowserComputerUse`, open browser, call `browser.search_jobs(skills_query)`, return results
    - _Requirements: 10.1, 10.3_

  - [x] 13.2 Write unit tests for worker integration
    - Test `execute_job_with_computer_use` calls `engine.run_goal` with correct goal string
    - Test `UpworkIntegration.get_available_jobs` falls back to `BrowserComputerUse` when `api_key is None`
    - Mock `ComputerUseEngine` and `BrowserComputerUse`
    - _Requirements: 10.1, 10.3_

- [x] 14. Integration with janus_automation_platform.py
  - [x] 14.1 Add `COMPUTER_USE` task type and handler to `JanusAutomationEngine`
    - In `janus_automation_platform.py`, add `COMPUTER_USE = "computer_use"` to the `TaskType` enum
    - Add `_handle_computer_use` async method to `JanusAutomationEngine`: import `ComputerUseEngine` inside the method; use `async with ComputerUseEngine(context=config.get("context")) as engine`; call `await engine.run_goal(config["goal"], max_steps=config.get("max_steps", 50))`; return `{"success": result.success, "data": result.data, "error": result.error_message}`
    - Register `TaskType.COMPUTER_USE: self._handle_computer_use` in `_setup_task_handlers`
    - _Requirements: 10.1_

  - [x] 14.2 Write unit tests for automation platform integration
    - Test `_handle_computer_use` calls `engine.run_goal` with correct goal and max_steps
    - Test `TaskType.COMPUTER_USE` is registered in task handlers
    - Mock `ComputerUseEngine`
    - _Requirements: 10.1_

- [x] 15. Checkpoint — integration complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Property-based test suite (pytest-hypothesis)
  - [x] 16.1 Create test file structure
    - Create `tests/` directory with `__init__.py`
    - Create test files: `test_mouse_controller.py`, `test_keyboard_controller.py`, `test_screen_reader.py`, `test_visual_detector.py`, `test_window_manager.py`, `test_action_planner.py`, `test_computer_use_engine.py`, `test_browser_computer_use.py`, `test_integration.py`
    - Add `conftest.py` with shared fixtures: mock screen size `(1920, 1080)`, mock `PIL.ImageGrab.grab`, mock `pyautogui`, mock `pytesseract`, mock `pygetwindow`
    - Configure Hypothesis settings: `settings.register_profile("ci", max_examples=100)` and `settings.load_profile("ci")`
    - _Requirements: all_

  - [x] 16.2 Implement all property-based tests in their respective test files
    - Move property test sub-tasks from tasks 3–11 into their designated test files
    - Each property test MUST include the comment: `# Feature: janus-computer-use, Property N: <title>`
    - Each test MUST use `@settings(max_examples=100)`
    - All OS calls MUST be mocked so tests run without a display
    - _Requirements: all_

  - [x] 16.3 Write integration smoke tests in `test_integration.py`
    - Test that `import janus_computer_use` succeeds when all dependencies are present (mock imports)
    - Test that `ImportError` is raised with correct message when packages are missing
    - Test `ComputerUseEngine` async context manager lifecycle (`__aenter__`/`__aexit__`)
    - _Requirements: 10.1, 10.2, 10.5_

- [x] 17. Final checkpoint — all tests pass
  - Run `pytest tests/ --tb=short` and ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- All OS calls (pyautogui, pygetwindow, pytesseract, PIL screenshot) MUST be mocked in tests so the suite runs without a physical display
- Property tests use `@settings(max_examples=100)` as configured in the design
- The 22 correctness properties are distributed across test files matching the component they validate
- Checkpoints at tasks 10 and 15 ensure incremental validation before integration work begins
