# Implementation Plan: Janus Screen Recorder

## Overview

Extend `janus_computer_use.py` with a `ScreenRecorder` class that provides continuous background screen capture, ring-buffer storage, motion-based frame filtering, video encoding (MP4 and GIF), UI transition / stuck-state detection, and temporal context injection into `ActionPlanner`. Also add minimal changes to `ComputerUseEngine` and `ActionPlanner` to wire the new component in.

Implementation language: **Python** (asyncio, dataclasses, pytest-hypothesis for property tests).

## Tasks

- [x] 1. New data models
  - [x] 1.1 Add `ScreenRecorderConfig` dataclass
    - Add `ScreenRecorderConfig` dataclass to `janus_computer_use.py` with fields: `capture_rate_fps` (default 5), `buffer_duration_seconds` (default 30), `motion_threshold` (default 5), `high_motion_threshold` (default 20), `stuck_duration_seconds` (default 10), `transition_settling_seconds` (default 1.5), `temporal_context_frames` (default 3), `gif_max_dimension` (default 640)
    - Add `__post_init__` that calls `_validate_range` for every field; `_validate_range(name, value, min_val, max_val)` raises `ValueError` with message `"ScreenRecorder: '{name}' must be in [{min_val}, {max_val}], got {value!r}"` when out of range
    - Add `buffer_capacity` property returning `capture_rate_fps * buffer_duration_seconds`
    - _Requirements: 8.1, 8.2_

  - [x] 1.2 Add `RecordedFrame`, `ScreenClip`, and `EncodeResult` dataclasses
    - Add `RecordedFrame(image: Any, timestamp: float, phash: Any)` dataclass
    - Add `ScreenClip(frames: List[RecordedFrame], start_time: float, end_time: float, warning: Optional[str] = None)` dataclass with `frame_count` property returning `len(self.frames)`
    - Add `EncodeResult(success: bool, output_path: Optional[str] = None, file_size_bytes: int = 0, error_message: Optional[str] = None)` dataclass
    - _Requirements: 2.5, 4.5_

  - [x] 1.3 Write property test — config validation rejects out-of-range values (Property 5)
    - **Property 5: Config validation rejects out-of-range values**
    - **Validates: Requirements 8.2**
    - Use `@given` with out-of-range integers/floats for each parameter (e.g. `capture_rate_fps=0`, `capture_rate_fps=31`, `motion_threshold=-1`, `motion_threshold=65`)
    - Assert `ValueError` is raised and the message contains the parameter name
    - Use `@settings(max_examples=100)`

- [x] 2. ScreenRecorder — constructor and configuration
  - [x] 2.1 Implement `ScreenRecorder.__init__` and `config` property
    - Accept all eight parameters as constructor arguments with the same defaults as `ScreenRecorderConfig`
    - Construct a `ScreenRecorderConfig` from the arguments (validation happens in `__post_init__`)
    - Initialise ring buffer: `self._buffer: deque = deque(maxlen=self._config.buffer_capacity)`
    - Initialise state: `self._last_frame_diff: int = 0`, `self._diff_history: deque = deque()` (for motion score), `self._last_retained_frame: Optional[RecordedFrame] = None`, `self._running: bool = False`, `self._task: Optional[asyncio.Task] = None`
    - Initialise callback lists: `self._transition_callbacks: List[Callable] = []`, `self._stuck_callbacks: List[Callable] = []`
    - Implement `config` property returning `dataclasses.asdict(self._config)`
    - Implement `is_running` property returning `self._running`
    - _Requirements: 8.1, 8.3_

  - [x] 2.2 Write property test — config round-trip (Property 5 companion)
    - Use `@given` with valid values for all eight parameters
    - Construct `ScreenRecorder` with those values
    - Assert `recorder.config` returns a dict containing all eight parameters with the values passed in
    - Use `@settings(max_examples=100)`
    - _Requirements: 8.3_

- [x] 3. Capture backend and frame capture
  - [x] 3.1 Implement capture backend selection
    - At module level (inside `janus_computer_use.py`), attempt `import mss` inside a `try/except ImportError`; set `_CAPTURE_BACKEND = "mss"` on success, `_CAPTURE_BACKEND = "pil"` on failure; emit `logging.warning` when falling back to PIL
    - Implement `_capture_frame_mss() -> PIL.Image` — use `mss.mss()` context manager, capture primary monitor, convert to `PIL.Image`
    - Implement `_capture_frame_pil() -> PIL.Image` — use `PIL.ImageGrab.grab()`
    - Implement `_capture_frame() -> PIL.Image` — dispatch to the selected backend
    - _Requirements: 9.4, 10.2_

  - [x] 3.2 Implement `ScreenRecorder._capture_one_frame` (single frame capture + phash)
    - Call `await asyncio.to_thread(_capture_frame)` to get a PIL image
    - Compute `phash = await asyncio.to_thread(imagehash.phash, image)`
    - Return `RecordedFrame(image=image, timestamp=time.monotonic(), phash=phash)`
    - _Requirements: 1.1, 3.1_

  - [x] 3.3 Write property test — frame diff is always in [0, 64] (Property 2)
    - **Property 2: Frame diff is always in [0, 64]**
    - **Validates: Requirements 3.1, 3.6**
    - Use `@given(st.binary(min_size=4, max_size=4096))` to generate two arbitrary byte sequences; create 1×1 PIL images from them; compute phash distance
    - Assert the result is an integer in `[0, 64]`
    - Use `@settings(max_examples=100)`

- [x] 4. Ring buffer and motion detection
  - [x] 4.1 Implement motion detection logic in `ScreenRecorder._process_frame`
    - Accept a `RecordedFrame`; if `_last_retained_frame` is `None`, always retain the frame
    - Compute `diff = self._last_retained_frame.phash - frame.phash` (imagehash Hamming distance)
    - Update `self._last_frame_diff = diff`
    - Append `(frame.timestamp, diff)` to `self._diff_history`; prune entries older than 5 seconds
    - If `self._config.motion_threshold == 0` or `diff >= self._config.motion_threshold`: append frame to `self._buffer`, update `self._last_retained_frame`
    - Otherwise: discard frame (do not append to buffer)
    - If `diff > self._config.high_motion_threshold`: schedule `_fire_transition_callbacks()`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.7_

  - [x] 4.2 Write property test — ring buffer never exceeds capacity (Property 1)
    - **Property 1: Ring buffer never exceeds capacity**
    - **Validates: Requirements 1.2, 1.3**
    - Use `@given(st.integers(1, 30), st.integers(5, 60), st.integers(1, 500))` for fps, duration, and number of frames to add
    - Construct a `ScreenRecorder` with those fps/duration values; add `n_frames` mock `RecordedFrame` objects directly to `_buffer`
    - Assert `len(recorder._buffer) <= recorder._config.buffer_capacity` at all times
    - Use `@settings(max_examples=100)`

  - [x] 4.3 Write property test — motion threshold=0 retains every frame (Property 4)
    - **Property 4: Motion threshold=0 retains every frame**
    - **Validates: Requirements 3.4, 3.5**
    - Use `@given(st.lists(st.integers(0, 64), min_size=1, max_size=50))` for a sequence of diff values
    - Construct `ScreenRecorder(motion_threshold=0)`; mock `_capture_one_frame` to return frames with those diffs
    - Assert every frame is appended to the buffer (buffer length equals number of frames, up to capacity)
    - Use `@settings(max_examples=100)`

- [x] 5. Capture loop
  - [x] 5.1 Implement `ScreenRecorder.start` and the background capture loop
    - `async start()`: if already running, return immediately; set `self._running = True`; launch `asyncio.create_task(self._capture_loop())` and store in `self._task`
    - `async _capture_loop()`: loop while `self._running`; record `t0 = time.monotonic()`; call `await self._capture_one_frame()` inside `try/except Exception` (log error and continue on failure); call `self._process_frame(frame)`; check stuck state; sleep `max(0, 1/fps - elapsed)` to maintain target rate
    - On unhandled exception in the loop: log at ERROR level; attempt one restart; if restart also fails, emit stuck-state callbacks
    - _Requirements: 1.4, 1.5, 1.7_

  - [x] 5.2 Implement `ScreenRecorder.stop`
    - `async stop()`: set `self._running = False`; if `self._task` is not None, cancel it and `await asyncio.gather(self._task, return_exceptions=True)`; clear `self._buffer`; set `self._task = None`
    - _Requirements: 1.6, 7.2_

  - [x] 5.3 Implement `ScreenRecorder.__aenter__` and `__aexit__`
    - `__aenter__`: call `await self.start()`; return `self`
    - `__aexit__`: call `await self.stop()`
    - _Requirements: 7.6_

- [x] 6. Clip extraction
  - [x] 6.1 Implement `ScreenRecorder.get_clip`
    - `async get_clip(start_time, end_time)`: acquire a snapshot of the buffer (copy to list); filter frames where `start_time <= frame.timestamp <= end_time`; sort by timestamp ascending
    - If no frames are available for the full interval (i.e. the earliest buffer timestamp is after `start_time`), set `warning = "Ring buffer does not cover the full requested interval"`
    - Return `ScreenClip(frames=filtered, start_time=start_time, end_time=end_time, warning=warning)`
    - _Requirements: 2.1, 2.3, 2.5_

  - [x] 6.2 Implement `ScreenRecorder.get_recent_frames`
    - `async get_recent_frames(n)`: take a snapshot of the buffer; return the last `min(n, len(buffer))` frames as a list (most recent last)
    - _Requirements: 6.1_

  - [x] 6.3 Write property test — frames returned by get_clip are in chronological order (Property 3)
    - **Property 3: Frames returned by get_clip are in chronological order**
    - **Validates: Requirements 2.1, 2.5**
    - Use `@given(st.lists(st.floats(0, 1000), min_size=0, max_size=50))` for frame timestamps; construct a `ScreenRecorder` and populate `_buffer` with mock frames at those timestamps
    - Call `get_clip(0, 1000)` and assert `frames[i].timestamp <= frames[i+1].timestamp` for all i
    - Use `@settings(max_examples=100)`

  - [x] 6.4 Write property test — get_recent_frames(n) returns at most n frames (Property 7)
    - **Property 7: get_recent_frames(n) returns at most n frames**
    - **Validates: Requirements 6.1**
    - Use `@given(st.integers(1, 10), st.integers(0, 200))` for n and buffer_size
    - Populate buffer with `buffer_size` mock frames; call `get_recent_frames(n)`
    - Assert `len(result) <= n`
    - Use `@settings(max_examples=100)`

- [x] 7. Properties: last_frame_diff and motion_score
  - [x] 7.1 Implement `last_frame_diff` and `motion_score` properties
    - `last_frame_diff` property: return `self._last_frame_diff` (int, 0 if no frames captured yet)
    - `motion_score` property: sum the diff values in `self._diff_history` for entries within the last 5 seconds (use `time.monotonic()` as reference); return as float
    - _Requirements: 3.6, 5.6_

- [x] 8. Event callbacks — UI transition and stuck state
  - [x] 8.1 Implement `on_ui_transition` and `on_stuck_state` registration
    - `on_ui_transition(callback)`: append `callback` to `self._transition_callbacks`
    - `on_stuck_state(callback)`: append `callback` to `self._stuck_callbacks`
    - _Requirements: 5.4_

  - [x] 8.2 Implement `_fire_transition_callbacks` and `_fire_stuck_callbacks`
    - `async _fire_transition_callbacks()`: iterate `self._transition_callbacks`; for each, call `await callback()` inside `try/except Exception`; log exception at ERROR level and continue
    - `async _fire_stuck_callbacks(duration, last_frame)`: iterate `self._stuck_callbacks`; for each, call `await callback(duration=duration, last_frame=last_frame)` inside `try/except Exception`; log and continue
    - _Requirements: 5.4, 5.5_

  - [x] 8.3 Implement stuck-state detection in the capture loop
    - Track `self._last_motion_time: float` (updated whenever a frame is retained)
    - In `_capture_loop`, after processing each frame: if `time.monotonic() - self._last_motion_time >= self._config.stuck_duration_seconds`, call `await self._fire_stuck_callbacks(duration=..., last_frame=self._last_retained_frame)`; reset `_last_motion_time` to avoid repeated firing
    - _Requirements: 5.2, 5.3_

  - [x] 8.4 Implement UI transition settling detection
    - Track `self._in_transition: bool` and `self._transition_settled_at: Optional[float]`
    - When `diff > high_motion_threshold`: set `_in_transition = True`, reset `_transition_settled_at`
    - When `_in_transition` is True and `diff < motion_threshold`: if `_transition_settled_at` is None, record `_transition_settled_at = time.monotonic()`; if `time.monotonic() - _transition_settled_at >= transition_settling_seconds`, fire transition callbacks and set `_in_transition = False`
    - _Requirements: 5.1_

- [x] 9. Checkpoint — ScreenRecorder core complete
  - Ensure all tests for tasks 1–8 pass; ask the user if questions arise.

- [x] 10. VideoEncoder implementation
  - [x] 10.1 Implement `VideoEncoder.encode_mp4`
    - If `clip.frames` is empty, return `EncodeResult(success=False, error_message="Cannot encode empty clip to MP4")`
    - In a `asyncio.to_thread` call: compute fps from frame timestamps (fallback to 5.0 if single frame or zero duration); create `cv2.VideoWriter` with `fourcc=cv2.VideoWriter_fourcc(*"mp4v")`; write each frame as BGR numpy array; release writer
    - On success: return `EncodeResult(success=True, output_path=output_path, file_size_bytes=os.path.getsize(output_path))`
    - On any exception: delete partial file if it exists; return `EncodeResult(success=False, error_message=str(exc))`
    - _Requirements: 4.1, 4.3, 4.6, 4.7_

  - [x] 10.2 Implement `VideoEncoder.encode_gif`
    - If `clip.frames` is empty, return `EncodeResult(success=False, error_message="Cannot encode empty clip to GIF")`
    - In a `asyncio.to_thread` call: for each frame, scale to `max_dimension` on longest side using `PIL.Image.LANCZOS`; convert to palette mode; compute per-frame duration in ms from timestamps (minimum 20 ms); call `images[0].save(..., save_all=True, append_images=images[1:], loop=0, duration=durations)`
    - On success: return `EncodeResult(success=True, output_path=output_path, file_size_bytes=os.path.getsize(output_path))`
    - On any exception: delete partial file if it exists; return `EncodeResult(success=False, error_message=str(exc))`
    - _Requirements: 4.2, 4.4, 4.6, 4.7_

  - [x] 10.3 Write property test — EncodeResult on empty clip is always failure (Property 6)
    - **Property 6: EncodeResult on empty clip is always failure**
    - **Validates: Requirements 4.8**
    - Use `@given(st.text(min_size=1))` for output path; construct `ScreenClip(frames=[], start_time=0, end_time=0)`
    - Call both `encode_mp4` and `encode_gif`; assert both return `EncodeResult(success=False)` with non-empty `error_message` and that no file was created at the output path
    - Use `@settings(max_examples=100)`

  - [x] 10.4 Write property test — GIF scaling respects max_dimension (Property companion)
    - Use `@given(st.integers(64, 4096), st.integers(10, 3840), st.integers(10, 2160))` for max_dimension, width, height
    - Create a single-frame clip with an image of the given dimensions; call `encode_gif` with the given max_dimension (mock file I/O)
    - Assert the scaled image dimensions satisfy `max(scaled_w, scaled_h) <= max_dimension`
    - Use `@settings(max_examples=100)`

- [x] 11. ComputerUseEngine integration
  - [x] 11.1 Add `enable_temporal_context` parameter to `ComputerUseEngine.__init__`
    - Add `enable_temporal_context: bool = False` parameter to `__init__`
    - Add `self._enable_temporal_context = enable_temporal_context`
    - Add `self._screen_recorder: Optional[ScreenRecorder] = None`
    - _Requirements: 6.6, 7.1_

  - [x] 11.2 Start and stop `ScreenRecorder` in `ComputerUseEngine.__aenter__` / `__aexit__`
    - In `__aenter__`: if `self._enable_temporal_context`, instantiate `ScreenRecorder` using the engine's config (or defaults) and call `await self._screen_recorder.start()`
    - In `__aexit__`: if `self._screen_recorder is not None`, call `await self._screen_recorder.stop()`
    - Add `@property recorder` returning `self._screen_recorder`
    - _Requirements: 7.1, 7.2_

  - [x] 11.3 Write unit tests for ComputerUseEngine + ScreenRecorder lifecycle
    - Test that `ScreenRecorder.start` is called in `__aenter__` when `enable_temporal_context=True`
    - Test that `ScreenRecorder.stop` is called in `__aexit__`
    - Test that `ScreenRecorder` is NOT started when `enable_temporal_context=False`
    - Mock `ScreenRecorder` entirely
    - _Requirements: 6.6, 7.1, 7.2_

- [x] 12. ActionPlanner temporal context injection
  - [x] 12.1 Implement `_build_temporal_context_section` helper
    - Accept `frames: List[RecordedFrame]`, `recorder: ScreenRecorder`, `now: float`
    - For each frame: resize image to max 320×240 preserving aspect ratio; encode as JPEG bytes; base64-encode; compute `relative_ts = now - frame.timestamp`
    - Build a text block with one line per frame: `"  Frame -{relative_ts:.1f}s: <base64>"`
    - Append a motion summary line: `"MOTION SUMMARY:\n  Current frame diff: {recorder.last_frame_diff}\n  Motion score (5s): {recorder.motion_score:.1f}"`
    - Return the complete section as a string
    - _Requirements: 6.2, 6.4_

  - [x] 12.2 Inject temporal context into `ActionPlanner.plan_next`
    - After existing OCR + element detection, check `getattr(self._engine, '_screen_recorder', None)`
    - If recorder is not None and `recorder.is_running`: call `await recorder.get_recent_frames(n)` where n comes from the recorder's config; if frames are non-empty, call `_build_temporal_context_section` and append to the prompt
    - If recorder is None, not running, or buffer is empty: skip silently (no exception)
    - _Requirements: 6.1, 6.3, 6.5_

  - [x] 12.3 Write property test — temporal context thumbnails are valid base64 (Property 8)
    - **Property 8: Temporal context thumbnails are valid base64**
    - **Validates: Requirements 6.2**
    - Use `@given(st.integers(1, 640), st.integers(1, 480))` for image width and height; create a random PIL image of those dimensions
    - Call `_build_temporal_context_section` with a single-frame list
    - Extract the base64 string from the output; assert `base64.b64decode(encoded)` succeeds and the result is non-empty
    - Use `@settings(max_examples=100)`

- [x] 13. Checkpoint — full integration complete
  - Ensure all tests for tasks 10–12 pass; ask the user if questions arise.

- [x] 14. Property-based test suite (pytest-hypothesis)
  - [x] 14.1 Create test file structure for screen recorder
    - Create `tests/test_screen_recorder.py` — Properties 1, 2, 3, 4, 5, 7 + unit tests for lifecycle, motion detection, stuck state, callbacks
    - Create `tests/test_video_encoder.py` — Property 6 + unit tests for MP4/GIF encoding with mocked cv2 and PIL
    - Create `tests/test_temporal_context.py` — Property 8 + unit tests for `_build_temporal_context_section` and `plan_next` injection
    - Create `tests/test_screen_recorder_engine.py` — integration tests for `ComputerUseEngine` + `ScreenRecorder` lifecycle
    - Update `tests/conftest.py` to add shared fixtures: mock `mss`, mock `PIL.ImageGrab.grab`, mock `imagehash.phash`, mock `cv2.VideoWriter`
    - Configure Hypothesis: `settings.register_profile("ci", max_examples=100)` and `settings.load_profile("ci")`
    - _Requirements: all_

  - [x] 14.2 Implement all property-based tests in their respective test files
    - Each property test MUST include the comment: `# Feature: janus-screen-recorder, Property N: <title>`
    - Each test MUST use `@settings(max_examples=100)`
    - All OS calls (screen capture, file I/O, cv2, imagehash) MUST be mocked so tests run without a display
    - _Requirements: all_

  - [x] 14.3 Write integration smoke tests in `tests/test_screen_recorder_engine.py`
    - Test that `ScreenRecorder` is importable from `janus_computer_use`
    - Test `mss` fallback: mock `mss` as unavailable, verify `PIL.ImageGrab` is used and no `ImportError` is raised
    - Test `ComputerUseEngine` async context manager with `enable_temporal_context=True`: verify recorder starts and stops
    - _Requirements: 10.1, 10.2, 10.4_

- [x] 15. Final checkpoint — all tests pass
  - Run `pytest tests/ --tb=short` and ensure all tests pass
  - Ensure all tests pass; ask the user if questions arise.

## Notes

- All OS calls (screen capture, file I/O, `cv2.VideoWriter`, `imagehash.phash`) MUST be mocked in tests so the suite runs without a physical display or codec
- Property tests use `@settings(max_examples=100)` as configured in the design
- The 8 correctness properties are distributed across test files matching the component they validate
- `ScreenRecorderConfig.__post_init__` is the single source of truth for parameter validation — no duplicate validation elsewhere
- The `mss` import is optional and must never cause an `ImportError` to propagate; use `try/except ImportError` at module level
- Checkpoints at tasks 9 and 13 ensure incremental validation before integration work begins
- The `VideoEncoder` class is stateless — it can be instantiated once and reused for multiple encode calls
