# Requirements Document

## Introduction

The Janus Screen Recorder extends the existing `janus_computer_use.py` module with a temporal video view of the screen. Rather than relying solely on single static screenshots, the `ScreenRecorder` component gives the AI a continuous, time-aware picture of what the screen has been doing — enabling it to detect UI transitions, identify stuck states, and reason about sequences of events rather than isolated moments.

The component integrates with the existing `ScreenReader`, `ActionPlanner`, and `ComputerUseEngine` classes. It operates on Windows using asyncio, offloads all blocking capture and encoding calls via `asyncio.to_thread`, and builds on the existing `Pillow`, `opencv-python`, and `imagehash` dependencies already present in the project.

Five capabilities are in scope:

1. **Continuous background recording (ring buffer)** — always-on capture that retains the last N seconds of screen activity in memory.
2. **On-demand clip capture** — record a bounded window around an action execution (before, during, and after).
3. **Frame diff / motion detection** — skip frames where nothing changed to reduce memory and CPU overhead.
4. **Video encoding (MP4 / GIF)** — encode retained frames into a clip file for storage or AI vision model input.
5. **Integration with ActionPlanner** — inject recent frames as temporal context into the planner's perception prompt.

---

## Glossary

- **Screen_Recorder**: The new component that manages continuous background capture, ring-buffer storage, clip extraction, motion detection, and video encoding.
- **Ring_Buffer**: A fixed-capacity in-memory circular buffer that holds the most recent N seconds of captured frames, automatically discarding the oldest frames when full.
- **Frame**: A single captured screen image paired with its capture timestamp and a perceptual hash.
- **Frame_Diff**: The perceptual-hash distance between two consecutive frames, used to determine whether the screen changed meaningfully between captures.
- **Motion_Threshold**: The minimum Frame_Diff value above which a frame is considered to contain meaningful change and is retained; frames below this threshold are discarded.
- **Clip**: A contiguous sequence of frames extracted from the Ring_Buffer covering a specified time window.
- **Action_Window**: The time interval surrounding a single action execution: a configurable number of seconds before the action starts and after it completes.
- **Video_Encoder**: The sub-component responsible for encoding a Clip into an MP4 file or an animated GIF.
- **Temporal_Context**: A set of recent frames (or a summary derived from them) injected into the Action_Planner's prompt to give the AI a time-aware view of screen activity.
- **UI_Transition**: A detectable change in screen state between two frames, such as a loading spinner appearing or disappearing, a dialog opening, or a page navigation completing.
- **Stuck_State**: A condition where the screen has not changed meaningfully for a configurable duration, indicating the system may be waiting, frozen, or looping.
- **Capture_Rate**: The number of frames captured per second by the Screen_Recorder background loop.
- **Computer_Use_Engine**: The existing top-level coordinator in `janus_computer_use.py` that manages all sub-systems.
- **Action_Planner**: The existing AI component in `janus_computer_use.py` that implements the perceive → reason → act loop.
- **Screen_Reader**: The existing sub-system in `janus_computer_use.py` that captures single screenshots via `PIL.ImageGrab.grab`.
- **AvusBrain**: The existing AI backend used by the Action_Planner to reason about screen state and select actions.

---

## Requirements

---

### Requirement 1: Continuous Background Recording

**User Story:** As Janus, I want the screen to be recorded continuously in the background so that I always have access to recent screen history without having to explicitly trigger a capture.

#### Acceptance Criteria

1. THE Screen_Recorder SHALL capture frames from the primary display at a configurable Capture_Rate (default 5 fps, minimum 1 fps, maximum 30 fps).
2. WHEN the Screen_Recorder is running, THE Screen_Recorder SHALL store captured frames in a Ring_Buffer whose capacity is defined by a configurable duration (default 30 seconds, minimum 5 seconds, maximum 300 seconds).
3. WHEN the Ring_Buffer reaches its capacity, THE Screen_Recorder SHALL discard the oldest frame to make room for the newest frame without raising an exception.
4. THE Screen_Recorder SHALL run its capture loop as a background asyncio task that does not block the event loop.
5. WHEN the Screen_Recorder is started, THE Screen_Recorder SHALL begin capturing within 100 ms of the start call returning.
6. WHEN the Screen_Recorder is stopped, THE Screen_Recorder SHALL complete the current frame capture, stop the background task, and release all held frame memory within 500 ms.
7. IF a frame capture fails (e.g. display driver error), THE Screen_Recorder SHALL log the error, skip that frame, and continue the capture loop without stopping.

---

### Requirement 2: On-Demand Clip Capture

**User Story:** As Janus, I want to capture a bounded video clip around a specific action so that I can record exactly what happened before, during, and after that action for debugging or AI analysis.

#### Acceptance Criteria

1. WHEN a clip capture is requested with a start time and end time, THE Screen_Recorder SHALL extract all frames from the Ring_Buffer whose timestamps fall within the specified interval and return them as a Clip.
2. WHEN an Action_Window is specified (pre-action seconds and post-action seconds), THE Screen_Recorder SHALL begin buffering frames at the pre-action offset before the action starts and stop buffering at the post-action offset after the action completes.
3. WHEN a clip capture is requested and the Ring_Buffer does not contain frames for the full requested interval (e.g. the recorder was started after the interval began), THE Screen_Recorder SHALL return the available frames and include a warning in the Clip metadata.
4. THE Screen_Recorder SHALL support concurrent clip capture requests without corrupting the Ring_Buffer or other in-progress clips.
5. WHEN a Clip is returned, THE Clip SHALL include: the list of frames in chronological order, the start timestamp, the end timestamp, and the total frame count.
6. WHEN an action completes, THE Computer_Use_Engine SHALL automatically capture a Clip covering the Action_Window and attach it to the corresponding ActionResult.

---

### Requirement 3: Frame Diff and Motion Detection

**User Story:** As Janus, I want the recorder to skip frames where nothing changed on screen so that storage and CPU usage stay low during idle periods.

#### Acceptance Criteria

1. WHEN a new frame is captured, THE Screen_Recorder SHALL compute the Frame_Diff between the new frame and the most recently retained frame using perceptual hashing (imagehash.phash).
2. WHEN the Frame_Diff is below the Motion_Threshold, THE Screen_Recorder SHALL discard the new frame and not add it to the Ring_Buffer.
3. WHEN the Frame_Diff meets or exceeds the Motion_Threshold, THE Screen_Recorder SHALL retain the new frame in the Ring_Buffer.
4. THE Motion_Threshold SHALL be configurable (default 5 hash bits, minimum 0 to disable filtering, maximum 64 hash bits).
5. WHEN motion detection is disabled (Motion_Threshold set to 0), THE Screen_Recorder SHALL retain every captured frame regardless of Frame_Diff.
6. THE Screen_Recorder SHALL expose the most recent Frame_Diff value as a readable property so that other components can query current screen activity level.
7. WHEN the Frame_Diff exceeds a configurable high-motion threshold (default 20 hash bits), THE Screen_Recorder SHALL emit a UI_Transition event that other components can subscribe to.

---

### Requirement 4: Video Encoding

**User Story:** As Janus, I want to encode captured clips into MP4 or GIF files so that I can store them for debugging or pass them to an AI vision model as input.

#### Acceptance Criteria

1. WHEN an encode request is issued for a Clip, THE Video_Encoder SHALL encode the frames into an MP4 file using the H.264 codec via opencv-python.
2. WHEN an encode request specifies GIF output, THE Video_Encoder SHALL encode the frames into an animated GIF file using Pillow.
3. WHEN encoding to MP4, THE Video_Encoder SHALL use the frame timestamps from the Clip to set the correct frame rate in the output file.
4. WHEN encoding to GIF, THE Video_Encoder SHALL scale frames to a configurable maximum dimension (default 640 px on the longest side) to keep file size manageable.
5. WHEN an encode operation completes, THE Video_Encoder SHALL return the path to the output file and the file size in bytes.
6. IF an encode operation fails (e.g. codec not available, disk full), THE Video_Encoder SHALL return a failed result with a descriptive error message and SHALL NOT leave a partial file on disk.
7. THE Video_Encoder SHALL perform all encoding work via asyncio.to_thread so that encoding does not block the event loop.
8. WHEN a Clip contains zero frames, THE Video_Encoder SHALL return a failed result with a descriptive error message without attempting to write any file.

---

### Requirement 5: UI Transition and Stuck-State Detection

**User Story:** As Janus, I want the screen recorder to detect when UI transitions complete and when the screen is stuck, so that I can wait for loading states to finish and identify when the system is frozen.

#### Acceptance Criteria

1. WHEN the Screen_Recorder observes a sequence of frames where the Frame_Diff drops from above the Motion_Threshold to below it and remains below it for a configurable settling time (default 1.5 seconds), THE Screen_Recorder SHALL emit a UI_Transition_Complete event.
2. WHEN the Screen_Recorder observes that no frame has exceeded the Motion_Threshold for a configurable stuck duration (default 10 seconds), THE Screen_Recorder SHALL emit a Stuck_State event containing the duration of inactivity and the last retained frame.
3. WHEN a Stuck_State event is emitted, THE Computer_Use_Engine SHALL receive the event and incorporate it into the next Action_Planner decision cycle as additional context.
4. THE Screen_Recorder SHALL allow external components to register async callback functions for UI_Transition_Complete and Stuck_State events.
5. WHEN a registered callback raises an exception, THE Screen_Recorder SHALL log the exception and continue processing without propagating the error to the capture loop.
6. THE Screen_Recorder SHALL track the cumulative motion score (sum of Frame_Diff values) over a configurable rolling window (default 5 seconds) and expose it as a readable property.

---

### Requirement 6: Integration with ActionPlanner

**User Story:** As Janus, I want the ActionPlanner to receive recent screen frames as temporal context so that it can reason about what the screen was doing in the last few seconds, not just what it looks like right now.

#### Acceptance Criteria

1. WHEN the Action_Planner builds a perception prompt, THE Action_Planner SHALL optionally include a configurable number of recent frames (default 3, maximum 10) from the Screen_Recorder's Ring_Buffer as Temporal_Context.
2. WHEN Temporal_Context is included in a prompt, THE Action_Planner SHALL encode each frame as a base64 JPEG thumbnail (maximum 320×240 px) and include the frame's relative timestamp (seconds before the current moment) alongside it.
3. WHEN the Screen_Recorder is not running or the Ring_Buffer is empty, THE Action_Planner SHALL build the prompt without Temporal_Context and SHALL NOT raise an exception.
4. THE Action_Planner SHALL include a summary of recent motion activity (current Frame_Diff, cumulative motion score, last UI_Transition_Complete timestamp) in the text portion of the prompt when Temporal_Context is enabled.
5. WHEN Temporal_Context is disabled via configuration, THE Action_Planner SHALL behave identically to its current behaviour without any performance overhead from the Screen_Recorder.
6. THE Computer_Use_Engine SHALL expose a configuration flag `enable_temporal_context` (default False) that activates Screen_Recorder integration; WHEN this flag is False, THE Screen_Recorder SHALL NOT be started.

---

### Requirement 7: Lifecycle and Resource Management

**User Story:** As Janus, I want the Screen_Recorder to start and stop cleanly with the ComputerUseEngine so that no background threads or file handles are leaked.

#### Acceptance Criteria

1. WHEN the Computer_Use_Engine enters its async context (`__aenter__`) and `enable_temporal_context` is True, THE Computer_Use_Engine SHALL start the Screen_Recorder background task before returning.
2. WHEN the Computer_Use_Engine exits its async context (`__aexit__`), THE Computer_Use_Engine SHALL stop the Screen_Recorder, wait for the background task to complete, and release all frame memory.
3. THE Screen_Recorder SHALL expose `start()` and `stop()` async methods so that it can also be used independently of the Computer_Use_Engine lifecycle.
4. WHEN the Screen_Recorder is stopped while an encode operation is in progress, THE Screen_Recorder SHALL wait for the encode to complete before releasing frame memory.
5. IF the Screen_Recorder background task raises an unhandled exception, THE Screen_Recorder SHALL log the exception, attempt to restart the capture loop once, and emit a Stuck_State event if the restart also fails.
6. THE Screen_Recorder SHALL be usable as an async context manager independently of the Computer_Use_Engine.

---

### Requirement 8: Configuration

**User Story:** As Janus, I want all Screen_Recorder parameters to be configurable at construction time so that I can tune performance, storage, and sensitivity for different use cases.

#### Acceptance Criteria

1. THE Screen_Recorder SHALL accept all configurable parameters as constructor arguments with documented defaults: `capture_rate_fps`, `buffer_duration_seconds`, `motion_threshold`, `high_motion_threshold`, `stuck_duration_seconds`, `transition_settling_seconds`, `temporal_context_frames`, `gif_max_dimension`.
2. WHEN a parameter value outside its valid range is provided, THE Screen_Recorder SHALL raise a `ValueError` with a descriptive message identifying the parameter and its valid range.
3. THE Screen_Recorder configuration SHALL be readable after construction via a `config` property that returns a dictionary of all parameter names and their current values.
4. WHEN the Screen_Recorder is running, THE Screen_Recorder SHALL NOT allow configuration parameters to be changed; IF a change is attempted, THE Screen_Recorder SHALL raise a `RuntimeError`.

---

### Requirement 9: Performance Constraints

**User Story:** As Janus, I want the Screen_Recorder to have a bounded CPU and memory footprint so that it does not degrade the performance of the rest of the computer-use system.

#### Acceptance Criteria

1. WHILE the Screen_Recorder is running at the default Capture_Rate of 5 fps with motion detection enabled, THE Screen_Recorder SHALL consume no more than 10% of a single CPU core on average over any 10-second window.
2. THE Ring_Buffer memory usage SHALL not exceed `capture_rate_fps × buffer_duration_seconds × average_frame_size_bytes`, where `average_frame_size_bytes` is computed from the first 10 captured frames.
3. WHEN motion detection discards a frame, THE Screen_Recorder SHALL release the frame's memory immediately.
4. THE Screen_Recorder SHALL use `mss` for screen capture when it is available (faster than `PIL.ImageGrab`), and fall back to `PIL.ImageGrab` when `mss` is not installed.
5. WHEN encoding a Clip to MP4 or GIF, THE Video_Encoder SHALL not hold more than 2 seconds of unencoded frames in memory at any point during the encode operation.

---

### Requirement 10: Dependency and Integration Compatibility

**User Story:** As Janus, I want the Screen_Recorder to integrate cleanly with the existing janus_computer_use.py module without breaking any existing functionality.

#### Acceptance Criteria

1. THE Screen_Recorder SHALL be implemented as a new class within `janus_computer_use.py` and SHALL NOT require changes to the public API of any existing class.
2. WHEN `mss` is not installed, THE Screen_Recorder SHALL function correctly using `PIL.ImageGrab` and SHALL NOT raise an ImportError for `mss`.
3. THE Screen_Recorder SHALL use only dependencies already present in the project (`Pillow`, `opencv-python`, `imagehash`) plus the optional `mss`; it SHALL NOT introduce any other new required dependencies.
4. WHEN the existing `_check_dependencies()` function runs, THE Screen_Recorder's optional `mss` dependency SHALL be reported as a warning rather than a hard error if absent.
5. THE Screen_Recorder SHALL be covered by the existing test infrastructure pattern: all blocking OS calls (screen capture, file I/O) SHALL be mockable via `unittest.mock.patch` so that tests run without a display.
