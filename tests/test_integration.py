"""
test_integration.py — Unit tests for worker integration with ComputerUseEngine.

Task 13.2: Tests for execute_job_with_computer_use and UpworkIntegration fallback.

Requirements: 10.1, 10.3
"""
from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import pytest


# ---------------------------------------------------------------------------
# Stub out janus_computer_use so we can import janus_autonomous_worker
# without the real library installed.
# ---------------------------------------------------------------------------

def _install_computer_use_stub():
    """Install a minimal stub for janus_computer_use.

    Saves the real module (if already imported) so it can be restored after
    this test module is done collecting.  This prevents the stub from
    shadowing the real module when tests from other files are collected in the
    same pytest session.
    """
    mod = types.ModuleType("janus_computer_use")

    class _FakeActionResult:
        def __init__(self, success=True, data=None, error_message=None):
            self.success = success
            self.data = data
            self.error_message = error_message

    class _FakeComputerUseEngine:
        def __init__(self, context=None):
            self._context = context or {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def run_goal(self, goal, max_steps=50):
            return _FakeActionResult(success=True, data={"summary": ""})

    class _FakeBrowserComputerUse:
        def __init__(self, engine, browser="chrome"):
            self.engine = engine

        async def search_jobs(self, query):
            return []

    mod.ComputerUseEngine = _FakeComputerUseEngine
    mod.BrowserComputerUse = _FakeBrowserComputerUse
    mod.ActionResult = _FakeActionResult

    # Save the real module so we can restore it after collection
    _real = sys.modules.get("janus_computer_use")
    sys.modules["janus_computer_use"] = mod
    return mod, _real


_stub_mod, _real_jcu_mod = _install_computer_use_stub()

# Restore the real module immediately after the worker/platform imports below
# so that other test files collected in the same session can still import the
# real janus_computer_use.
def _restore_real_module():
    if _real_jcu_mod is not None:
        sys.modules["janus_computer_use"] = _real_jcu_mod
    else:
        # The real module was not loaded before; remove the stub so the next
        # importer gets the real module from disk.
        sys.modules.pop("janus_computer_use", None)

# Now import the worker module (it will pick up the stub above)
from janus_autonomous_worker import (  # noqa: E402
    JanusAutonomousWorker,
    Job,
    JobStatus,
    UpworkIntegration,
)

# Restore the real janus_computer_use module so other test files in the same
# pytest session can import the real classes (ActionType, ActionPlanner, etc.)
_restore_real_module()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(
    job_id: str = "job_001",
    description: str = "Write a Python script",
    platform: str = "upwork",
) -> Job:
    return Job(
        id=job_id,
        title="Test Job",
        description=description,
        required_skills=["python"],
        budget=100.0,
        deadline=datetime.now() + timedelta(days=7),
        platform=platform,
        status=JobStatus.AVAILABLE,
    )


# ---------------------------------------------------------------------------
# Tests for execute_job_with_computer_use
# ---------------------------------------------------------------------------


class TestExecuteJobWithComputerUse:
    """Tests for JanusAutonomousWorker.execute_job_with_computer_use."""

    @pytest.mark.asyncio
    async def test_run_goal_called_with_job_description(self):
        """execute_job_with_computer_use calls engine.run_goal with job.description."""
        worker = JanusAutonomousWorker()
        job = _make_job(description="Build a REST API")

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {"summary": "Done"}

        mock_engine = AsyncMock()
        mock_engine.run_goal = AsyncMock(return_value=mock_result)
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=False)

        MockEngineClass = MagicMock(return_value=mock_engine)

        with patch("janus_autonomous_worker.HAS_COMPUTER_USE", True), \
             patch("janus_autonomous_worker.ComputerUseEngine", MockEngineClass), \
             patch.object(worker, "_submit_work", new=AsyncMock(return_value=True)):
            result = await worker.execute_job_with_computer_use(job)

        mock_engine.run_goal.assert_awaited_once_with(job.description)
        assert result is True

    @pytest.mark.asyncio
    async def test_context_dict_built_from_job_fields(self):
        """ComputerUseEngine is initialised with context containing job_id, goal, platform."""
        worker = JanusAutonomousWorker()
        job = _make_job(job_id="j42", description="Analyse data", platform="fiverr")

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {}

        mock_engine = AsyncMock()
        mock_engine.run_goal = AsyncMock(return_value=mock_result)
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=False)

        MockEngineClass = MagicMock(return_value=mock_engine)

        with patch("janus_autonomous_worker.HAS_COMPUTER_USE", True), \
             patch("janus_autonomous_worker.ComputerUseEngine", MockEngineClass), \
             patch.object(worker, "_submit_work", new=AsyncMock(return_value=True)):
            await worker.execute_job_with_computer_use(job)

        call_kwargs = MockEngineClass.call_args
        context = call_kwargs[1].get("context") or call_kwargs[0][0]
        assert context["job_id"] == "j42"
        assert context["goal"] == "Analyse data"
        assert context["platform"] == "fiverr"

    @pytest.mark.asyncio
    async def test_submit_work_called_on_success(self):
        """_submit_work is called when engine.run_goal succeeds."""
        worker = JanusAutonomousWorker()
        job = _make_job()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {"summary": "Work done"}

        mock_engine = AsyncMock()
        mock_engine.run_goal = AsyncMock(return_value=mock_result)
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=False)

        mock_submit = AsyncMock(return_value=True)

        with patch("janus_autonomous_worker.HAS_COMPUTER_USE", True), \
             patch("janus_autonomous_worker.ComputerUseEngine", MagicMock(return_value=mock_engine)), \
             patch.object(worker, "_submit_work", new=mock_submit):
            result = await worker.execute_job_with_computer_use(job)

        mock_submit.assert_awaited_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_engine_fails(self):
        """execute_job_with_computer_use returns False when run_goal reports failure."""
        worker = JanusAutonomousWorker()
        job = _make_job()

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "Stuck state detected"

        mock_engine = AsyncMock()
        mock_engine.run_goal = AsyncMock(return_value=mock_result)
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=False)

        with patch("janus_autonomous_worker.HAS_COMPUTER_USE", True), \
             patch("janus_autonomous_worker.ComputerUseEngine", MagicMock(return_value=mock_engine)), \
             patch.object(worker, "_submit_work", new=AsyncMock(return_value=True)):
            result = await worker.execute_job_with_computer_use(job)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_computer_use_unavailable(self):
        """execute_job_with_computer_use returns False when HAS_COMPUTER_USE is False."""
        worker = JanusAutonomousWorker()
        job = _make_job()

        with patch("janus_autonomous_worker.HAS_COMPUTER_USE", False):
            result = await worker.execute_job_with_computer_use(job)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_engine_exception(self):
        """execute_job_with_computer_use returns False (no exception) when engine raises."""
        worker = JanusAutonomousWorker()
        job = _make_job()

        mock_engine = AsyncMock()
        mock_engine.__aenter__ = AsyncMock(side_effect=RuntimeError("engine crash"))
        mock_engine.__aexit__ = AsyncMock(return_value=False)

        with patch("janus_autonomous_worker.HAS_COMPUTER_USE", True), \
             patch("janus_autonomous_worker.ComputerUseEngine", MagicMock(return_value=mock_engine)):
            result = await worker.execute_job_with_computer_use(job)

        assert result is False


# ---------------------------------------------------------------------------
# Tests for UpworkIntegration.get_available_jobs fallback
# ---------------------------------------------------------------------------


class TestUpworkIntegrationBrowserFallback:
    """Tests for UpworkIntegration.get_available_jobs BrowserComputerUse fallback."""

    @pytest.mark.asyncio
    async def test_falls_back_to_browser_when_api_key_is_none(self):
        """get_available_jobs uses BrowserComputerUse when api_key is None."""
        upwork = UpworkIntegration(api_key=None)

        mock_engine = AsyncMock()
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=False)

        mock_browser = AsyncMock()
        mock_browser.search_jobs = AsyncMock(return_value=[])

        MockEngineClass = MagicMock(return_value=mock_engine)
        MockBrowserClass = MagicMock(return_value=mock_browser)

        with patch("janus_autonomous_worker.HAS_COMPUTER_USE", True), \
             patch("janus_autonomous_worker.ComputerUseEngine", MockEngineClass), \
             patch("janus_autonomous_worker.BrowserComputerUse", MockBrowserClass):
            jobs = await upwork.get_available_jobs(["python"])

        mock_browser.search_jobs.assert_awaited_once()
        assert isinstance(jobs, list)

    @pytest.mark.asyncio
    async def test_browser_search_called_with_skills_query(self):
        """search_jobs is called with a space-joined skills string."""
        upwork = UpworkIntegration(api_key=None)

        mock_engine = AsyncMock()
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=False)

        mock_browser = AsyncMock()
        mock_browser.search_jobs = AsyncMock(return_value=[])

        MockEngineClass = MagicMock(return_value=mock_engine)
        MockBrowserClass = MagicMock(return_value=mock_browser)

        with patch("janus_autonomous_worker.HAS_COMPUTER_USE", True), \
             patch("janus_autonomous_worker.ComputerUseEngine", MockEngineClass), \
             patch("janus_autonomous_worker.BrowserComputerUse", MockBrowserClass):
            await upwork.get_available_jobs(["python", "django"])

        call_args = mock_browser.search_jobs.call_args
        query = call_args[0][0]
        assert "python" in query
        assert "django" in query

    @pytest.mark.asyncio
    async def test_browser_results_converted_to_job_objects(self):
        """Job dicts returned by search_jobs are converted to Job instances."""
        upwork = UpworkIntegration(api_key=None)

        raw_jobs = [
            {
                "id": "bj_001",
                "title": "Browser Job",
                "description": "Do something",
                "skills": ["python"],
                "budget": 200.0,
            }
        ]

        mock_engine = AsyncMock()
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=False)

        mock_browser = AsyncMock()
        mock_browser.search_jobs = AsyncMock(return_value=raw_jobs)

        MockEngineClass = MagicMock(return_value=mock_engine)
        MockBrowserClass = MagicMock(return_value=mock_browser)

        with patch("janus_autonomous_worker.HAS_COMPUTER_USE", True), \
             patch("janus_autonomous_worker.ComputerUseEngine", MockEngineClass), \
             patch("janus_autonomous_worker.BrowserComputerUse", MockBrowserClass):
            jobs = await upwork.get_available_jobs(["python"])

        assert len(jobs) == 1
        assert isinstance(jobs[0], Job)
        assert jobs[0].id == "bj_001"
        assert jobs[0].platform == "upwork"
        assert jobs[0].status == JobStatus.AVAILABLE

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_computer_use_unavailable_and_no_api_key(self):
        """Returns [] when api_key is None and HAS_COMPUTER_USE is False."""
        upwork = UpworkIntegration(api_key=None)

        with patch("janus_autonomous_worker.HAS_COMPUTER_USE", False):
            jobs = await upwork.get_available_jobs(["python"])

        assert jobs == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_browser_raises(self):
        """Returns [] (no exception) when BrowserComputerUse raises an error."""
        upwork = UpworkIntegration(api_key=None)

        mock_engine = AsyncMock()
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=False)

        mock_browser = AsyncMock()
        mock_browser.search_jobs = AsyncMock(side_effect=RuntimeError("browser crash"))

        MockEngineClass = MagicMock(return_value=mock_engine)
        MockBrowserClass = MagicMock(return_value=mock_browser)

        with patch("janus_autonomous_worker.HAS_COMPUTER_USE", True), \
             patch("janus_autonomous_worker.ComputerUseEngine", MockEngineClass), \
             patch("janus_autonomous_worker.BrowserComputerUse", MockBrowserClass):
            jobs = await upwork.get_available_jobs(["python"])

        assert jobs == []

    @pytest.mark.asyncio
    async def test_api_key_present_does_not_use_browser(self):
        """When api_key is set, BrowserComputerUse is NOT used."""
        upwork = UpworkIntegration(api_key="real_key")

        MockBrowserClass = MagicMock()

        import requests

        mock_response = MagicMock()
        mock_response.json.return_value = {"jobs": []}

        with patch("janus_autonomous_worker.HAS_COMPUTER_USE", True), \
             patch("janus_autonomous_worker.BrowserComputerUse", MockBrowserClass), \
             patch("requests.get", return_value=mock_response):
            await upwork.get_available_jobs(["python"])

        MockBrowserClass.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for automation platform integration (Task 14.2)
# ---------------------------------------------------------------------------

# Stub out heavy platform dependencies so janus_automation_platform can be
# imported without FastAPI, uvicorn, etc. being fully installed.
def _install_automation_platform_stubs():
    """Install minimal stubs for janus_automation_platform dependencies."""
    # Janus-specific modules
    for mod_name in [
        "avus_brain",
        "janus_video_comprehension",
        "browser_automation",
        "janus_fault_integration",
    ]:
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            stub.AvusBrain = None
            stub.JanusVideoComprehension = None
            stub.BrowserAutomation = None
            stub.JanusAIGuard = None
            sys.modules[mod_name] = stub

    # Third-party modules that may not be installed in the test environment
    for mod_name in ["schedule", "uvicorn"]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    # fastapi stubs (only if not already installed)
    if "fastapi" not in sys.modules:
        fastapi_stub = types.ModuleType("fastapi")
        fastapi_stub.FastAPI = MagicMock
        fastapi_stub.HTTPException = Exception
        fastapi_stub.Depends = lambda x: x
        fastapi_stub.Security = lambda x: x
        fastapi_stub.status = types.SimpleNamespace(HTTP_401_UNAUTHORIZED=401)
        fastapi_stub.BackgroundTasks = MagicMock
        sys.modules["fastapi"] = fastapi_stub

        security_stub = types.ModuleType("fastapi.security")
        security_stub.HTTPBearer = MagicMock
        security_stub.HTTPAuthorizationCredentials = MagicMock
        sys.modules["fastapi.security"] = security_stub

        middleware_stub = types.ModuleType("fastapi.middleware")
        sys.modules["fastapi.middleware"] = middleware_stub

        cors_stub = types.ModuleType("fastapi.middleware.cors")
        cors_stub.CORSMiddleware = MagicMock
        sys.modules["fastapi.middleware.cors"] = cors_stub

    # pydantic stub
    if "pydantic" not in sys.modules:
        pydantic_stub = types.ModuleType("pydantic")
        pydantic_stub.BaseModel = object
        pydantic_stub.Field = lambda *a, **kw: None
        sys.modules["pydantic"] = pydantic_stub


_install_automation_platform_stubs()

from janus_automation_platform import (  # noqa: E402
    JanusAutomationEngine,
    TaskType,
)


class TestAutomationPlatformIntegration:
    """Tests for JanusAutomationEngine COMPUTER_USE task type and handler."""

    # ------------------------------------------------------------------
    # Task type registration
    # ------------------------------------------------------------------

    def test_computer_use_task_type_exists(self):
        """TaskType.COMPUTER_USE enum member exists with value 'computer_use'."""
        assert TaskType.COMPUTER_USE.value == "computer_use"

    def test_computer_use_registered_in_task_handlers(self):
        """TaskType.COMPUTER_USE is registered in the engine's task_handlers dict."""
        engine = JanusAutomationEngine()
        assert TaskType.COMPUTER_USE in engine.task_handlers

    def test_computer_use_handler_is_callable(self):
        """The registered handler for COMPUTER_USE is callable."""
        engine = JanusAutomationEngine()
        handler = engine.task_handlers[TaskType.COMPUTER_USE]
        assert callable(handler)

    # ------------------------------------------------------------------
    # _handle_computer_use behaviour
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_handle_computer_use_calls_run_goal_with_goal(self):
        """_handle_computer_use calls engine.run_goal with config['goal']."""
        automation_engine = JanusAutomationEngine()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {"output": "done"}
        mock_result.error_message = None

        mock_engine = AsyncMock()
        mock_engine.run_goal = AsyncMock(return_value=mock_result)
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=False)

        MockEngineClass = MagicMock(return_value=mock_engine)

        config = {"goal": "Open Notepad and type hello", "context": {"user": "test"}}

        with patch("janus_automation_platform.ComputerUseEngine", MockEngineClass, create=True):
            # Patch the import inside the method
            import importlib
            import janus_automation_platform as jap
            with patch.dict("sys.modules", {"janus_computer_use": types.SimpleNamespace(
                ComputerUseEngine=MockEngineClass
            )}):
                result = await automation_engine._handle_computer_use(config)

        mock_engine.run_goal.assert_awaited_once_with(
            "Open Notepad and type hello", max_steps=50
        )

    @pytest.mark.asyncio
    async def test_handle_computer_use_uses_custom_max_steps(self):
        """_handle_computer_use passes max_steps from config when provided."""
        automation_engine = JanusAutomationEngine()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = None
        mock_result.error_message = None

        mock_engine = AsyncMock()
        mock_engine.run_goal = AsyncMock(return_value=mock_result)
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=False)

        MockEngineClass = MagicMock(return_value=mock_engine)

        config = {"goal": "Search for jobs", "max_steps": 10}

        with patch.dict("sys.modules", {"janus_computer_use": types.SimpleNamespace(
            ComputerUseEngine=MockEngineClass
        )}):
            result = await automation_engine._handle_computer_use(config)

        mock_engine.run_goal.assert_awaited_once_with("Search for jobs", max_steps=10)

    @pytest.mark.asyncio
    async def test_handle_computer_use_returns_correct_dict_on_success(self):
        """_handle_computer_use returns dict with success, data, error keys."""
        automation_engine = JanusAutomationEngine()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {"summary": "task complete"}
        mock_result.error_message = None

        mock_engine = AsyncMock()
        mock_engine.run_goal = AsyncMock(return_value=mock_result)
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=False)

        MockEngineClass = MagicMock(return_value=mock_engine)

        config = {"goal": "Do something"}

        with patch.dict("sys.modules", {"janus_computer_use": types.SimpleNamespace(
            ComputerUseEngine=MockEngineClass
        )}):
            result = await automation_engine._handle_computer_use(config)

        assert result["success"] is True
        assert result["data"] == {"summary": "task complete"}
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_handle_computer_use_returns_correct_dict_on_failure(self):
        """_handle_computer_use returns dict with success=False and error message on failure."""
        automation_engine = JanusAutomationEngine()

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.data = None
        mock_result.error_message = "Stuck state detected"

        mock_engine = AsyncMock()
        mock_engine.run_goal = AsyncMock(return_value=mock_result)
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=False)

        MockEngineClass = MagicMock(return_value=mock_engine)

        config = {"goal": "Impossible task"}

        with patch.dict("sys.modules", {"janus_computer_use": types.SimpleNamespace(
            ComputerUseEngine=MockEngineClass
        )}):
            result = await automation_engine._handle_computer_use(config)

        assert result["success"] is False
        assert result["error"] == "Stuck state detected"

    @pytest.mark.asyncio
    async def test_handle_computer_use_passes_context_to_engine(self):
        """_handle_computer_use passes config['context'] to ComputerUseEngine constructor."""
        automation_engine = JanusAutomationEngine()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = None
        mock_result.error_message = None

        mock_engine = AsyncMock()
        mock_engine.run_goal = AsyncMock(return_value=mock_result)
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=False)

        MockEngineClass = MagicMock(return_value=mock_engine)

        ctx = {"session_id": "abc123", "user": "janus"}
        config = {"goal": "Do work", "context": ctx}

        with patch.dict("sys.modules", {"janus_computer_use": types.SimpleNamespace(
            ComputerUseEngine=MockEngineClass
        )}):
            await automation_engine._handle_computer_use(config)

        MockEngineClass.assert_called_once_with(context=ctx)

    @pytest.mark.asyncio
    async def test_handle_computer_use_context_defaults_to_none(self):
        """_handle_computer_use passes context=None when config has no 'context' key."""
        automation_engine = JanusAutomationEngine()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = None
        mock_result.error_message = None

        mock_engine = AsyncMock()
        mock_engine.run_goal = AsyncMock(return_value=mock_result)
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=False)

        MockEngineClass = MagicMock(return_value=mock_engine)

        config = {"goal": "Do work"}

        with patch.dict("sys.modules", {"janus_computer_use": types.SimpleNamespace(
            ComputerUseEngine=MockEngineClass
        )}):
            await automation_engine._handle_computer_use(config)

        MockEngineClass.assert_called_once_with(context=None)


# ---------------------------------------------------------------------------
# Task 16.3: Smoke tests for janus_computer_use module
# Requirements: 10.1, 10.2, 10.5
# ---------------------------------------------------------------------------


class TestImportSmoke:
    """Smoke tests for janus_computer_use import and dependency checking."""

    def test_import_succeeds_when_all_dependencies_present(self):
        """
        Test that `import janus_computer_use` succeeds when all dependencies
        are present (mocked).

        Requirements: 10.1
        """
        import importlib
        import types as _types

        # Build minimal stubs for all required packages
        _required_stubs = {
            "pyautogui": {"size": lambda: (1920, 1080), "FAILSAFE": True, "easeInOutQuad": "easeInOutQuad"},
            "pytesseract": {},
            "PIL": {},
            "PIL.Image": {},
            "PIL.ImageGrab": {},
            "pygetwindow": {},
            "cv2": {},
            "imagehash": {},
        }

        saved = {}
        for pkg, attrs in _required_stubs.items():
            if pkg in sys.modules:
                saved[pkg] = sys.modules[pkg]
            mod = _types.ModuleType(pkg)
            for attr, val in attrs.items():
                setattr(mod, attr, val)
            sys.modules[pkg] = mod

        try:
            # Remove janus_computer_use from sys.modules to force a fresh import
            saved_jcu = sys.modules.pop("janus_computer_use", None)
            try:
                # Patch _check_dependencies to no-op so the import succeeds
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "janus_computer_use_smoke",
                    "janus_computer_use.py",
                )
                mod = importlib.util.module_from_spec(spec)
                # Patch _check_dependencies before exec_module
                mod._check_dependencies = lambda: None  # type: ignore[attr-defined]
                # We can't easily patch before exec_module, so just import normally
                # with the stubs in place and _check_dependencies patched
                import janus_computer_use as jcu
                jcu._check_dependencies = lambda: None
                # Verify key classes are accessible
                assert hasattr(jcu, "ComputerUseEngine"), "ComputerUseEngine not found"
                assert hasattr(jcu, "MouseController"), "MouseController not found"
                assert hasattr(jcu, "KeyboardController"), "KeyboardController not found"
                assert hasattr(jcu, "ScreenReader"), "ScreenReader not found"
                assert hasattr(jcu, "ActionType"), "ActionType not found"
            finally:
                if saved_jcu is not None:
                    sys.modules["janus_computer_use"] = saved_jcu
        finally:
            # Restore original modules
            for pkg, mod in saved.items():
                sys.modules[pkg] = mod

    def test_import_error_raised_when_packages_missing(self):
        """
        Test that ImportError is raised with a message listing all missing
        packages when required dependencies are absent.

        Requirements: 10.2
        """
        import builtins
        import janus_computer_use as jcu

        # Save the real _check_dependencies
        real_check = jcu._check_dependencies

        # Temporarily restore the real function for this test
        import importlib.util
        loader = importlib.util.spec_from_file_location("_jcu_tmp", jcu.__file__)
        tmp_spec = loader
        tmp_mod = importlib.util.module_from_spec(tmp_spec)

        # Load the source to get the real _check_dependencies
        import importlib.machinery
        src_loader = importlib.machinery.SourceFileLoader("_jcu_src", jcu.__file__)
        src_spec = importlib.util.spec_from_loader("_jcu_src", src_loader)
        src_mod = importlib.util.module_from_spec(src_spec)

        # Patch builtins.__import__ to suppress ImportErrors during exec_module
        real_import = builtins.__import__

        def _safe_import(name, *args, **kwargs):
            try:
                return real_import(name, *args, **kwargs)
            except ImportError:
                stub = types.ModuleType(name)
                sys.modules[name] = stub
                return stub

        builtins.__import__ = _safe_import
        try:
            src_spec.loader.exec_module(src_mod)
        except Exception:
            pass
        finally:
            builtins.__import__ = real_import

        real_check_deps = src_mod.__dict__.get("_check_dependencies")
        if real_check_deps is None:
            pytest.skip("Could not load real _check_dependencies for testing")

        missing_packages = ["pyautogui", "pytesseract", "PIL"]

        # Remove the packages from sys.modules so __import__ actually tries to import them
        saved_mods = {}
        for pkg in missing_packages:
            if pkg in sys.modules:
                saved_mods[pkg] = sys.modules.pop(pkg)

        real_import2 = builtins.__import__

        def _failing_import(name, *args, **kwargs):
            if name in missing_packages:
                raise ImportError(f"No module named '{name}'")
            return real_import2(name, *args, **kwargs)

        # Patch _REQUIRED_PACKAGES to only contain the missing packages
        original_required = src_mod._REQUIRED_PACKAGES
        src_mod._REQUIRED_PACKAGES = [(pkg, pkg) for pkg in missing_packages]

        try:
            builtins.__import__ = _failing_import
            with pytest.raises(ImportError) as exc_info:
                real_check_deps()

            error_msg = str(exc_info.value)
            for pkg in missing_packages:
                assert pkg in error_msg, (
                    f"ImportError message does not mention missing package '{pkg}': {error_msg!r}"
                )
        finally:
            builtins.__import__ = real_import2
            src_mod._REQUIRED_PACKAGES = original_required
            # Restore saved modules
            for pkg, mod in saved_mods.items():
                sys.modules[pkg] = mod


class TestComputerUseEngineLifecycle:
    """Tests for ComputerUseEngine async context manager lifecycle.

    Requirements: 10.5
    """

    @pytest.mark.asyncio
    async def test_aenter_returns_engine_instance(self):
        """
        __aenter__ must return the ComputerUseEngine instance itself.

        Requirements: 10.5
        """
        import janus_computer_use as jcu

        # Patch _check_dependencies so __aenter__ doesn't fail
        original_check = jcu._check_dependencies
        jcu._check_dependencies = lambda: None

        try:
            engine = jcu.ComputerUseEngine(context={"test": "value"})
            result = await engine.__aenter__()
            assert result is engine, (
                "__aenter__ must return the engine instance (self)"
            )
            # Clean up
            await engine.__aexit__(None, None, None)
        finally:
            jcu._check_dependencies = original_check

    @pytest.mark.asyncio
    async def test_aexit_does_not_raise(self):
        """
        __aexit__ must complete without raising an exception.

        Requirements: 10.5
        """
        import janus_computer_use as jcu

        original_check = jcu._check_dependencies
        jcu._check_dependencies = lambda: None

        try:
            engine = jcu.ComputerUseEngine(context={})
            await engine.__aenter__()
            # Should not raise
            await engine.__aexit__(None, None, None)
        finally:
            jcu._check_dependencies = original_check

    @pytest.mark.asyncio
    async def test_context_manager_as_async_with(self):
        """
        ComputerUseEngine can be used as `async with ComputerUseEngine(...) as engine`.

        Requirements: 10.5
        """
        import janus_computer_use as jcu

        original_check = jcu._check_dependencies
        jcu._check_dependencies = lambda: None

        try:
            ctx = {"job_id": "test_123", "goal": "test goal"}
            async with jcu.ComputerUseEngine(context=ctx) as engine:
                assert engine is not None
                assert engine._context == ctx
        finally:
            jcu._check_dependencies = original_check

    @pytest.mark.asyncio
    async def test_context_manager_exposes_subsystem_properties(self):
        """
        After __aenter__, the engine exposes mouse, keyboard, screen, vision,
        windows, and planner properties.

        Requirements: 10.5
        """
        import janus_computer_use as jcu

        original_check = jcu._check_dependencies
        jcu._check_dependencies = lambda: None

        try:
            async with jcu.ComputerUseEngine(context={}) as engine:
                assert engine.mouse is not None, "engine.mouse must not be None"
                assert engine.keyboard is not None, "engine.keyboard must not be None"
                assert engine.screen is not None, "engine.screen must not be None"
                assert engine.vision is not None, "engine.vision must not be None"
                assert engine.windows is not None, "engine.windows must not be None"
        finally:
            jcu._check_dependencies = original_check


# ===========================================================================
# Tasks 12.1, 12.2, 12.3 — WorkCycle integration tests
# Requirements: 1.1, 1.4, 4.2, 5.3, 9.3, 14.3, 15.1, 20.1
# ===========================================================================

import asyncio
import tempfile
import os
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from janus_worker_core import (
    BrowserJob,
    CycleSummary,
    DecisionEngine,
    InvestmentEngine,
    JobStatus,
    LearningEngine,
    MarketAnalyzer,
    MonitoringSystem,
    QAResult,
    QualityAssurance,
    WorkCycle,
    WorkGenerator,
    WorkResult,
    WorkerDatabase,
)
from janus_wallet import JanusWallet, TransactionType


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_browser_job(
    job_id: str = "test-job-001",
    title: str = "Write a Python script",
    description: str = "Create a script that processes CSV files",
    platform: str = "upwork",
    budget: float = 75.0,
    job_type: str = "code",
) -> BrowserJob:
    return BrowserJob(
        id=job_id,
        title=title,
        description=description,
        platform=platform,
        budget=budget,
        required_skills=["python", "csv"],
        deadline=None,
        job_type=job_type,
    )


def _make_work_cycle(
    db: WorkerDatabase,
    wallet: JanusWallet,
    brain_ask: AsyncMock,
    max_concurrent_jobs: int = 5,
) -> WorkCycle:
    """Build a WorkCycle wired to real DB + real wallet, with a mocked brain."""
    brain = MagicMock()
    brain.ask = brain_ask

    work_generator = WorkGenerator(brain=brain)
    quality_assurance = QualityAssurance()
    decision_engine = DecisionEngine()

    learning_engine = MagicMock(spec=LearningEngine)
    learning_engine.learn_skill = AsyncMock(
        return_value=MagicMock(skill_name="python", resources_used=[])
    )

    investment_engine = MagicMock(spec=InvestmentEngine)
    investment_engine.evaluate_and_invest = AsyncMock(return_value=[])

    market_analyzer = MagicMock(spec=MarketAnalyzer)

    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        log_path = f.name
    monitor = MonitoringSystem(log_path=log_path)

    return WorkCycle(
        db=db,
        wallet=wallet,
        decision_engine=decision_engine,
        work_generator=work_generator,
        learning_engine=learning_engine,
        quality_assurance=quality_assurance,
        investment_engine=investment_engine,
        market_analyzer=market_analyzer,
        monitor=monitor,
        max_concurrent_jobs=max_concurrent_jobs,
    )


# ---------------------------------------------------------------------------
# Task 12.1 — Full cycle smoke test
# Requirements: 1.1, 4.2, 9.3, 15.1
# ---------------------------------------------------------------------------

class TestFullCycleSmokeTest:
    """
    Task 12.1: Full cycle smoke test.

    Mock PlatformBrowser to return one BrowserJob; mock AvusBrain.ask() to
    return valid work text; run WorkCycle.run_one_cycle(); assert DB has a job
    record with status 'completed' and JanusWallet ledger has an income
    transaction.
    """

    @pytest.mark.asyncio
    async def test_full_cycle_completes_job_and_records_income(self):
        """
        Full cycle: one job discovered → work generated → QA passes →
        DB record status=completed, wallet has income transaction.

        Requirements: 1.1, 4.2, 9.3, 15.1
        """
        db = WorkerDatabase(":memory:")
        wallet = JanusWallet(db_path=":memory:")

        # Brain returns a long, keyword-rich response that will pass QA
        job = _make_browser_job()
        valid_work = (
            "```python\n"
            "import csv\n\n"
            "# Script that processes CSV files as requested\n"
            "# python csv processing solution\n\n"
            "def process_csv(filepath):\n"
            "    with open(filepath) as f:\n"
            "        reader = csv.DictReader(f)\n"
            "        rows = list(reader)\n"
            "    return rows\n\n"
            "if __name__ == '__main__':\n"
            "    result = process_csv('data.csv')\n"
            "    print(f'Processed {len(result)} rows')\n"
            "```\n"
        )
        brain_ask = AsyncMock(return_value=valid_work)

        wc = _make_work_cycle(db, wallet, brain_ask)

        # Patch _discover_jobs to return our single test job
        async def mock_discover():
            return [job]

        wc._discover_jobs = mock_discover

        # Run one cycle
        summary = await wc.run_one_cycle()

        # Assert DB has the job record with status=completed
        record = db.get_job(job.id)
        assert record is not None, "Job record should exist in DB"
        assert record.status == JobStatus.COMPLETED.value, (
            f"Expected status 'completed', got '{record.status}'"
        )

        # Assert wallet ledger has at least one income transaction
        income_txs = wallet._ledger.list_transactions(
            tx_type=TransactionType.INCOME
        )
        assert len(income_txs) >= 1, (
            "Expected at least one income transaction in wallet ledger"
        )
        # Verify the income amount matches the job budget
        assert income_txs[0].amount == Decimal(str(job.budget)), (
            f"Expected income amount {job.budget}, got {income_txs[0].amount}"
        )

    @pytest.mark.asyncio
    async def test_full_cycle_returns_cycle_summary(self):
        """run_one_cycle() returns a CycleSummary with jobs_processed >= 1."""
        db = WorkerDatabase(":memory:")
        wallet = JanusWallet(db_path=":memory:")

        job = _make_browser_job()
        # Use the same rich work content that passes QA (long, keyword-rich, code-formatted)
        valid_work = (
            "```python\n"
            "import csv\n\n"
            "# Script that processes CSV files as requested\n"
            "# python csv processing solution\n\n"
            "def process_csv(filepath):\n"
            "    with open(filepath) as f:\n"
            "        reader = csv.DictReader(f)\n"
            "        rows = list(reader)\n"
            "    return rows\n\n"
            "if __name__ == '__main__':\n"
            "    result = process_csv('data.csv')\n"
            "    print(f'Processed {len(result)} rows')\n"
            "```\n"
        )
        brain_ask = AsyncMock(return_value=valid_work)
        wc = _make_work_cycle(db, wallet, brain_ask)

        async def mock_discover():
            return [job]

        wc._discover_jobs = mock_discover

        summary = await wc.run_one_cycle()

        assert isinstance(summary, CycleSummary)
        assert summary.jobs_processed >= 1
        assert summary.earnings > Decimal("0")


# ---------------------------------------------------------------------------
# Task 12.2 — Browser fallback test
# Requirements: 5.3, 20.1
# ---------------------------------------------------------------------------

class TestBrowserFallbackTest:
    """
    Task 12.2: Browser fallback test.

    Mock UpworkBrowser to raise an exception; assert FiverrBrowser is called
    next; assert cycle completes without crashing.
    """

    @pytest.mark.asyncio
    async def test_upwork_failure_falls_back_to_fiverr(self):
        """
        When UpworkBrowser.find_jobs() raises, FiverrBrowser.find_jobs() is
        called and the cycle completes without crashing.

        Requirements: 5.3, 20.1
        """
        db = WorkerDatabase(":memory:")
        wallet = JanusWallet(db_path=":memory:")
        brain_ask = AsyncMock(return_value="some work output")
        wc = _make_work_cycle(db, wallet, brain_ask)

        fiverr_job = _make_browser_job(
            job_id="fiverr-job-001",
            platform="fiverr",
            title="Design a logo",
            description="Create a modern logo for a tech startup",
            job_type="general",
        )

        mock_upwork = MagicMock()
        mock_upwork.find_jobs = AsyncMock(side_effect=RuntimeError("Upwork browser crashed"))

        mock_fiverr = MagicMock()
        mock_fiverr.find_jobs = AsyncMock(return_value=[fiverr_job])

        with patch("janus_worker_core._HAS_PLATFORM_BROWSER", True), \
             patch("janus_worker_core.UpworkBrowser", return_value=mock_upwork), \
             patch("janus_worker_core.FiverrBrowser", return_value=mock_fiverr):
            # Should not raise
            summary = await wc.run_one_cycle()

        # FiverrBrowser.find_jobs was called
        mock_fiverr.find_jobs.assert_awaited_once()

        # Cycle completed without crashing
        assert isinstance(summary, CycleSummary)

    @pytest.mark.asyncio
    async def test_both_browsers_fail_cycle_still_completes(self):
        """
        When both UpworkBrowser and FiverrBrowser raise, the cycle completes
        with zero jobs processed (no crash).

        Requirements: 5.3, 20.1
        """
        db = WorkerDatabase(":memory:")
        wallet = JanusWallet(db_path=":memory:")
        brain_ask = AsyncMock(return_value="some work output")
        wc = _make_work_cycle(db, wallet, brain_ask)

        mock_upwork = MagicMock()
        mock_upwork.find_jobs = AsyncMock(side_effect=RuntimeError("Upwork down"))

        mock_fiverr = MagicMock()
        mock_fiverr.find_jobs = AsyncMock(side_effect=RuntimeError("Fiverr down"))

        with patch("janus_worker_core._HAS_PLATFORM_BROWSER", True), \
             patch("janus_worker_core.UpworkBrowser", return_value=mock_upwork), \
             patch("janus_worker_core.FiverrBrowser", return_value=mock_fiverr):
            summary = await wc.run_one_cycle()

        assert isinstance(summary, CycleSummary)
        assert summary.jobs_processed == 0

    @pytest.mark.asyncio
    async def test_upwork_returns_empty_falls_back_to_fiverr(self):
        """
        When UpworkBrowser.find_jobs() returns an empty list, FiverrBrowser
        is tried next.

        Requirements: 5.3, 20.1
        """
        db = WorkerDatabase(":memory:")
        wallet = JanusWallet(db_path=":memory:")
        brain_ask = AsyncMock(return_value="some work output")
        wc = _make_work_cycle(db, wallet, brain_ask)

        fiverr_job = _make_browser_job(job_id="fiverr-002", platform="fiverr")

        mock_upwork = MagicMock()
        mock_upwork.find_jobs = AsyncMock(return_value=[])

        mock_fiverr = MagicMock()
        mock_fiverr.find_jobs = AsyncMock(return_value=[fiverr_job])

        with patch("janus_worker_core._HAS_PLATFORM_BROWSER", True), \
             patch("janus_worker_core.UpworkBrowser", return_value=mock_upwork), \
             patch("janus_worker_core.FiverrBrowser", return_value=mock_fiverr):
            summary = await wc.run_one_cycle()

        mock_fiverr.find_jobs.assert_awaited_once()
        assert isinstance(summary, CycleSummary)


# ---------------------------------------------------------------------------
# Task 12.3 — Quality gate integration test
# Requirements: 1.4, 14.3
# ---------------------------------------------------------------------------

class TestQualityGateIntegrationTest:
    """
    Task 12.3: Quality gate integration test.

    Mock AvusBrain.ask() to return low-quality work (score < 0.7) for all 3
    retries; assert PlatformBrowser.deliver() is never called and job is
    marked failed in DB.
    """

    @pytest.mark.asyncio
    async def test_low_quality_work_never_delivered_and_job_marked_failed(self):
        """
        When brain returns low-quality work for all retries, deliver/submit_work
        is never called and the job is marked failed in DB.

        Requirements: 1.4, 14.3
        """
        db = WorkerDatabase(":memory:")
        wallet = JanusWallet(db_path=":memory:")

        # Return a very short string that will score below 0.7 in QA
        # (< 50 chars → completeness=0.0, no keyword overlap, no code format)
        brain_ask = AsyncMock(return_value="bad")

        wc = _make_work_cycle(db, wallet, brain_ask)

        job = _make_browser_job(
            job_id="low-quality-job-001",
            title="Write a Python script",
            description="Create a script that processes CSV files",
            job_type="code",
        )

        async def mock_discover():
            return [job]

        wc._discover_jobs = mock_discover

        mock_upwork = MagicMock()
        mock_upwork.submit_work = MagicMock(return_value=False)
        mock_upwork.submit_work_async = AsyncMock(return_value=False)

        mock_fiverr = MagicMock()
        mock_fiverr.deliver_order = MagicMock(return_value=False)
        mock_fiverr.deliver_order_async = AsyncMock(return_value=False)

        with patch("janus_worker_core._HAS_PLATFORM_BROWSER", True), \
             patch("janus_worker_core.UpworkBrowser", return_value=mock_upwork), \
             patch("janus_worker_core.FiverrBrowser", return_value=mock_fiverr):
            summary = await wc.run_one_cycle()

        # submit_work and deliver_order must never be called
        mock_upwork.submit_work.assert_not_called()
        mock_fiverr.deliver_order.assert_not_called()

        # Job must be marked failed in DB
        record = db.get_job(job.id)
        assert record is not None, "Job record should exist in DB"
        assert record.status == JobStatus.FAILED.value, (
            f"Expected status 'failed', got '{record.status}'"
        )

    @pytest.mark.asyncio
    async def test_low_quality_work_no_income_recorded(self):
        """
        When QA fails, no income transaction is recorded in the wallet.

        Requirements: 1.4, 14.3
        """
        db = WorkerDatabase(":memory:")
        wallet = JanusWallet(db_path=":memory:")

        brain_ask = AsyncMock(return_value="bad")
        wc = _make_work_cycle(db, wallet, brain_ask)

        job = _make_browser_job(job_id="low-quality-job-002", job_type="code")

        async def mock_discover():
            return [job]

        wc._discover_jobs = mock_discover

        with patch("janus_worker_core._HAS_PLATFORM_BROWSER", True), \
             patch("janus_worker_core.UpworkBrowser", return_value=MagicMock()), \
             patch("janus_worker_core.FiverrBrowser", return_value=MagicMock()):
            await wc.run_one_cycle()

        income_txs = wallet._ledger.list_transactions(tx_type=TransactionType.INCOME)
        assert len(income_txs) == 0, (
            f"Expected no income transactions for failed job, got {len(income_txs)}"
        )

    @pytest.mark.asyncio
    async def test_quality_gate_all_three_retries_exhausted(self):
        """
        WorkGenerator retries up to MAX_RETRIES=3 times; brain.ask() is called
        at most 3 times and the job is still marked failed.

        Requirements: 1.4, 14.3
        """
        db = WorkerDatabase(":memory:")
        wallet = JanusWallet(db_path=":memory:")

        brain_ask = AsyncMock(return_value="bad")
        wc = _make_work_cycle(db, wallet, brain_ask)

        job = _make_browser_job(job_id="low-quality-job-003", job_type="code")

        async def mock_discover():
            return [job]

        wc._discover_jobs = mock_discover

        with patch("janus_worker_core._HAS_PLATFORM_BROWSER", True), \
             patch("janus_worker_core.UpworkBrowser", return_value=MagicMock()), \
             patch("janus_worker_core.FiverrBrowser", return_value=MagicMock()):
            await wc.run_one_cycle()

        # brain.ask() called at most MAX_RETRIES=3 times
        assert brain_ask.await_count <= 3, (
            f"Expected at most 3 brain.ask() calls, got {brain_ask.await_count}"
        )

        record = db.get_job(job.id)
        assert record is not None
        assert record.status == JobStatus.FAILED.value
