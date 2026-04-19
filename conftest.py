# Root-level conftest.py
# This file prevents pytest from trying to import the workspace root __init__.py
# as a package when collecting tests from the tests/ subdirectory.
import sys
import os
import types

collect_ignore_glob = ["__init__.py"]

# Ensure the workspace root is on sys.path so janus_computer_use is importable
_workspace = os.path.dirname(os.path.abspath(__file__))
if _workspace not in sys.path:
    sys.path.insert(0, _workspace)


def pytest_configure(config):
    """
    Monkey-patch _pytest.python.Package.setup to skip importing the broken
    workspace root __init__.py. This is needed because the workspace root
    contains a __init__.py with relative imports that fail when imported
    as a top-level module.
    """
    import _pytest.python as _pytest_python
    from pathlib import Path

    _original_setup = _pytest_python.Package.setup

    def _patched_setup(self):
        init_py = self.path / "__init__.py"
        # Skip importing the workspace root __init__.py if it would fail
        if init_py == Path(_workspace) / "__init__.py":
            return  # Skip — the root __init__.py has broken relative imports
        _original_setup(self)

    _pytest_python.Package.setup = _patched_setup
