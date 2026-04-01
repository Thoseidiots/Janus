try:
    from .base import RepairPlugin
    from .javascript_typescript_plugin import JavaScriptTypeScriptRepairPlugin
    from .python_plugin import PythonRepairPlugin
except ImportError:  # pragma: no cover - top-level package fallback
    from analysis.repair_plugins.base import RepairPlugin
    from analysis.repair_plugins.javascript_typescript_plugin import JavaScriptTypeScriptRepairPlugin
    from analysis.repair_plugins.python_plugin import PythonRepairPlugin


BUILTIN_REPAIR_PLUGINS = [
    PythonRepairPlugin(),
    JavaScriptTypeScriptRepairPlugin(),
]

__all__ = [
    "RepairPlugin",
    "PythonRepairPlugin",
    "JavaScriptTypeScriptRepairPlugin",
    "BUILTIN_REPAIR_PLUGINS",
]
