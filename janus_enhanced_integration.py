"""
janus_enhanced_integration.py
==============================
Hooks Janus's new capabilities into the existing core systems:

    consciousness.py        — decision-making and goal planning
    memory.py               — experience persistence
    proxy_layer.py          — request filtering
    moral_core_immutable.py — ethical constraints

All existing-system imports are optional; the module runs in standalone
mode if they are unavailable.
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

# ── Optional existing-system imports ─────────────────────────────────────────

try:
    from consciousness import Consciousness
except ImportError:
    Consciousness = None  # type: ignore[assignment,misc]

try:
    from memory import Memory
except ImportError:
    Memory = None  # type: ignore[assignment,misc]

try:
    from proxy_layer import ProxyLayer
except ImportError:
    ProxyLayer = None  # type: ignore[assignment,misc]

try:
    from moral_core_immutable import MoralCore
except ImportError:
    MoralCore = None  # type: ignore[assignment,misc]


# ── EnhancedJanusCore ─────────────────────────────────────────────────────────

class EnhancedJanusCore:
    """
    Integrates new Janus capabilities with existing core systems.

    Pipeline for each request:
        1. Proxy layer filtering  (if available)
        2. Consciousness approval (if available)
        3. Moral core validation  (if available)
        4. Capability execution
        5. Memory storage         (if available)
    """

    def __init__(
        self,
        use_existing_systems: bool = True,
        workspace_dir: str = "/tmp/janus_enhanced",
    ) -> None:
        self.workspace_dir = workspace_dir
        self.session_id    = self._make_session_id()
        self.action_history: List[Dict] = []

        # Wire existing systems if requested and available
        all_present = all([Consciousness, Memory, ProxyLayer, MoralCore])
        self.use_existing = use_existing_systems and all_present

        if self.use_existing:
            print("Connecting to existing Janus systems...")
            self.consciousness = Consciousness() if Consciousness else None
            self.memory        = Memory()        if Memory        else None
            self.proxy         = ProxyLayer()    if ProxyLayer    else None
            self.moral_core    = MoralCore()     if MoralCore     else None
        else:
            print("Running in standalone mode (existing systems not available).")
            self.consciousness = None
            self.memory        = None
            self.proxy         = None
            self.moral_core    = None

    # ── request pipeline ──────────────────────────────────────────────────────

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a capability request through the full Janus pipeline.

        request keys:
            capability_type (str)  — e.g. 'code_project_create'
            description     (str)  — human-readable intent
            parameters      (dict) — capability-specific params
            priority        (int)  — 1–10, default 5
        """
        # 1. Proxy filtering
        if self.proxy:
            filtered = self.proxy.filter_request(request)
            if filtered.get("blocked"):
                return {
                    "success": False,
                    "error":   "Request blocked by proxy layer",
                    "reason":  filtered.get("reason"),
                }
            request = filtered

        # 2. Consciousness approval
        if self.consciousness:
            decision = self.consciousness.evaluate_action(request)
            if not decision.get("approved"):
                return {
                    "success": False,
                    "error":   "Action not approved by consciousness layer",
                    "reason":  decision.get("reason"),
                }

        # 3. Moral core validation
        if self.moral_core:
            check = self.moral_core.validate_action(request)
            if not check.get("valid"):
                return {
                    "success":    False,
                    "error":      "Action violates moral core constraints",
                    "violations": check.get("violations"),
                }

        # 4. Execute capability
        result = self._execute_capability(request)

        # 5. Store in memory
        if self.memory and result.get("success"):
            try:
                self.memory.store_experience({
                    "session_id": self.session_id,
                    "request":    request,
                    "result":     result,
                    "timestamp":  datetime.now().isoformat(),
                })
            except Exception:
                pass

        self.action_history.append({
            "type":      request.get("capability_type"),
            "success":   result.get("success"),
            "timestamp": datetime.now().isoformat(),
        })
        return result

    def _execute_capability(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatch to the appropriate capability handler.
        Extend this method to add new capability types.
        """
        cap_type = request.get("capability_type", "")
        params   = request.get("parameters", {})

        # Code project capabilities
        if cap_type == "code_project_create":
            return self._cap_create_project(params)
        if cap_type == "code_project_add_file":
            return self._cap_add_file(params)
        if cap_type == "code_project_analyze":
            return self._cap_analyze(params)
        if cap_type == "code_project_execute":
            return self._cap_execute(params)
        if cap_type == "code_project_deploy":
            return self._cap_deploy(params)

        return {"success": False, "error": f"Unknown capability: {cap_type}"}

    # ── capability handlers ───────────────────────────────────────────────────

    def _cap_create_project(self, params: Dict) -> Dict:
        try:
            from autonomous_ide import JanusIDE
            ide = JanusIDE(self.workspace_dir)
            pid = ide.create_project(
                name=params.get("name", "Unnamed"),
                description=params.get("description", ""),
                language=params.get("language", "python"),
            )
            return {"success": True, "result_data": {"project_id": pid}}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _cap_add_file(self, params: Dict) -> Dict:
        try:
            from autonomous_ide import JanusIDE
            ide = JanusIDE(self.workspace_dir)
            ide.add_file(
                params["project_id"],
                params.get("filepath", "main.py"),
                params.get("content", ""),
            )
            return {"success": True, "result_data": {}}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _cap_analyze(self, params: Dict) -> Dict:
        try:
            from autonomous_ide import JanusIDE
            ide = JanusIDE(self.workspace_dir)
            analysis = ide.analyze_project(params["project_id"])
            return {"success": True, "result_data": analysis}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _cap_execute(self, params: Dict) -> Dict:
        try:
            from autonomous_ide import JanusIDE
            ide = JanusIDE(self.workspace_dir)
            result = ide.execute_project(
                params["project_id"],
                timeout=params.get("timeout", 30),
            )
            return {
                "success":     result.success,
                "result_data": {
                    "output":         result.output,
                    "errors":         result.errors,
                    "execution_time": result.execution_time,
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _cap_deploy(self, params: Dict) -> Dict:
        try:
            from autonomous_ide import JanusIDE
            ide = JanusIDE(self.workspace_dir)
            info = ide.deploy_project(params["project_id"])
            return {"success": info.get("success", False), "result_data": info}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── high-level helpers ────────────────────────────────────────────────────

    def autonomous_code_generation(self, goal: str) -> Dict[str, Any]:
        """
        Full autonomous pipeline: create project → add code → analyse →
        execute (if safe) → deploy.
        """
        results: List[Dict] = []

        # Create project
        r = self.process_request({
            "capability_type": "code_project_create",
            "description":     f"Create project for: {goal}",
            "parameters": {
                "name":        f"Auto: {goal[:40]}",
                "description": goal,
                "language":    "python",
            },
        })
        if not r["success"]:
            return {"success": False, "error": "Failed to create project", "results": results}
        project_id = r["result_data"]["project_id"]
        results.append({"step": "create_project", "success": True})

        # Add generated code
        code = self._generate_code_for_goal(goal)
        r = self.process_request({
            "capability_type": "code_project_add_file",
            "description":     "Add generated code",
            "parameters": {
                "project_id": project_id,
                "filepath":   "main.py",
                "content":    code,
            },
        })
        results.append({"step": "add_code", "success": r["success"]})

        # Analyse
        r = self.process_request({
            "capability_type": "code_project_analyze",
            "description":     "Analyse code safety",
            "parameters":      {"project_id": project_id},
        })
        safety = r.get("result_data", {}).get("average_safety_score", 0) if r["success"] else 0
        results.append({"step": "analyze", "success": r["success"], "safety_score": safety})

        # Execute only if safe
        if safety >= 70:
            r = self.process_request({
                "capability_type": "code_project_execute",
                "description":     "Execute code",
                "parameters":      {"project_id": project_id, "timeout": 30},
            })
            results.append({"step": "execute", "success": r["success"]})

            if r["success"]:
                r = self.process_request({
                    "capability_type": "code_project_deploy",
                    "description":     "Deploy project",
                    "parameters":      {"project_id": project_id},
                })
                results.append({"step": "deploy", "success": r["success"]})
        else:
            results.append({"step": "execute", "success": False, "reason": "Safety score too low"})

        return {
            "success":    all(s["success"] for s in results),
            "project_id": project_id,
            "results":    results,
        }

    def _generate_code_for_goal(self, goal: str) -> str:
        """
        Generate a minimal Python stub for a goal.
        In production this would call the Janus LLM (janus_gpt.py).
        """
        return (
            f'"""\nAuto-generated for goal: {goal}\n"""\n\n'
            f"def main():\n"
            f"    print('Executing: {goal}')\n\n"
            f"if __name__ == '__main__':\n"
            f"    main()\n"
        )

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the current system state."""
        return {
            "session_id":       self.session_id,
            "actions_performed": len(self.action_history),
            "recent_actions":   self.action_history[-5:],
            "integrated_systems": {
                "consciousness": self.consciousness is not None,
                "memory":        self.memory        is not None,
                "proxy":         self.proxy         is not None,
                "moral_core":    self.moral_core    is not None,
            },
        }

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_session_id() -> str:
        return hashlib.sha256(datetime.now().isoformat().encode()).hexdigest()[:16]


# ── Standalone demo ───────────────────────────────────────────────────────────

def main() -> None:
    print("=== Janus Enhanced Integration Demo ===\n")
    core = EnhancedJanusCore(use_existing_systems=False)

    print("System status:")
    print(json.dumps(core.get_status(), indent=2))

    print("\n--- Autonomous code generation ---")
    result = core.autonomous_code_generation("Create a prime number checker")
    print(f"Success: {result['success']}")
    for step in result.get("results", []):
        print(f"  {step['step']}: {'OK' if step['success'] else 'FAIL'}")


if __name__ == "__main__":
    main()
