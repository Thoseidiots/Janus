import subprocess
import shlex
import logging
import time

class TaskExecutionEngine:
    def __init__(self, workspace_dir="."):
        self.workspace_dir = workspace_dir
        self.logger = logging.getLogger("TaskExecutionEngine")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

    def execute(self, task_id, command, expected_output_pattern=None):
        self.logger.info(f"Executing task {task_id}: {command}")
        start_time = time.time()
        try:
            result = subprocess.run(
                shlex.split(command),
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            output = result.stdout + result.stderr
            success = (result.returncode == 0)
            
            if success and expected_output_pattern:
                import re
                if not re.search(expected_output_pattern, output):
                    success = False
                    self.logger.warning(f"Task {task_id} failed validation pattern.")

            return {
                "task_id": task_id,
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "duration": time.time() - start_time
            }
        except Exception as e:
            self.logger.error(f"Task {task_id} failed with error: {str(e)}")
            return {
                "task_id": task_id,
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }
