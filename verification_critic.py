class VerificationCritic:
    def __init__(self, memory):
        self.memory = memory

    def verify(self, task_id, result):
        if isinstance(result, dict) and result.get("success"):
            return True, "Task completed successfully."
        return False, "Task failed or produced unexpected output."

    def adapt_plan(self, failed_task_id, error_message):
        return f"Plan adaptation required for {failed_task_id}: {error_message}"
