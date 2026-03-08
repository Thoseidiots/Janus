import time
from task_execution_engine import TaskExecutionEngine
from structured_memory import StructuredMemory
from verification_critic import VerificationCritic

class JanusAgent:
    def __init__(self):
        self.memory = StructuredMemory()
        self.engine = TaskExecutionEngine()
        self.critic = VerificationCritic(self.memory)

    def run_task(self, task_id, command):
        self.memory.set_working_context("current_task", task_id)
        result = self.engine.execute(task_id, command)
        success, message = self.critic.verify(task_id, result)
        self.memory.add_episode(task_id, command, result, success)
        return success, message

if __name__ == "__main__":
    agent = JanusAgent()
    success, msg = agent.run_task("T001", "ls -l")
    print(f"Task T001: {msg}")
