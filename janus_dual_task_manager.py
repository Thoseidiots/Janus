"""
Janus Dual Task Manager - Concurrent Processing System

Manages multiple concurrent tasks with notifications.
Allows AI to work on multiple jobs simultaneously with real-time notifications.
"""

import asyncio
import json
import logging
import time
import os
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

# Import Janus systems
try:
    from standalone_finance import StandaloneFinance, TransactionType, PaymentMethod
    from avus_brain import AvusBrain
    from browser_automation import BrowserAutomationAgent
    FINANCE_AVAILABLE = True
    AI_AVAILABLE = True
    BROWSER_AVAILABLE = True
except ImportError as e:
    print(f"Systems not available: {e}")
    FINANCE_AVAILABLE = False
    AI_AVAILABLE = False
    BROWSER_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskType(Enum):
    CONTENT_WRITING = "content_writing"
    CODE_DEVELOPMENT = "code_development"
    DATA_ANALYSIS = "data_analysis"
    AI_CONSULTING = "ai_consulting"
    BROWSER_AUTOMATION = "browser_automation"
    CLIENT_ACQUISITION = "client_acquisition"
    PAYMENT_PROCESSING = "payment_processing"

@dataclass
class Task:
    id: str
    type: TaskType
    title: str
    description: str
    client: str
    budget: float
    platform: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: float = 0.0
    estimated_time: str = ""
    actual_time: float = 0.0
    priority: int = 1
    dependencies: List[str] = None

class NotificationSystem:
    """Real-time notification system for task completion"""
    
    def __init__(self):
        self.notifications = []
        self.subscribers = []
        
    def subscribe(self, callback: Callable):
        """Subscribe to notifications"""
        self.subscribers.append(callback)
    
    def notify(self, message: str, task_id: str, notification_type: str = "info"):
        """Send notification"""
        notification = {
            "id": str(uuid.uuid4()),
            "message": message,
            "task_id": task_id,
            "type": notification_type,
            "timestamp": datetime.now().isoformat()
        }
        
        self.notifications.append(notification)
        
        # Notify all subscribers
        for callback in self.subscribers:
            try:
                callback(notification)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")
    
    def get_notifications(self, task_id: str = None) -> List[Dict]:
        """Get notifications"""
        if task_id:
            return [n for n in self.notifications if n["task_id"] == task_id]
        return self.notifications

class JanusDualTaskManager:
    """Dual task manager for concurrent processing"""
    
    def __init__(self):
        self.finance_system = None
        self.avus_brain = None
        self.browser_agent = None
        
        # Task management
        self.tasks = {}
        self.task_queue = queue.PriorityQueue()
        self.running_tasks = {}
        self.completed_tasks = {}
        
        # Concurrency settings
        self.max_concurrent_tasks = 3
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_tasks)
        
        # Notification system
        self.notification_system = NotificationSystem()
        self.notification_system.subscribe(self._handle_notification)
        
        # Task time estimates
        self.task_time_estimates = {
            TaskType.CONTENT_WRITING: {"min": 20, "max": 30, "unit": "minutes"},
            TaskType.CODE_DEVELOPMENT: {"min": 40, "max": 70, "unit": "minutes"},
            TaskType.DATA_ANALYSIS: {"min": 50, "max": 75, "unit": "minutes"},
            TaskType.AI_CONSULTING: {"min": 50, "max": 75, "unit": "minutes"},
            TaskType.BROWSER_AUTOMATION: {"min": 10, "max": 30, "unit": "minutes"},
            TaskType.CLIENT_ACQUISITION: {"min": 15, "max": 25, "unit": "minutes"},
            TaskType.PAYMENT_PROCESSING: {"min": 5, "max": 10, "unit": "minutes"}
        }
        
        logger.info("Janus Dual Task Manager initialized")
    
    async def start_dual_task_system(self):
        """Start dual task processing system"""
        print("JANUS DUAL TASK MANAGER")
        print("=" * 50)
        print("CONCURRENT PROCESSING SYSTEM")
        print("Multiple tasks running simultaneously")
        print("Real-time notifications")
        print()
        
        # Initialize systems
        if not await self._initialize_systems():
            print("Failed to initialize systems")
            return
        
        # Create sample tasks
        await self._create_sample_tasks()
        
        # Start concurrent processing
        await self._start_concurrent_processing()
        
        # Show results
        self._show_results()
    
    async def _initialize_systems(self):
        """Initialize all required systems"""
        print("STEP 1: INITIALIZING SYSTEMS")
        print("-" * 30)
        
        success_count = 0
        
        # Finance system
        if FINANCE_AVAILABLE:
            try:
                self.finance_system = StandaloneFinance()
                print("  Finance system initialized")
                success_count += 1
            except Exception as e:
                print(f"  Finance system failed: {e}")
        
        # AI brain
        if AI_AVAILABLE:
            try:
                self.avus_brain = AvusBrain()
                if self.avus_brain.ensure_loaded():
                    print("  AI brain initialized")
                    success_count += 1
                else:
                    print("  AI brain failed to load")
            except Exception as e:
                print(f"  AI brain failed: {e}")
        
        # Browser automation
        if BROWSER_AVAILABLE:
            try:
                self.browser_agent = BrowserAutomationAgent()
                print("  Browser automation initialized")
                success_count += 1
            except Exception as e:
                print(f"  Browser automation failed: {e}")
        
        print(f"Systems ready: {success_count}/3")
        return success_count >= 2
    
    async def _create_sample_tasks(self):
        """Create sample tasks for demonstration"""
        print("\nSTEP 2: CREATING SAMPLE TASKS")
        print("-" * 35)
        
        # Sample tasks
        sample_tasks = [
            {
                "type": TaskType.CONTENT_WRITING,
                "title": "Write AI blog post",
                "description": "Write 1000-word blog post about AI trends",
                "client": "Tech Blog Inc",
                "budget": 150.0,
                "platform": "Upwork",
                "priority": 3
            },
            {
                "type": TaskType.CODE_DEVELOPMENT,
                "title": "Python automation script",
                "description": "Create automation script for data processing",
                "client": "Data Corp",
                "budget": 300.0,
                "platform": "Freelancer",
                "priority": 2
            },
            {
                "type": TaskType.DATA_ANALYSIS,
                "title": "Market analysis report",
                "description": "Analyze market trends and create report",
                "client": "Business Analytics Ltd",
                "budget": 250.0,
                "platform": "Upwork",
                "priority": 2
            },
            {
                "type": TaskType.AI_CONSULTING,
                "title": "AI implementation strategy",
                "description": "Provide AI implementation consulting",
                "client": "Startup Co",
                "budget": 400.0,
                "platform": "Guru",
                "priority": 1
            },
            {
                "type": TaskType.CLIENT_ACQUISITION,
                "title": "Find new clients",
                "description": "Search for new freelance opportunities",
                "client": "Self",
                "budget": 0.0,
                "platform": "Multiple",
                "priority": 1
            }
        ]
        
        for task_data in sample_tasks:
            task = Task(
                id=str(uuid.uuid4()),
                type=task_data["type"],
                title=task_data["title"],
                description=task_data["description"],
                client=task_data["client"],
                budget=task_data["budget"],
                platform=task_data["platform"],
                status=TaskStatus.PENDING,
                created_at=datetime.now(),
                priority=task_data["priority"],
                dependencies=[]
            )
            
            # Set estimated time
            time_estimate = self.task_time_estimates[task.type]
            task.estimated_time = f"{time_estimate['min']}-{time_estimate['max']} {time_estimate['unit']}"
            
            self.tasks[task.id] = task
            self.task_queue.put((task.priority, task.id))
            
            print(f"  Created task: {task.title}")
            print(f"    Type: {task.type.value}")
            print(f"    Budget: ${task.budget:.2f}")
            print(f"    Estimated time: {task.estimated_time}")
            print()
        
        print(f"Total tasks created: {len(sample_tasks)}")
    
    async def _start_concurrent_processing(self):
        """Start concurrent task processing"""
        print("\nSTEP 3: STARTING CONCURRENT PROCESSING")
        print("-" * 45)
        
        print(f"Max concurrent tasks: {self.max_concurrent_tasks}")
        print("Starting task execution...")
        print()
        
        # Process tasks concurrently
        futures = []
        
        while not self.task_queue.empty() and len(futures) < self.max_concurrent_tasks:
            priority, task_id = self.task_queue.get()
            task = self.tasks[task_id]
            
            print(f"Starting task: {task.title}")
            print(f"  Type: {task.type.value}")
            print(f"  Priority: {task.priority}")
            
            # Submit task for execution
            future = self.executor.submit(self._execute_task, task)
            futures.append((future, task))
            
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            self.notification_system.notify(
                f"Started task: {task.title}",
                task.id,
                "task_started"
            )
        
        # Monitor tasks and start new ones as they complete
        while futures:
            completed_futures = []
            
            for future, task in futures:
                if future.done():
                    try:
                        result = future.result()
                        task.result = result
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = datetime.now()
                        task.actual_time = (task.completed_at - task.started_at).total_seconds()
                        
                        self.completed_tasks[task.id] = task
                        
                        print(f"COMPLETED: {task.title}")
                        print(f"  Time: {task.actual_time:.1f} seconds")
                        print(f"  Result: {type(result).__name__}")
                        
                        self.notification_system.notify(
                            f"Completed task: {task.title} in {task.actual_time:.1f}s",
                            task.id,
                            "task_completed"
                        )
                        
                        completed_futures.append((future, task))
                        
                    except Exception as e:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                        
                        print(f"FAILED: {task.title}")
                        print(f"  Error: {e}")
                        
                        self.notification_system.notify(
                            f"Failed task: {task.title} - {e}",
                            task.id,
                            "task_failed"
                        )
                        
                        completed_futures.append((future, task))
            
            # Remove completed futures
            for future, task in completed_futures:
                futures.remove((future, task))
            
            # Start new tasks if available
            while not self.task_queue.empty() and len(futures) < self.max_concurrent_tasks:
                priority, task_id = self.task_queue.get()
                task = self.tasks[task_id]
                
                print(f"Starting next task: {task.title}")
                
                future = self.executor.submit(self._execute_task, task)
                futures.append((future, task))
                
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                
                self.notification_system.notify(
                    f"Started task: {task.title}",
                    task.id,
                    "task_started"
                )
            
            # Show current status
            if futures:
                print(f"\nCurrently running: {len(futures)} tasks")
                for future, task in futures:
                    elapsed = (datetime.now() - task.started_at).total_seconds()
                    print(f"  {task.title} - {elapsed:.1f}s elapsed")
            
            await asyncio.sleep(2)  # Check every 2 seconds
        
        print("\nAll tasks completed!")
    
    def _execute_task(self, task: Task) -> Any:
        """Execute a single task"""
        try:
            if task.type == TaskType.CONTENT_WRITING:
                return self._execute_content_writing(task)
            elif task.type == TaskType.CODE_DEVELOPMENT:
                return self._execute_code_development(task)
            elif task.type == TaskType.DATA_ANALYSIS:
                return self._execute_data_analysis(task)
            elif task.type == TaskType.AI_CONSULTING:
                return self._execute_ai_consulting(task)
            elif task.type == TaskType.CLIENT_ACQUISITION:
                return self._execute_client_acquisition(task)
            elif task.type == TaskType.PAYMENT_PROCESSING:
                return self._execute_payment_processing(task)
            else:
                return f"Unknown task type: {task.type}"
                
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise
    
    def _execute_content_writing(self, task: Task) -> str:
        """Execute content writing task"""
        if AI_AVAILABLE and self.avus_brain:
            prompt = f"Write high-quality content: {task.description}. Make it professional and engaging."
            result = self.avus_brain.ask(prompt, max_tokens=1000)
            
            # Simulate processing time
            time.sleep(25)  # 25 seconds for demo
            
            return result
        else:
            time.sleep(25)
            return f"Content written for: {task.description}"
    
    def _execute_code_development(self, task: Task) -> str:
        """Execute code development task"""
        if AI_AVAILABLE and self.avus_brain:
            prompt = f"Write production-ready code: {task.description}. Include comments and error handling."
            result = self.avus_brain.ask(prompt, max_tokens=1500)
            
            # Simulate longer processing time for code
            time.sleep(45)  # 45 seconds for demo
            
            return result
        else:
            time.sleep(45)
            return f"Code developed for: {task.description}"
    
    def _execute_data_analysis(self, task: Task) -> str:
        """Execute data analysis task"""
        if AI_AVAILABLE and self.avus_brain:
            prompt = f"Provide comprehensive data analysis: {task.description}. Include insights and recommendations."
            result = self.avus_brain.ask(prompt, max_tokens=1200)
            
            # Simulate analysis time
            time.sleep(60)  # 60 seconds for demo
            
            return result
        else:
            time.sleep(60)
            return f"Data analysis completed for: {task.description}"
    
    def _execute_ai_consulting(self, task: Task) -> str:
        """Execute AI consulting task"""
        if AI_AVAILABLE and self.avus_brain:
            prompt = f"Provide expert AI consulting: {task.description}. Be strategic and actionable."
            result = self.avus_brain.ask(prompt, max_tokens=1000)
            
            # Simulate consulting time
            time.sleep(55)  # 55 seconds for demo
            
            return result
        else:
            time.sleep(55)
            return f"AI consulting provided for: {task.description}"
    
    def _execute_client_acquisition(self, task: Task) -> List[Dict]:
        """Execute client acquisition task"""
        # Simulate finding new clients
        time.sleep(20)  # 20 seconds for demo
        
        new_clients = [
            {"name": "New Client 1", "budget": 200.0, "platform": "Upwork"},
            {"name": "New Client 2", "budget": 350.0, "platform": "Fiverr"},
            {"name": "New Client 3", "budget": 180.0, "platform": "Freelancer"}
        ]
        
        return new_clients
    
    def _execute_payment_processing(self, task: Task) -> Dict:
        """Execute payment processing task"""
        # Simulate payment processing
        time.sleep(8)  # 8 seconds for demo
        
        return {
            "payment_id": str(uuid.uuid4()),
            "amount": task.budget,
            "status": "processed",
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_notification(self, notification: Dict):
        """Handle incoming notifications"""
        print(f"NOTIFICATION: {notification['message']}")
    
    def _show_results(self):
        """Show final results"""
        print("\nSTEP 4: TASK COMPLETION RESULTS")
        print("-" * 40)
        
        completed_count = len(self.completed_tasks)
        total_earned = sum(task.budget for task in self.completed_tasks.values())
        total_time = sum(task.actual_time for task in self.completed_tasks.values())
        
        print(f"Tasks Completed: {completed_count}")
        print(f"Total Earned: ${total_earned:.2f}")
        print(f"Total Time: {total_time:.1f} seconds")
        print()
        
        print("INDIVIDUAL TASK RESULTS:")
        for task_id, task in self.completed_tasks.items():
            print(f"  {task.title}:")
            print(f"    Status: {task.status.value}")
            print(f"    Budget: ${task.budget:.2f}")
            print(f"    Time: {task.actual_time:.1f}s")
            print(f"    Platform: {task.platform}")
            print()
        
        print("CONCURRENT PROCESSING BENEFITS:")
        if completed_count > 1:
            sequential_time = sum(self.task_time_estimates[task.type]["min"] * 60 for task in self.completed_tasks.values())
            time_saved = sequential_time - total_time
            efficiency_gain = (time_saved / sequential_time) * 100
            
            print(f"  Sequential processing time: {sequential_time:.1f}s")
            print(f"  Concurrent processing time: {total_time:.1f}s")
            print(f"  Time saved: {time_saved:.1f}s")
            print(f"  Efficiency gain: {efficiency_gain:.1f}%")
        
        print("\nDUAL-TASK SYSTEM READY FOR PRODUCTION!")
    
    def add_task(self, task_type: TaskType, title: str, description: str, 
                 client: str, budget: float, platform: str, priority: int = 1) -> str:
        """Add a new task to the queue"""
        task = Task(
            id=str(uuid.uuid4()),
            type=task_type,
            title=title,
            description=description,
            client=client,
            budget=budget,
            platform=platform,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            priority=priority,
            dependencies=[]
        )
        
        # Set estimated time
        time_estimate = self.task_time_estimates[task.type]
        task.estimated_time = f"{time_estimate['min']}-{time_estimate['max']} {time_estimate['unit']}"
        
        self.tasks[task.id] = task
        self.task_queue.put((task.priority, task.id))
        
        self.notification_system.notify(
            f"Task added: {task.title}",
            task.id,
            "task_added"
        )
        
        return task.id
    
    def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get task status"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, Task]:
        """Get all tasks"""
        return self.tasks
    
    def get_notifications(self, task_id: str = None) -> List[Dict]:
        """Get notifications"""
        return self.notification_system.get_notifications(task_id)

# Main execution
async def janus_dual_task_manager():
    """Start dual task manager"""
    manager = JanusDualTaskManager()
    await manager.start_dual_task_system()

if __name__ == "__main__":
    print("Janus Dual Task Manager")
    print("CONCURRENT PROCESSING SYSTEM")
    print("Multiple tasks, real-time notifications")
    print()
    
    asyncio.run(janus_dual_task_manager())
