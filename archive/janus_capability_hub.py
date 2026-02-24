“””
Janus Capability Integration Layer
Connects the IDE, 3D face generator, and package manager to Janus core
Provides unified interface for autonomous operations
“””

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# Import Janus systems

sys.path.insert(0, str(Path(**file**).parent))

from autonomous_ide import JanusIDE, CodeProject, ExecutionResult
from advanced_3d_face_generator import ProceduralFaceGenerator, FacialFeatures
from package_manager import JanusPackageManager

@dataclass
class TaskRequest:
“”“Represents a task request to Janus”””
task_id: str
task_type: str  # ‘code’, ‘generate_face’, ‘package_management’
description: str
parameters: Dict[str, Any]
priority: int = 5
created_at: str = None

```
def __post_init__(self):
    if self.created_at is None:
        self.created_at = datetime.now().isoformat()
```

@dataclass
class TaskResult:
“”“Result of a completed task”””
task_id: str
success: bool
result_data: Any
execution_time: float
errors: Optional[str] = None
metadata: Optional[Dict] = None
completed_at: str = None

```
def __post_init__(self):
    if self.completed_at is None:
        self.completed_at = datetime.now().isoformat()
```

class JanusCapabilityHub:
“”“Central hub for all Janus capabilities”””

```
def __init__(self, workspace_dir: str = "/tmp/janus_hub"):
    self.workspace_dir = Path(workspace_dir)
    self.workspace_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize subsystems
    self.ide = JanusIDE(str(self.workspace_dir / "ide"))
    self.face_generator = ProceduralFaceGenerator()
    self.package_manager = JanusPackageManager()
    
    # Task tracking
    self.tasks_file = self.workspace_dir / "tasks.json"
    self.tasks: Dict[str, TaskRequest] = {}
    self.results: Dict[str, TaskResult] = {}
    self._load_tasks()
    
    # Capability registry
    self.capabilities = self._register_capabilities()

def _register_capabilities(self) -> Dict[str, Dict]:
    """Registers all available capabilities"""
    return {
        'code_project_create': {
            'description': 'Create a new code project',
            'parameters': ['name', 'description', 'language'],
            'handler': self._handle_create_project
        },
        'code_project_add_file': {
            'description': 'Add file to code project',
            'parameters': ['project_id', 'filepath', 'content'],
            'handler': self._handle_add_file
        },
        'code_project_analyze': {
            'description': 'Analyze code for safety and quality',
            'parameters': ['project_id'],
            'handler': self._handle_analyze_project
        },
        'code_project_execute': {
            'description': 'Execute code project',
            'parameters': ['project_id', 'timeout'],
            'handler': self._handle_execute_project
        },
        'code_project_test': {
            'description': 'Run tests on code project',
            'parameters': ['project_id'],
            'handler': self._handle_test_project
        },
        'code_project_deploy': {
            'description': 'Deploy code project',
            'parameters': ['project_id'],
            'handler': self._handle_deploy_project
        },
        'generate_face': {
            'description': 'Generate 3D face with features',
            'parameters': ['features', 'expressions', 'output_format'],
            'handler': self._handle_generate_face
        },
        'generate_face_animation': {
            'description': 'Generate face animation sequence',
            'parameters': ['features', 'timeline', 'fps'],
            'handler': self._handle_generate_animation
        },
        'package_search': {
            'description': 'Search for packages',
            'parameters': ['query'],
            'handler': self._handle_package_search
        },
        'package_install': {
            'description': 'Install a package',
            'parameters': ['name', 'version'],
            'handler': self._handle_package_install
        },
        'package_create': {
            'description': 'Create and publish new package',
            'parameters': ['name', 'version', 'description', 'files', 'language'],
            'handler': self._handle_package_create
        },
        'package_list': {
            'description': 'List installed or available packages',
            'parameters': ['list_type'],
            'handler': self._handle_package_list
        }
    }

def _load_tasks(self):
    """Loads task history from disk"""
    if self.tasks_file.exists():
        with open(self.tasks_file, 'r') as f:
            data = json.load(f)
            self.tasks = {
                tid: TaskRequest(**tdata)
                for tid, tdata in data.get('tasks', {}).items()
            }
            self.results = {
                tid: TaskResult(**rdata)
                for tid, rdata in data.get('results', {}).items()
            }

def _save_tasks(self):
    """Saves task history to disk"""
    data = {
        'tasks': {tid: asdict(t) for tid, t in self.tasks.items()},
        'results': {tid: asdict(r) for tid, r in self.results.items()}
    }
    with open(self.tasks_file, 'w') as f:
        json.dump(data, f, indent=2)

# === Code Project Handlers ===

def _handle_create_project(self, params: Dict) -> Dict:
    """Creates new code project"""
    project_id = self.ide.create_project(
        name=params['name'],
        description=params['description'],
        language=params.get('language', 'python')
    )
    return {
        'project_id': project_id,
        'message': f'Created project: {params["name"]}'
    }

def _handle_add_file(self, params: Dict) -> Dict:
    """Adds file to project"""
    self.ide.add_file(
        params['project_id'],
        params['filepath'],
        params['content']
    )
    return {'message': f'Added file: {params["filepath"]}'}

def _handle_analyze_project(self, params: Dict) -> Dict:
    """Analyzes project code"""
    analysis = self.ide.analyze_project(params['project_id'])
    return analysis

def _handle_execute_project(self, params: Dict) -> Dict:
    """Executes project code"""
    result = self.ide.execute_project(
        params['project_id'],
        timeout=params.get('timeout', 30)
    )
    return asdict(result)

def _handle_test_project(self, params: Dict) -> Dict:
    """Tests project code"""
    return self.ide.test_project(params['project_id'])

def _handle_deploy_project(self, params: Dict) -> Dict:
    """Deploys project"""
    return self.ide.deploy_project(params['project_id'])

# === 3D Face Generation Handlers ===

def _handle_generate_face(self, params: Dict) -> Dict:
    """Generates 3D face"""
    features_dict = params.get('features', {})
    features = FacialFeatures(**features_dict) if features_dict else None
    
    expressions = params.get('expressions', {})
    output_format = params.get('output_format', 'json')
    
    face_data = self.face_generator.generate_face(features, expressions)
    
    # Save to file
    output_dir = self.workspace_dir / "faces"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_format == 'json':
        output_file = output_dir / f"face_{timestamp}.json"
        self.face_generator.export_to_json(face_data, str(output_file))
    elif output_format == 'obj':
        output_file = output_dir / f"face_{timestamp}.obj"
        self.face_generator.export_to_obj(face_data, str(output_file))
    else:
        # Save both
        json_file = output_dir / f"face_{timestamp}.json"
        obj_file = output_dir / f"face_{timestamp}.obj"
        self.face_generator.export_to_json(face_data, str(json_file))
        self.face_generator.export_to_obj(face_data, str(obj_file))
        output_file = json_file
    
    return {
        'message': 'Face generated successfully',
        'output_file': str(output_file),
        'vertex_count': face_data['metadata']['vertex_count'],
        'face_count': face_data['metadata']['face_count']
    }

def _handle_generate_animation(self, params: Dict) -> Dict:
    """Generates face animation"""
    features_dict = params.get('features', {})
    features = FacialFeatures(**features_dict) if features_dict else FacialFeatures()
    
    timeline = params.get('timeline', [])
    fps = params.get('fps', 30)
    
    frames = self.face_generator.create_animation_sequence(
        features, timeline, fps
    )
    
    # Save animation
    output_dir = self.workspace_dir / "faces"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"animation_{timestamp}.json"
    
    animation_data = {
        'metadata': {
            'fps': fps,
            'frame_count': len(frames),
            'duration': len(frames) / fps
        },
        'frames': frames
    }
    
    with open(output_file, 'w') as f:
        json.dump(animation_data, f)
    
    return {
        'message': 'Animation generated successfully',
        'output_file': str(output_file),
        'frame_count': len(frames),
        'duration_seconds': len(frames) / fps
    }

# === Package Management Handlers ===

def _handle_package_search(self, params: Dict) -> Dict:
    """Searches for packages"""
    results = self.package_manager.search(params['query'])
    return {'packages': results}

def _handle_package_install(self, params: Dict) -> Dict:
    """Installs a package"""
    success, message = self.package_manager.install(
        params['name'],
        params.get('version', 'latest')
    )
    return {'success': success, 'message': message}

def _handle_package_create(self, params: Dict) -> Dict:
    """Creates and publishes new package"""
    package = self.package_manager.builder.create_package_from_code(
        name=params['name'],
        version=params['version'],
        description=params['description'],
        author=params.get('author', 'Janus'),
        language=params['language'],
        source_files=params['files'],
        dependencies=params.get('dependencies', [])
    )
    
    success = self.package_manager.publish(package)
    
    return {
        'success': success,
        'message': f'Published {params["name"]}@{params["version"]}'
    }

def _handle_package_list(self, params: Dict) -> Dict:
    """Lists packages"""
    list_type = params.get('list_type', 'installed')
    
    if list_type == 'installed':
        packages = self.package_manager.list_installed()
    else:
        packages = self.package_manager.list_available()
    
    return {'packages': packages}

# === Task Management ===

def submit_task(self, task: TaskRequest) -> str:
    """Submits a task for execution"""
    self.tasks[task.task_id] = task
    self._save_tasks()
    return task.task_id

def execute_task(self, task: TaskRequest) -> TaskResult:
    """Executes a task and returns result"""
    start_time = datetime.now()
    
    try:
        # Find capability handler
        capability = self.capabilities.get(task.task_type)
        
        if not capability:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                result_data=None,
                execution_time=0,
                errors=f"Unknown task type: {task.task_type}"
            )
        
        # Execute handler
        handler = capability['handler']
        result_data = handler(task.parameters)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        result = TaskResult(
            task_id=task.task_id,
            success=True,
            result_data=result_data,
            execution_time=execution_time
        )
        
        self.results[task.task_id] = result
        self._save_tasks()
        
        return result
    
    except Exception as e:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        result = TaskResult(
            task_id=task.task_id,
            success=False,
            result_data=None,
            execution_time=execution_time,
            errors=str(e)
        )
        
        self.results[task.task_id] = result
        self._save_tasks()
        
        return result

def get_task_result(self, task_id: str) -> Optional[TaskResult]:
    """Gets result of a completed task"""
    return self.results.get(task_id)

def list_capabilities(self) -> List[Dict]:
    """Lists all available capabilities"""
    return [
        {
            'name': name,
            'description': cap['description'],
            'parameters': cap['parameters']
        }
        for name, cap in self.capabilities.items()
    ]

def get_status(self) -> Dict:
    """Gets overall system status"""
    return {
        'capabilities': len(self.capabilities),
        'projects': len(self.ide.projects),
        'installed_packages': len(self.package_manager.installer.installed),
        'available_packages': len(self.package_manager.registry.index['packages']),
        'tasks_completed': len(self.results),
        'workspace': str(self.workspace_dir)
    }

def autonomous_workflow_example(self) -> Dict:
    """Demonstrates autonomous multi-step workflow"""
    workflow_results = []
    
    # Step 1: Create a code project
    print("Step 1: Creating code project...")
    task1 = TaskRequest(
        task_id="task_1",
        task_type="code_project_create",
        description="Create math utilities project",
        parameters={
            'name': 'Advanced Math Utils',
            'description': 'Advanced mathematical operations',
            'language': 'python'
        }
    )
    result1 = self.execute_task(task1)
    workflow_results.append(result1)
    project_id = result1.result_data['project_id']
    
    # Step 2: Add code to project
    print("Step 2: Adding code...")
    code = """
```

import math

def compute_pi(iterations=10000):
‘’‘Compute pi using Leibniz formula’’’
pi = 0
for i in range(iterations):
pi += ((-1)**i) / (2*i + 1)
return pi * 4

def matrix_multiply(a, b):
‘’‘Multiply two matrices’’’
rows_a, cols_a = len(a), len(a[0])
rows_b, cols_b = len(b), len(b[0])

```
if cols_a != rows_b:
    raise ValueError("Incompatible dimensions")

result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

for i in range(rows_a):
    for j in range(cols_b):
        for k in range(cols_a):
            result[i][j] += a[i][k] * b[k][j]

return result
```

def main():
print(f”Pi approximation: {compute_pi()}”)

```
# Test matrix multiplication
a = [[1, 2], [3, 4]]
b = [[5, 6], [7, 8]]
result = matrix_multiply(a, b)
print(f"Matrix product: {result}")
```

if **name** == ‘**main**’:
main()
“””
task2 = TaskRequest(
task_id=“task_2”,
task_type=“code_project_add_file”,
description=“Add main code file”,
parameters={
‘project_id’: project_id,
‘filepath’: ‘main.py’,
‘content’: code
}
)
result2 = self.execute_task(task2)
workflow_results.append(result2)

```
    # Step 3: Analyze code
    print("Step 3: Analyzing code...")
    task3 = TaskRequest(
        task_id="task_3",
        task_type="code_project_analyze",
        description="Analyze code safety",
        parameters={'project_id': project_id}
    )
    result3 = self.execute_task(task3)
    workflow_results.append(result3)
    
    # Step 4: Execute code
    print("Step 4: Executing code...")
    task4 = TaskRequest(
        task_id="task_4",
        task_type="code_project_execute",
        description="Execute project",
        parameters={'project_id': project_id, 'timeout': 10}
    )
    result4 = self.execute_task(task4)
    workflow_results.append(result4)
    
    # Step 5: Deploy
    print("Step 5: Deploying...")
    task5 = TaskRequest(
        task_id="task_5",
        task_type="code_project_deploy",
        description="Deploy project",
        parameters={'project_id': project_id}
    )
    result5 = self.execute_task(task5)
    workflow_results.append(result5)
    
    # Step 6: Generate a face
    print("Step 6: Generating 3D face...")
    task6 = TaskRequest(
        task_id="task_6",
        task_type="generate_face",
        description="Generate face for avatar",
        parameters={
            'features': {
                'head_width': 1.0,
                'eye_size': 1.2,
                'nose_length': 0.9
            },
            'expressions': {'smile': 0.7},
            'output_format': 'both'
        }
    )
    result6 = self.execute_task(task6)
    workflow_results.append(result6)
    
    return {
        'workflow': 'autonomous_demo',
        'steps_completed': len(workflow_results),
        'all_successful': all(r.success for r in workflow_results),
        'results': [asdict(r) for r in workflow_results]
    }
```

def main():
“”“Demo of Janus capability hub”””
print(”=== Janus Capability Hub ===\n”)

```
hub = JanusCapabilityHub()

# Show available capabilities
print("Available capabilities:")
for cap in hub.list_capabilities():
    print(f"  - {cap['name']}: {cap['description']}")

print("\n" + "="*50 + "\n")

# Run autonomous workflow
print("Running autonomous workflow demonstration...\n")
workflow_result = hub.autonomous_workflow_example()

print("\n" + "="*50 + "\n")
print("Workflow Summary:")
print(f"  Steps completed: {workflow_result['steps_completed']}")
print(f"  All successful: {workflow_result['all_successful']}")

print("\nStep details:")
for i, result in enumerate(workflow_result['results'], 1):
    print(f"\n  Step {i}: {result['success']}")
    if result['success']:
        print(f"    Execution time: {result['execution_time']:.3f}s")
    else:
        print(f"    Error: {result['errors']}")

# Show system status
print("\n" + "="*50 + "\n")
print("System Status:")
status = hub.get_status()
for key, value in status.items():
    print(f"  {key}: {value}")
```

if **name** == ‘**main**’:
main()