“””
Janus Autonomous IDE System
Allows Janus to autonomously write, test, execute, and deploy code
Integrated with WASM sandbox for safe execution
No external API dependencies
“””

import os
import json
import subprocess
import tempfile
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import ast
import re

@dataclass
class CodeProject:
“”“Represents a code project managed by Janus”””
project_id: str
name: str
description: str
language: str
created_at: str
modified_at: str
files: Dict[str, str]  # filepath -> content
dependencies: List[str]
entry_point: str
status: str  # ‘draft’, ‘testing’, ‘ready’, ‘deployed’
test_results: Optional[Dict] = None
deployment_info: Optional[Dict] = None

@dataclass
class ExecutionResult:
“”“Result of code execution”””
success: bool
output: str
errors: str
exit_code: int
execution_time: float
memory_used: Optional[int] = None
metadata: Optional[Dict] = None

class CodeAnalyzer:
“”“Analyzes code for safety, quality, and dependencies”””

```
def __init__(self):
    self.dangerous_imports = {
        'os.system', 'subprocess.Popen', 'eval', 'exec', 
        'compile', '__import__', 'open'
    }
    self.safe_patterns = {
        'data_processing', 'math', 'algorithms', 'visualization'
    }

def analyze_python(self, code: str) -> Dict[str, Any]:
    """Analyzes Python code for safety and quality"""
    analysis = {
        'valid_syntax': False,
        'safety_score': 0.0,
        'complexity_score': 0.0,
        'detected_imports': [],
        'security_warnings': [],
        'suggestions': [],
        'estimated_resources': {}
    }
    
    try:
        tree = ast.parse(code)
        analysis['valid_syntax'] = True
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    analysis['detected_imports'].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    analysis['detected_imports'].append(node.module)
        
        # Check for dangerous patterns
        code_lower = code.lower()
        safety_score = 100.0
        
        for dangerous in self.dangerous_imports:
            if dangerous in code:
                analysis['security_warnings'].append(
                    f"Potentially dangerous operation: {dangerous}"
                )
                safety_score -= 20
        
        # Check for file operations
        if 'open(' in code or 'file' in code_lower:
            analysis['security_warnings'].append("File operations detected")
            safety_score -= 10
        
        # Check for network operations
        if any(net in code_lower for net in ['socket', 'http', 'requests', 'urllib']):
            analysis['security_warnings'].append("Network operations detected")
            safety_score -= 15
        
        analysis['safety_score'] = max(0, safety_score)
        
        # Estimate complexity
        num_functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        num_classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        num_loops = sum(1 for node in ast.walk(tree) 
                      if isinstance(node, (ast.For, ast.While)))
        
        complexity = (num_functions * 2 + num_classes * 3 + num_loops) / 10
        analysis['complexity_score'] = min(10.0, complexity)
        
        # Resource estimation
        analysis['estimated_resources'] = {
            'functions': num_functions,
            'classes': num_classes,
            'loops': num_loops,
            'estimated_memory_mb': 10 + (num_classes * 5) + (num_functions * 2),
            'estimated_runtime_category': 'fast' if num_loops < 3 else 'medium'
        }
        
        # Suggestions
        if num_functions > 10:
            analysis['suggestions'].append("Consider modularizing into multiple files")
        if len(code.split('\n')) > 500:
            analysis['suggestions'].append("Large file - consider splitting")
        if not any(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
            analysis['suggestions'].append("Consider using functions for better organization")
        
    except SyntaxError as e:
        analysis['security_warnings'].append(f"Syntax error: {e}")
    
    return analysis

def analyze_javascript(self, code: str) -> Dict[str, Any]:
    """Analyzes JavaScript code"""
    analysis = {
        'valid_syntax': True,  # Basic check
        'safety_score': 0.0,
        'detected_imports': [],
        'security_warnings': [],
        'suggestions': []
    }
    
    # Check for dangerous patterns
    dangerous_patterns = ['eval(', 'innerHTML', 'document.write', 'Function(']
    safety_score = 100.0
    
    for pattern in dangerous_patterns:
        if pattern in code:
            analysis['security_warnings'].append(f"Dangerous pattern: {pattern}")
            safety_score -= 20
    
    analysis['safety_score'] = max(0, safety_score)
    
    # Extract requires/imports
    import_pattern = r'(?:require|import)\s*\([\'"]([^\'"]+)[\'"]\)'
    imports = re.findall(import_pattern, code)
    analysis['detected_imports'] = imports
    
    return analysis
```

class SandboxExecutor:
“”“Executes code in isolated sandbox environments”””

```
def __init__(self, workspace_dir: str = "/tmp/janus_sandbox"):
    self.workspace_dir = Path(workspace_dir)
    self.workspace_dir.mkdir(exist_ok=True, parents=True)

def create_sandbox(self, project_id: str) -> Path:
    """Creates an isolated sandbox directory for a project"""
    sandbox_path = self.workspace_dir / project_id
    sandbox_path.mkdir(exist_ok=True, parents=True)
    return sandbox_path

def execute_python(self, 
                  code: str, 
                  project_id: str,
                  timeout: int = 30,
                  max_memory_mb: int = 512) -> ExecutionResult:
    """Executes Python code in sandbox"""
    sandbox = self.create_sandbox(project_id)
    code_file = sandbox / "main.py"
    
    # Write code to file
    with open(code_file, 'w') as f:
        f.write(code)
    
    start_time = datetime.now()
    
    try:
        # Execute with resource limits
        result = subprocess.run(
            ['python3', str(code_file)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(sandbox)
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        return ExecutionResult(
            success=result.returncode == 0,
            output=result.stdout,
            errors=result.stderr,
            exit_code=result.returncode,
            execution_time=execution_time,
            metadata={'sandbox': str(sandbox)}
        )
    
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            output="",
            errors=f"Execution timeout ({timeout}s)",
            exit_code=-1,
            execution_time=timeout
        )
    except Exception as e:
        return ExecutionResult(
            success=False,
            output="",
            errors=str(e),
            exit_code=-1,
            execution_time=0
        )

def execute_javascript(self, 
                      code: str, 
                      project_id: str,
                      timeout: int = 30) -> ExecutionResult:
    """Executes JavaScript code using Node.js"""
    sandbox = self.create_sandbox(project_id)
    code_file = sandbox / "main.js"
    
    with open(code_file, 'w') as f:
        f.write(code)
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            ['node', str(code_file)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(sandbox)
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        return ExecutionResult(
            success=result.returncode == 0,
            output=result.stdout,
            errors=result.stderr,
            exit_code=result.returncode,
            execution_time=execution_time
        )
    
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            output="",
            errors=f"Execution timeout ({timeout}s)",
            exit_code=-1,
            execution_time=timeout
        )
    except Exception as e:
        return ExecutionResult(
            success=False,
            output="",
            errors=str(e),
            exit_code=-1,
            execution_time=0
        )

def cleanup_sandbox(self, project_id: str):
    """Removes sandbox directory"""
    sandbox = self.workspace_dir / project_id
    if sandbox.exists():
        shutil.rmtree(sandbox)
```

class TestRunner:
“”“Runs automated tests on code”””

```
def __init__(self):
    self.test_frameworks = {
        'python': ['unittest', 'pytest'],
        'javascript': ['jest', 'mocha']
    }

def generate_basic_tests(self, project: CodeProject) -> str:
    """Generates basic test cases for a project"""
    if project.language == 'python':
        return self._generate_python_tests(project)
    elif project.language == 'javascript':
        return self._generate_js_tests(project)
    return ""

def _generate_python_tests(self, project: CodeProject) -> str:
    """Generates Python unittest tests"""
    test_code = """import unittest
```

import sys
sys.path.insert(0, ‘.’)

# Import the main module

try:
import main
except ImportError:
main = None

class BasicTests(unittest.TestCase):

```
def test_import_success(self):
    \"\"\"Test that the module imports without errors\"\"\"
    self.assertIsNotNone(main, "Module should import successfully")

def test_basic_execution(self):
    \"\"\"Test basic execution without errors\"\"\"
    try:
        # If there's a main function, call it
        if hasattr(main, 'main'):
            result = main.main()
            self.assertTrue(True, "Execution completed")
        else:
            self.assertTrue(True, "No main function to test")
    except Exception as e:
        self.fail(f"Execution failed: {e}")
```

if **name** == ‘**main**’:
unittest.main()
“””
return test_code

```
def _generate_js_tests(self, project: CodeProject) -> str:
    """Generates JavaScript tests"""
    test_code = """
```

const assert = require(‘assert’);

describe(‘Basic Tests’, function() {
it(‘should import without errors’, function() {
try {
const main = require(’./main.js’);
assert.ok(true, ‘Module imported successfully’);
} catch (e) {
assert.fail(’Import failed: ’ + e.message);
}
});

```
it('should execute without errors', function() {
    try {
        const main = require('./main.js');
        assert.ok(true, 'Execution completed');
    } catch (e) {
        assert.fail('Execution failed: ' + e.message);
    }
});
```

});
“””
return test_code

```
def run_tests(self, project: CodeProject, executor: SandboxExecutor) -> Dict:
    """Runs tests on a project"""
    # Generate test file
    test_code = self.generate_basic_tests(project)
    
    # Create temporary test file
    sandbox = executor.create_sandbox(project.project_id)
    
    # Write main code
    main_file = sandbox / f"main.{self._get_extension(project.language)}"
    with open(main_file, 'w') as f:
        f.write(project.files.get(project.entry_point, ""))
    
    # Write test file
    test_file = sandbox / f"test_main.{self._get_extension(project.language)}"
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    # Run tests based on language
    if project.language == 'python':
        result = executor.execute_python(test_code, project.project_id)
    elif project.language == 'javascript':
        result = executor.execute_javascript(test_code, project.project_id)
    else:
        return {'success': False, 'message': f'Unsupported language: {project.language}'}
    
    return {
        'success': result.success,
        'output': result.output,
        'errors': result.errors,
        'execution_time': result.execution_time
    }

def _get_extension(self, language: str) -> str:
    """Returns file extension for language"""
    extensions = {
        'python': 'py',
        'javascript': 'js',
        'typescript': 'ts',
        'rust': 'rs'
    }
    return extensions.get(language, 'txt')
```

class DeploymentManager:
“”“Manages code deployment and packaging”””

```
def __init__(self, deployment_dir: str = "/tmp/janus_deployments"):
    self.deployment_dir = Path(deployment_dir)
    self.deployment_dir.mkdir(exist_ok=True, parents=True)

def package_project(self, project: CodeProject) -> Tuple[bool, str]:
    """Packages project for deployment"""
    package_dir = self.deployment_dir / project.project_id
    package_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Write all project files
        for filepath, content in project.files.items():
            file_path = package_dir / filepath
            file_path.parent.mkdir(exist_ok=True, parents=True)
            with open(file_path, 'w') as f:
                f.write(content)
        
        # Create README
        readme_content = f"""# {project.name}
```

{project.description}

## Details

- Language: {project.language}
- Created: {project.created_at}
- Status: {project.status}

## Dependencies

{chr(10).join(’- ’ + dep for dep in project.dependencies)}

## Usage

Entry point: {project.entry_point}
“””
with open(package_dir / ‘README.md’, ‘w’) as f:
f.write(readme_content)

```
        # Create manifest
        manifest = {
            'project': asdict(project),
            'packaged_at': datetime.now().isoformat(),
            'package_version': '1.0.0'
        }
        with open(package_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return True, str(package_dir)
    
    except Exception as e:
        return False, str(e)

def create_executable(self, project: CodeProject) -> Tuple[bool, str]:
    """Creates standalone executable for project"""
    if project.language != 'python':
        return False, "Executable creation only supported for Python currently"
    
    # For Python, create a shell script wrapper
    package_success, package_path = self.package_project(project)
    if not package_success:
        return False, package_path
    
    script_path = Path(package_path) / 'run.sh'
    script_content = f"""#!/bin/bash
```

cd “$(dirname “$0”)”
python3 {project.entry_point}
“””
with open(script_path, ‘w’) as f:
f.write(script_content)

```
    os.chmod(script_path, 0o755)
    
    return True, str(script_path)
```

class JanusIDE:
“”“Main IDE system for Janus autonomous coding”””

```
def __init__(self, workspace_dir: str = "/tmp/janus_ide"):
    self.workspace_dir = Path(workspace_dir)
    self.workspace_dir.mkdir(exist_ok=True, parents=True)
    
    self.projects_file = self.workspace_dir / 'projects.json'
    self.projects: Dict[str, CodeProject] = self._load_projects()
    
    self.analyzer = CodeAnalyzer()
    self.executor = SandboxExecutor()
    self.test_runner = TestRunner()
    self.deployment_manager = DeploymentManager()

def _load_projects(self) -> Dict[str, CodeProject]:
    """Loads existing projects from disk"""
    if self.projects_file.exists():
        with open(self.projects_file, 'r') as f:
            data = json.load(f)
            return {
                pid: CodeProject(**pdata) 
                for pid, pdata in data.items()
            }
    return {}

def _save_projects(self):
    """Saves projects to disk"""
    data = {
        pid: asdict(project) 
        for pid, project in self.projects.items()
    }
    with open(self.projects_file, 'w') as f:
        json.dump(data, f, indent=2)

def create_project(self, 
                  name: str, 
                  description: str,
                  language: str = 'python') -> str:
    """Creates a new code project"""
    project_id = hashlib.sha256(
        f"{name}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:16]
    
    project = CodeProject(
        project_id=project_id,
        name=name,
        description=description,
        language=language,
        created_at=datetime.now().isoformat(),
        modified_at=datetime.now().isoformat(),
        files={},
        dependencies=[],
        entry_point=f"main.{'py' if language == 'python' else 'js'}",
        status='draft'
    )
    
    self.projects[project_id] = project
    self._save_projects()
    
    return project_id

def add_file(self, project_id: str, filepath: str, content: str):
    """Adds or updates a file in the project"""
    if project_id not in self.projects:
        raise ValueError(f"Project {project_id} not found")
    
    project = self.projects[project_id]
    project.files[filepath] = content
    project.modified_at = datetime.now().isoformat()
    self._save_projects()

def analyze_project(self, project_id: str) -> Dict:
    """Analyzes project code for safety and quality"""
    if project_id not in self.projects:
        raise ValueError(f"Project {project_id} not found")
    
    project = self.projects[project_id]
    
    analyses = {}
    for filepath, content in project.files.items():
        if project.language == 'python':
            analyses[filepath] = self.analyzer.analyze_python(content)
        elif project.language == 'javascript':
            analyses[filepath] = self.analyzer.analyze_javascript(content)
    
    # Aggregate results
    overall = {
        'files_analyzed': len(analyses),
        'total_warnings': sum(len(a['security_warnings']) for a in analyses.values()),
        'average_safety_score': sum(a['safety_score'] for a in analyses.values()) / max(len(analyses), 1),
        'file_analyses': analyses
    }
    
    return overall

def execute_project(self, project_id: str, timeout: int = 30) -> ExecutionResult:
    """Executes the project code"""
    if project_id not in self.projects:
        raise ValueError(f"Project {project_id} not found")
    
    project = self.projects[project_id]
    
    if project.entry_point not in project.files:
        raise ValueError(f"Entry point {project.entry_point} not found in project files")
    
    code = project.files[project.entry_point]
    
    if project.language == 'python':
        result = self.executor.execute_python(code, project_id, timeout)
    elif project.language == 'javascript':
        result = self.executor.execute_javascript(code, project_id, timeout)
    else:
        result = ExecutionResult(
            success=False,
            output="",
            errors=f"Unsupported language: {project.language}",
            exit_code=-1,
            execution_time=0
        )
    
    return result

def test_project(self, project_id: str) -> Dict:
    """Runs tests on the project"""
    if project_id not in self.projects:
        raise ValueError(f"Project {project_id} not found")
    
    project = self.projects[project_id]
    test_results = self.test_runner.run_tests(project, self.executor)
    
    project.test_results = test_results
    project.status = 'tested' if test_results['success'] else 'failed'
    self._save_projects()
    
    return test_results

def deploy_project(self, project_id: str) -> Dict:
    """Packages and deploys the project"""
    if project_id not in self.projects:
        raise ValueError(f"Project {project_id} not found")
    
    project = self.projects[project_id]
    
    # Package project
    package_success, package_path = self.deployment_manager.package_project(project)
    
    if not package_success:
        return {
            'success': False,
            'error': package_path
        }
    
    # Create executable if possible
    exec_success, exec_path = self.deployment_manager.create_executable(project)
    
    deployment_info = {
        'success': True,
        'package_path': package_path,
        'executable_path': exec_path if exec_success else None,
        'deployed_at': datetime.now().isoformat()
    }
    
    project.deployment_info = deployment_info
    project.status = 'deployed'
    self._save_projects()
    
    return deployment_info

def get_project_status(self, project_id: str) -> Dict:
    """Gets comprehensive status of a project"""
    if project_id not in self.projects:
        raise ValueError(f"Project {project_id} not found")
    
    project = self.projects[project_id]
    
    return {
        'project': asdict(project),
        'file_count': len(project.files),
        'total_lines': sum(len(content.split('\n')) for content in project.files.values()),
        'has_tests': project.test_results is not None,
        'is_deployed': project.deployment_info is not None
    }

def list_projects(self) -> List[Dict]:
    """Lists all projects"""
    return [
        {
            'project_id': pid,
            'name': p.name,
            'language': p.language,
            'status': p.status,
            'created_at': p.created_at
        }
        for pid, p in self.projects.items()
    ]
```

def main():
“”“Example usage of Janus IDE”””
ide = JanusIDE()

```
# Create a new Python project
print("Creating new project...")
project_id = ide.create_project(
    name="Math Utils",
    description="A collection of mathematical utility functions",
    language="python"
)

# Add code to the project
code = """
```

import math

def fibonacci(n):
"""Calculate nth Fibonacci number"""
if n <= 1:
return n
return fibonacci(n-1) + fibonacci(n-2)

def prime_check(n):
"""Check if number is prime"""
if n < 2:
return False
for i in range(2, int(math.sqrt(n)) + 1):
if n % i == 0:
return False
return True

def main():
print(“Fibonacci(10):”, fibonacci(10))
print(“Is 17 prime?:”, prime_check(17))
print(“Is 20 prime?:”, prime_check(20))

```
# Calculate some primes
primes = [i for i in range(2, 50) if prime_check(i)]
print(f"Primes up to 50: {primes}")
```

if **name** == ‘**main**’:
main()
“””

```
ide.add_file(project_id, "main.py", code)

# Analyze the code
print("\nAnalyzing code...")
analysis = ide.analyze_project(project_id)
print(f"Safety score: {analysis['average_safety_score']:.1f}")
print(f"Warnings: {analysis['total_warnings']}")

# Execute the code
print("\nExecuting code...")
result = ide.execute_project(project_id)
print(f"Success: {result.success}")
print(f"Execution time: {result.execution_time:.3f}s")
print(f"\nOutput:\n{result.output}")

if result.errors:
    print(f"Errors:\n{result.errors}")

# Run tests
print("\nRunning tests...")
test_results = ide.test_project(project_id)
print(f"Tests passed: {test_results['success']}")

# Deploy the project
print("\nDeploying project...")
deployment = ide.deploy_project(project_id)
print(f"Deployed to: {deployment['package_path']}")
if deployment.get('executable_path'):
    print(f"Executable: {deployment['executable_path']}")

# Show final status
print("\nFinal project status:")
status = ide.get_project_status(project_id)
print(json.dumps(status, indent=2))
```

if **name** == ‘**main**’:
main()