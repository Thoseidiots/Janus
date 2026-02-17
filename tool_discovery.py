"""
tool_discovery.py
────────────────────────────────────────────────────────────
Dynamic tool discovery and on-the-fly tool generation for Janus.
Janus can discover existing tools, create new tools from descriptions,
and auto-generate code to handle new tasks.

Features:
- Tool discovery from code, APIs, and documentation
- Natural language to tool mapping
- Auto-generated tool code
- Tool validation and sandboxing
- Tool registry management
"""

import json
import re
import ast
import inspect
import textwrap
import subprocess
import tempfile
import hashlib
import time
import threading
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Callable, Any, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum


class ToolSource(Enum):
    """Source of discovered tool"""
    BUILTIN = "builtin"
    GENERATED = "generated"
    API = "api"
    PLUGIN = "plugin"
    USER_CREATED = "user_created"


@dataclass
class DiscoveredTool:
    """A discovered or generated tool"""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]  # name -> {type, description, required, default}
    return_type: str
    source: ToolSource
    handler: Optional[Callable] = None
    code: Optional[str] = None
    file_path: Optional[str] = None
    risk_level: str = "medium"  # low, medium, high
    tags: List[str] = field(default_factory=list)
    usage_count: int = 0
    success_rate: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary (excludes handler)"""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
            'return_type': self.return_type,
            'source': self.source.value,
            'code': self.code,
            'file_path': self.file_path,
            'risk_level': self.risk_level,
            'tags': self.tags,
            'usage_count': self.usage_count,
            'success_rate': self.success_rate,
            'created_at': self.created_at,
            'last_used': self.last_used,
        }


class ToolTemplate:
    """Templates for generating common tool types"""
    
    FILE_OPERATION = '''
def {name}(path: str, content: str = None) -> dict:
    """
    {description}
    
    Args:
        path: File path to operate on
        content: Content to write (for write operations)
    
    Returns:
        dict with 'success', 'content' or 'error'
    """
    from pathlib import Path
    
    try:
        p = Path(path)
        
        if content is not None:
            # Write operation
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return {{"success": True, "message": f"Written to {{path}}"}}
        else:
            # Read operation
            if not p.exists():
                return {{"success": False, "error": f"File not found: {{path}}"}}
            return {{"success": True, "content": p.read_text()}}
    
    except Exception as e:
        return {{"success": False, "error": str(e)}}
'''
    
    API_CALL = '''
def {name}(endpoint: str, method: str = "GET", data: dict = None, 
           headers: dict = None) -> dict:
    """
    {description}
    
    Args:
        endpoint: API endpoint URL
        method: HTTP method (GET, POST, PUT, DELETE)
        data: Request body data
        headers: Additional headers
    
    Returns:
        dict with 'success', 'status_code', 'response' or 'error'
    """
    import requests
    
    try:
        url = endpoint if endpoint.startswith('http') else f"{{base_url}}{{endpoint}}"
        
        response = requests.request(
            method=method.upper(),
            url=url,
            json=data,
            headers=headers,
            timeout=30
        )
        
        return {{
            "success": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
        }}
    
    except Exception as e:
        return {{"success": False, "error": str(e)}}
'''
    
    DATA_PROCESSING = '''
def {name}(data: list, operation: str = "process") -> dict:
    """
    {description}
    
    Args:
        data: List of data items to process
        operation: Type of operation to perform
    
    Returns:
        dict with 'success', 'result' or 'error'
    """
    try:
        result = []
        
        for item in data:
            # Process each item
            processed = item  # TODO: implement processing logic
            result.append(processed)
        
        return {{"success": True, "result": result, "count": len(result)}}
    
    except Exception as e:
        return {{"success": False, "error": str(e)}}
'''
    
    SHELL_COMMAND = '''
def {name}(command: str, args: list = None, cwd: str = None) -> dict:
    """
    {description}
    
    Args:
        command: Command to execute
        args: List of command arguments
        cwd: Working directory for command
    
    Returns:
        dict with 'success', 'stdout', 'stderr', 'returncode'
    """
    import subprocess
    from pathlib import Path
    
    try:
        cmd = [command] + (args or [])
        
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        return {{
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }}
    
    except Exception as e:
        return {{"success": False, "error": str(e)}}
'''


class CodeGenerator:
    """Generate Python code for new tools"""
    
    def __init__(self):
        self.templates = {
            'file': ToolTemplate.FILE_OPERATION,
            'api': ToolTemplate.API_CALL,
            'data': ToolTemplate.DATA_PROCESSING,
            'shell': ToolTemplate.SHELL_COMMAND,
        }
    
    def generate_from_description(self, name: str, description: str, 
                                   tool_type: str = "auto") -> str:
        """
        Generate tool code from natural language description
        
        Args:
            name: Tool name
            description: What the tool should do
            tool_type: 'file', 'api', 'data', 'shell', or 'auto'
        
        Returns:
            Generated Python code
        """
        if tool_type == "auto":
            tool_type = self._infer_tool_type(description)
        
        template = self.templates.get(tool_type, ToolTemplate.DATA_PROCESSING)
        
        # Generate code from template
        code = template.format(name=name, description=description)
        
        return code
    
    def _infer_tool_type(self, description: str) -> str:
        """Infer tool type from description"""
        desc_lower = description.lower()
        
        if any(w in desc_lower for w in ['file', 'read', 'write', 'save', 'load']):
            return 'file'
        elif any(w in desc_lower for w in ['api', 'http', 'request', 'fetch', 'url']):
            return 'api'
        elif any(w in desc_lower for w in ['command', 'shell', 'run', 'execute', 'script']):
            return 'shell'
        else:
            return 'data'
    
    def generate_from_signature(self, name: str, description: str,
                                 parameters: Dict[str, str],
                                 return_type: str = "dict") -> str:
        """
        Generate tool from explicit signature
        
        Args:
            name: Function name
            description: Function description
            parameters: {param_name: param_type}
            return_type: Return type annotation
        
        Returns:
            Generated Python code
        """
        # Build parameter list
        param_list = []
        for param_name, param_type in parameters.items():
            param_list.append(f"{param_name}: {param_type}")
        
        params_str = ", ".join(param_list)
        
        code = f'''
def {name}({params_str}) -> {return_type}:
    """
    {description}
    """
    try:
        # TODO: Implement tool logic
        result = None
        
        return {{"success": True, "result": result}}
    
    except Exception as e:
        return {{"success": False, "error": str(e)}}
'''
        return code
    
    def validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate generated code
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Check for function definition
            has_function = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
            if not has_function:
                return False, "Code must contain a function definition"
            
            # Check for dangerous operations
            dangerous = ['eval', 'exec', '__import__', 'os.system', 'subprocess.call']
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in dangerous:
                            return False, f"Dangerous operation detected: {node.func.id}"
            
            return True, None
        
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, str(e)


class ToolDiscoveryEngine:
    """
    Discover and generate tools for Janus
    """
    
    def __init__(self, tools_dir: str = "discovered_tools"):
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(exist_ok=True, parents=True)
        
        self.discovered_tools: Dict[str, DiscoveredTool] = {}
        self.code_generator = CodeGenerator()
        
        self._lock = threading.Lock()
        
        # Load existing discovered tools
        self._load_discovered_tools()
    
    def _load_discovered_tools(self):
        """Load previously discovered tools"""
        manifest_path = self.tools_dir / "discovered_tools.json"
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text())
                for tool_data in data.get('tools', []):
                    tool = DiscoveredTool(
                        name=tool_data['name'],
                        description=tool_data['description'],
                        parameters=tool_data['parameters'],
                        return_type=tool_data['return_type'],
                        source=ToolSource(tool_data['source']),
                        code=tool_data.get('code'),
                        file_path=tool_data.get('file_path'),
                        risk_level=tool_data.get('risk_level', 'medium'),
                        tags=tool_data.get('tags', []),
                    )
                    self.discovered_tools[tool.name] = tool
                print(f"[ToolDiscovery] Loaded {len(self.discovered_tools)} discovered tools")
            except Exception as e:
                print(f"[ToolDiscovery] Load error: {e}")
    
    def _save_discovered_tools(self):
        """Save discovered tools manifest"""
        manifest_path = self.tools_dir / "discovered_tools.json"
        
        data = {
            'saved_at': datetime.now().isoformat(),
            'tools': [tool.to_dict() for tool in self.discovered_tools.values()]
        }
        
        manifest_path.write_text(json.dumps(data, indent=2))
    
    def discover_from_module(self, module_name: str) -> List[DiscoveredTool]:
        """
        Discover tools from a Python module
        
        Args:
            module_name: Name of module to inspect
        
        Returns:
            List of discovered tools
        """
        discovered = []
        
        try:
            module = __import__(module_name)
            
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj):
                    # Check if function looks like a tool
                    if self._looks_like_tool(obj):
                        tool = self._function_to_tool(obj, ToolSource.PLUGIN)
                        discovered.append(tool)
                        
                        with self._lock:
                            self.discovered_tools[tool.name] = tool
            
            print(f"[ToolDiscovery] Discovered {len(discovered)} tools from {module_name}")
            
        except Exception as e:
            print(f"[ToolDiscovery] Error discovering from {module_name}: {e}")
        
        return discovered
    
    def _looks_like_tool(self, func: Callable) -> bool:
        """Check if a function looks like a tool"""
        # Has docstring
        if not func.__doc__:
            return False
        
        # Has parameters
        sig = inspect.signature(func)
        if len(sig.parameters) == 0:
            return False
        
        # Name doesn't start with underscore
        if func.__name__.startswith('_'):
            return False
        
        return True
    
    def _function_to_tool(self, func: Callable, source: ToolSource) -> DiscoveredTool:
        """Convert a function to a DiscoveredTool"""
        sig = inspect.signature(func)
        
        parameters = {}
        for name, param in sig.parameters.items():
            param_info = {
                'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any',
                'description': '',
                'required': param.default == inspect.Parameter.empty,
                'default': param.default if param.default != inspect.Parameter.empty else None
            }
            parameters[name] = param_info
        
        # Get return type
        return_type = str(sig.return_annotation) if sig.return_annotation != inspect.Parameter.empty else 'Any'
        
        # Get source code
        try:
            code = inspect.getsource(func)
        except:
            code = None
        
        return DiscoveredTool(
            name=func.__name__,
            description=func.__doc__ or f"Function {func.__name__}",
            parameters=parameters,
            return_type=return_type,
            source=source,
            handler=func,
            code=code,
            tags=['discovered']
        )
    
    def generate_tool(self, description: str, name: Optional[str] = None,
                      tool_type: str = "auto") -> Optional[DiscoveredTool]:
        """
        Generate a new tool from natural language description
        
        Args:
            description: What the tool should do
            name: Optional tool name (auto-generated if not provided)
            tool_type: Type of tool to generate
        
        Returns:
            Generated tool or None if generation failed
        """
        # Generate name if not provided
        if name is None:
            name = self._generate_name(description)
        
        print(f"[ToolDiscovery] Generating tool: {name}")
        
        # Generate code
        code = self.code_generator.generate_from_description(
            name, description, tool_type
        )
        
        # Validate code
        is_valid, error = self.code_generator.validate_code(code)
        if not is_valid:
            print(f"[ToolDiscovery] Code validation failed: {error}")
            return None
        
        # Save to file
        file_path = self.tools_dir / f"{name}.py"
        file_path.write_text(code)
        
        # Create tool object
        tool = DiscoveredTool(
            name=name,
            description=description,
            parameters={},  # Would be extracted from code
            return_type="dict",
            source=ToolSource.GENERATED,
            code=code,
            file_path=str(file_path),
            risk_level="medium",
            tags=['generated', tool_type]
        )
        
        # Try to compile and create handler
        try:
            namespace = {}
            exec(code, namespace)
            tool.handler = namespace.get(name)
        except Exception as e:
            print(f"[ToolDiscovery] Failed to compile handler: {e}")
        
        # Store tool
        with self._lock:
            self.discovered_tools[name] = tool
            self._save_discovered_tools()
        
        print(f"[ToolDiscovery] Generated tool: {name} -> {file_path}")
        return tool
    
    def _generate_name(self, description: str) -> str:
        """Generate a tool name from description"""
        # Extract key words
        words = re.findall(r'\b[a-zA-Z]+\b', description.lower())
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'to', 'and', 'or', 'for', 'in', 'on', 'at', 'from', 'by', 'with'}
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Take first 3 keywords
        name_words = keywords[:3]
        
        # Create snake_case name
        name = '_'.join(name_words)
        
        # Ensure uniqueness
        base_name = name
        counter = 1
        while name in self.discovered_tools:
            name = f"{base_name}_{counter}"
            counter += 1
        
        return name
    
    def find_tool_for_task(self, task_description: str) -> Optional[DiscoveredTool]:
        """
        Find the best tool for a given task
        
        Args:
            task_description: Description of what needs to be done
        
        Returns:
            Best matching tool or None
        """
        task_lower = task_description.lower()
        
        best_match = None
        best_score = 0
        
        for tool in self.discovered_tools.values():
            score = self._match_score(task_lower, tool)
            if score > best_score:
                best_score = score
                best_match = tool
        
        return best_match if best_score > 0.3 else None
    
    def _match_score(self, task: str, tool: DiscoveredTool) -> float:
        """Calculate match score between task and tool"""
        score = 0.0
        
        # Description similarity
        desc_words = set(tool.description.lower().split())
        task_words = set(task.split())
        
        if desc_words & task_words:
            score += len(desc_words & task_words) / len(desc_words | task_words)
        
        # Tag matching
        for tag in tool.tags:
            if tag in task:
                score += 0.2
        
        # Name matching
        if tool.name.replace('_', ' ') in task:
            score += 0.3
        
        # Usage bias (prefer frequently used tools)
        score += min(tool.usage_count * 0.01, 0.1)
        
        return score
    
    def execute_tool(self, name: str, args: dict) -> dict:
        """
        Execute a discovered tool
        
        Args:
            name: Tool name
            args: Arguments to pass to tool
        
        Returns:
            Tool execution result
        """
        tool = self.discovered_tools.get(name)
        if not tool:
            return {"success": False, "error": f"Tool not found: {name}"}
        
        if not tool.handler:
            return {"success": False, "error": f"Tool has no handler: {name}"}
        
        try:
            result = tool.handler(**args)
            
            # Update stats
            tool.usage_count += 1
            tool.last_used = datetime.now().isoformat()
            
            return result if isinstance(result, dict) else {"success": True, "result": result}
        
        except Exception as e:
            tool.success_rate = (tool.success_rate * tool.usage_count) / (tool.usage_count + 1)
            return {"success": False, "error": str(e)}
    
    def list_tools(self, tag: Optional[str] = None) -> List[Dict]:
        """List all discovered tools"""
        tools = []
        for tool in self.discovered_tools.values():
            if tag is None or tag in tool.tags:
                tools.append(tool.to_dict())
        return tools
    
    def get_tool(self, name: str) -> Optional[DiscoveredTool]:
        """Get a specific tool by name"""
        return self.discovered_tools.get(name)
    
    def delete_tool(self, name: str) -> bool:
        """Delete a discovered tool"""
        with self._lock:
            if name in self.discovered_tools:
                tool = self.discovered_tools[name]
                
                # Delete file if exists
                if tool.file_path:
                    try:
                        Path(tool.file_path).unlink()
                    except:
                        pass
                
                del self.discovered_tools[name]
                self._save_discovered_tools()
                return True
            return False


def main():
    """Demo of tool discovery"""
    print("=== Janus Tool Discovery Demo ===\n")
    
    discovery = ToolDiscoveryEngine()
    
    # Demo 1: Generate a file tool
    print("1. Generating file tool...")
    file_tool = discovery.generate_tool(
        "Read and write text files to the local filesystem",
        name="file_operations",
        tool_type="file"
    )
    if file_tool:
        print(f"   Generated: {file_tool.name}")
        print(f"   Code preview:\n{file_tool.code[:500]}...")
    
    # Demo 2: Generate an API tool
    print("\n2. Generating API tool...")
    api_tool = discovery.generate_tool(
        "Make HTTP requests to REST APIs with GET, POST, PUT, DELETE methods",
        name="api_request",
        tool_type="api"
    )
    if api_tool:
        print(f"   Generated: {api_tool.name}")
    
    # Demo 3: Find tool for task
    print("\n3. Finding tool for task...")
    task = "I need to save some text to a file"
    best_tool = discovery.find_tool_for_task(task)
    if best_tool:
        print(f"   Task: '{task}'")
        print(f"   Best match: {best_tool.name}")
    
    # Demo 4: List all tools
    print("\n4. All discovered tools:")
    for tool in discovery.list_tools():
        print(f"   - {tool['name']}: {tool['description'][:50]}...")
    
    # Demo 5: Execute generated tool
    print("\n5. Testing file tool...")
    if file_tool:
        result = discovery.execute_tool(
            file_tool.name,
            {"path": "/tmp/janus_test.txt", "content": "Hello from Janus!"}
        )
        print(f"   Result: {result}")
        
        # Read it back
        result = discovery.execute_tool(
            file_tool.name,
            {"path": "/tmp/janus_test.txt"}
        )
        print(f"   Read back: {result}")
    
    print("\n=== Demo complete ===")


if __name__ == "__main__":
    main()