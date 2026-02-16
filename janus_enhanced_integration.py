“””
Janus Enhanced Capability Integration
Hooks the new capabilities into existing Janus systems:

- consciousness.py for decision-making
- memory.py for state persistence
- proxy_layer.py for request filtering
- moral_core_immutable.py for ethical constraints
  “””

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Import new capabilities

from janus_capability_hub import JanusCapabilityHub, TaskRequest, TaskResult

# Import existing Janus systems

try:
from consciousness import Consciousness
from memory import Memory
from proxy_layer import ProxyLayer
from moral_core_immutable import MoralCore
except ImportError as e:
print(f”Warning: Could not import existing Janus modules: {e}”)
print(“Running in standalone mode…”)
Consciousness = None
Memory = None
ProxyLayer = None
MoralCore = None

class EnhancedJanusCore:
“””
Enhanced Janus Core that integrates new capabilities with existing systems
“””

```
def __init__(self, 
             use_existing_systems: bool = True,
             workspace_dir: str = "/tmp/janus_enhanced"):
    
    # Initialize new capability hub
    self.capabilities = JanusCapabilityHub(workspace_dir)
    
    # Connect to existing Janus systems if available
    self.use_existing = use_existing_systems and all([
        Consciousness, Memory, ProxyLayer, MoralCore
    ])
    
    if self.use_existing:
        print("Connecting to existing Janus systems...")
        self.consciousness = Consciousness() if Consciousness else None
        self.memory = Memory() if Memory else None
        self.proxy = ProxyLayer() if ProxyLayer else None
        self.moral_core = MoralCore() if MoralCore else None
    else:
        print("Running with new capabilities only...")
        self.consciousness = None
        self.memory = None
        self.proxy = None
        self.moral_core = None
    
    # State tracking
    self.session_id = self._generate_session_id()
    self.action_history = []

def _generate_session_id(self) -> str:
    """Generate unique session ID"""
    import hashlib
    return hashlib.sha256(
        datetime.now().isoformat().encode()
    ).hexdigest()[:16]

def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a request through the full Janus pipeline:
    1. Proxy layer filtering (if available)
    2. Consciousness decision-making (if available)
    3. Moral core validation (if available)
    4. Capability execution
    5. Memory storage (if available)
    """
    
    # Step 1: Proxy layer filtering
    if self.proxy:
        filtered_request = self.proxy.filter_request(request)
        if filtered_request.get('blocked'):
            return {
                'success': False,
                'error': 'Request blocked by proxy layer',
                'reason': filtered_request.get('reason')
            }
        request = filtered_request
    
    # Step 2: Consciousness evaluation
    if self.consciousness:
        decision = self.consciousness.evaluate_action(request)
        if not decision.get('approved'):
            return {
                'success': False,
                'error': 'Action not approved by consciousness layer',
                'reason': decision.get('reason')
            }
    
    # Step 3: Moral core validation
    if self.moral_core:
        moral_check = self.moral_core.validate_action(request)
        if not moral_check.get('valid'):
            return {
                'success': False,
                'error': 'Action violates moral core constraints',
                'violations': moral_check.get('violations')
            }
    
    # Step 4: Execute capability
    task = TaskRequest(
        task_id=f"{self.session_id}_{len(self.action_history)}",
        task_type=request['capability_type'],
        description=request.get('description', ''),
        parameters=request.get('parameters', {}),
        priority=request.get('priority', 5)
    )
    
    result = self.capabilities.execute_task(task)
    
    # Step 5: Store in memory
    if self.memory and result.success:
        self.memory.store_experience({
            'session_id': self.session_id,
            'task': task.__dict__,
            'result': result.__dict__,
            'timestamp': datetime.now().isoformat()
        })
    
    # Track action
    self.action_history.append({
        'task_id': task.task_id,
        'type': task.task_type,
        'success': result.success,
        'timestamp': datetime.now().isoformat()
    })
    
    return {
        'success': result.success,
        'task_id': result.task_id,
        'result_data': result.result_data,
        'errors': result.errors,
        'execution_time': result.execution_time
    }

def autonomous_code_generation(self, goal: str) -> Dict[str, Any]:
    """
    Autonomous code generation using consciousness for planning
    and capabilities for execution
    """
    
    # Use consciousness to plan if available
    if self.consciousness:
        plan = self.consciousness.plan_for_goal(goal)
    else:
        # Simple default plan
        plan = {
            'steps': [
                'create_project',
                'generate_code',
                'analyze_code',
                'execute_code',
                'deploy'
            ]
        }
    
    results = []
    project_id = None
    
    # Step 1: Create project
    result = self.process_request({
        'capability_type': 'code_project_create',
        'description': f'Create project for: {goal}',
        'parameters': {
            'name': f'Autonomous: {goal[:30]}',
            'description': goal,
            'language': 'python'
        }
    })
    
    if result['success']:
        project_id = result['result_data']['project_id']
        results.append({'step': 'create_project', 'success': True})
    else:
        return {'success': False, 'error': 'Failed to create project', 'results': results}
    
    # Step 2: Generate code (would use your consciousness/LLM here)
    code = self._generate_code_for_goal(goal)
    
    result = self.process_request({
        'capability_type': 'code_project_add_file',
        'description': 'Add generated code',
        'parameters': {
            'project_id': project_id,
            'filepath': 'main.py',
            'content': code
        }
    })
    results.append({'step': 'add_code', 'success': result['success']})
    
    # Step 3: Analyze
    result = self.process_request({
        'capability_type': 'code_project_analyze',
        'description': 'Analyze code safety',
        'parameters': {'project_id': project_id}
    })
    
    safety_score = result['result_data'].get('average_safety_score', 0) if result['success'] else 0
    results.append({'step': 'analyze', 'success': result['success'], 'safety_score': safety_score})
    
    # Step 4: Execute (only if safe)
    if safety_score > 70:
        result = self.process_request({
            'capability_type': 'code_project_execute',
            'description': 'Execute code',
            'parameters': {'project_id': project_id, 'timeout': 30}
        })
        results.append({'step': 'execute', 'success': result['success']})
        
        # Step 5: Deploy if successful
        if result['success']:
            result = self.process_request({
                'capability_type': 'code_project_deploy',
                'description': 'Deploy project',
                'parameters': {'project_id': project_id}
            })
            results.append({'step': 'deploy', 'success': result['success']})
    else:
        results.append({'step': 'execute', 'success': False, 'reason': 'Safety score too low'})
    
    return {
        'success': all(r['success'] for r in results),
        'project_id': project_id,
        'results': results
    }

def _generate_code_for_goal(self, goal: str) -> str:
    """
    Generate code for a goal
    In a real implementation, this would use your custom LLM
    For now, returns a template
    """
    # This is where you'd call your custom Janus LLM
    # For now, return a simple template
    return f'''
```

“””
Auto-generated code for goal: {goal}
Generated by Janus Enhanced System
“””

def main():
print(“Executing goal: {goal}”)
# TODO: Implement goal logic
return “Goal completed”

if **name** == ‘**main**’:
result = main()
print(f”Result: {{result}}”)
‘’’

```
def generate_avatar_face(self, personality_traits: Dict[str, float]) -> Dict[str, Any]:
    """
    Generate 3D face avatar based on personality traits
    Uses the existing knowledge_3d_face_generator.json if available
    """
    
    # Map personality traits to facial features
    features = self._personality_to_features(personality_traits)
    
    # Determine expression from traits
    expressions = {}
    if personality_traits.get('friendliness', 0.5) > 0.7:
        expressions['smile'] = 0.8
    if personality_traits.get('seriousness', 0.5) > 0.7:
        expressions['neutral'] = 1.0
    
    return self.process_request({
        'capability_type': 'generate_face',
        'description': 'Generate avatar for personality',
        'parameters': {
            'features': features,
            'expressions': expressions,
            'output_format': 'both'
        }
    })

def _personality_to_features(self, traits: Dict[str, float]) -> Dict[str, float]:
    """Map personality traits to facial features"""
    return {
        'head_width': 0.9 + traits.get('confidence', 0.5) * 0.2,
        'eye_size': 0.9 + traits.get('curiosity', 0.5) * 0.3,
        'eye_spacing': 1.0,
        'nose_length': 0.9 + traits.get('assertiveness', 0.5) * 0.2,
        'mouth_width': 0.9 + traits.get('friendliness', 0.5) * 0.2,
        'jaw_width': 0.9 + traits.get('determination', 0.5) * 0.2,
    }

def install_capability_extension(self, package_name: str) -> Dict[str, Any]:
    """Install a capability extension package"""
    return self.process_request({
        'capability_type': 'package_install',
        'description': f'Install extension: {package_name}',
        'parameters': {
            'name': package_name,
            'version': 'latest'
        }
    })

def get_system_status(self) -> Dict[str, Any]:
    """Get comprehensive system status"""
    status = {
        'session_id': self.session_id,
        'capabilities': self.capabilities.get_status(),
        'actions_performed': len(self.action_history),
        'recent_actions': self.action_history[-5:] if self.action_history else [],
        'integrated_systems': {
            'consciousness': self.consciousness is not None,
            'memory': self.memory is not None,
            'proxy': self.proxy is not None,
            'moral_core': self.moral_core is not None
        }
    }
    
    return status
```

# Example integration with existing main.py

def integrate_with_main():
“””
Example of how to integrate with existing main.py
Add this to your main.py file:
“””
example_code = ‘’’

# In your existing main.py, add:

from janus_enhanced_integration import EnhancedJanusCore

class JanusMain:
def **init**(self):
# … your existing initialization …

```
    # Add enhanced capabilities
    self.enhanced = EnhancedJanusCore(use_existing_systems=True)
    
    print("Janus Enhanced Capabilities loaded!")
    print(self.enhanced.get_system_status())

def process_user_command(self, command: str):
    # ... your existing command processing ...
    
    # Add capability commands
    if command.startswith('/create_project'):
        goal = command.replace('/create_project', '').strip()
        result = self.enhanced.autonomous_code_generation(goal)
        return result
    
    elif command.startswith('/generate_avatar'):
        # Parse personality traits from command
        traits = {'friendliness': 0.8, 'curiosity': 0.7}
        result = self.enhanced.generate_avatar_face(traits)
        return result
    
    # ... rest of your command handling ...
```

‘’’

```
return example_code
```

def main():
“”“Standalone demo”””
print(”=== Janus Enhanced Integration Demo ===\n”)

```
# Initialize
core = EnhancedJanusCore(use_existing_systems=False)

# Show status
status = core.get_system_status()
print("System Status:")
print(json.dumps(status, indent=2))

# Demo autonomous code generation
print("\n" + "="*50)
print("Demo: Autonomous Code Generation")
print("="*50 + "\n")

result = core.autonomous_code_generation("Create a prime number checker")
print(f"Success: {result['success']}")
print(f"Steps completed: {len(result['results'])}")

# Demo avatar generation
print("\n" + "="*50)
print("Demo: Avatar Generation")
print("="*50 + "\n")

personality = {
    'friendliness': 0.9,
    'curiosity': 0.8,
    'confidence': 0.7
}

result = core.generate_avatar_face(personality)
print(f"Avatar generation: {result['success']}")
if result['success']:
    print(f"Output: {result['result_data']}")

print("\n=== Integration Complete ===")
```

if **name** == ‘**main**’:
main()