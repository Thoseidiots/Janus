"""
Error Recovery System for Janus
Handles failures gracefully and provides retry logic with fallbacks
"""

from typing import Callable, Any, Optional, Dict, List
import time
import traceback
from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = 1      # Recoverable, retry immediately
    MEDIUM = 2   # Recoverable, retry with delay
    HIGH = 3     # Requires fallback strategy
    CRITICAL = 4 # Cannot recover, abort


@dataclass
class RecoveryStrategy:
    """Recovery strategy for an error"""
    max_retries: int
    retry_delay: float
    fallback_action: Optional[Callable] = None
    escalate_after: int = 3


class ErrorRecovery:
    """
    Error recovery system with retry logic and fallbacks
    """
    
    def __init__(self):
        self.error_history = []
        self.recovery_stats = {
            'total_errors': 0,
            'recovered': 0,
            'failed': 0,
            'fallbacks_used': 0
        }
        
        # Default recovery strategies by error type
        self.strategies = {
            'ElementNotFound': RecoveryStrategy(max_retries=3, retry_delay=1.0),
            'WindowNotFound': RecoveryStrategy(max_retries=2, retry_delay=2.0),
            'ActionTimeout': RecoveryStrategy(max_retries=2, retry_delay=3.0),
            'ScreenCaptureError': RecoveryStrategy(max_retries=3, retry_delay=0.5),
            'NetworkError': RecoveryStrategy(max_retries=5, retry_delay=5.0),
            'default': RecoveryStrategy(max_retries=2, retry_delay=1.0)
        }
    
    def execute_with_recovery(self, 
                             action: Callable,
                             error_type: str = 'default',
                             context: Dict = None) -> Dict[str, Any]:
        """
        Execute action with automatic error recovery
        
        Args:
            action: Function to execute
            error_type: Type of error expected (for strategy selection)
            context: Additional context for recovery
            
        Returns:
            {
                'success': bool,
                'result': Any,
                'attempts': int,
                'recovery_used': bool,
                'error': Optional[str]
            }
        """
        strategy = self.strategies.get(error_type, self.strategies['default'])
        attempts = 0
        last_error = None
        
        while attempts <= strategy.max_retries:
            attempts += 1
            
            try:
                result = action()
                
                # Success!
                if attempts > 1:
                    self.recovery_stats['recovered'] += 1
                
                return {
                    'success': True,
                    'result': result,
                    'attempts': attempts,
                    'recovery_used': attempts > 1,
                    'error': None
                }
                
            except Exception as e:
                last_error = str(e)
                self.recovery_stats['total_errors'] += 1
                
                # Log error
                self.error_history.append({
                    'error_type': error_type,
                    'error': last_error,
                    'attempt': attempts,
                    'context': context,
                    'traceback': traceback.format_exc()
                })
                
                # Check if we should retry
                if attempts <= strategy.max_retries:
                    print(f"  Attempt {attempts} failed: {last_error}")
                    print(f"  Retrying in {strategy.retry_delay}s...")
                    time.sleep(strategy.retry_delay)
                else:
                    # Max retries reached, try fallback
                    if strategy.fallback_action:
                        print(f"  Max retries reached, trying fallback...")
                        try:
                            result = strategy.fallback_action(context)
                            self.recovery_stats['fallbacks_used'] += 1
                            return {
                                'success': True,
                                'result': result,
                                'attempts': attempts,
                                'recovery_used': True,
                                'error': None
                            }
                        except Exception as fallback_error:
                            last_error = f"Fallback failed: {fallback_error}"
        
        # All recovery attempts failed
        self.recovery_stats['failed'] += 1
        return {
            'success': False,
            'result': None,
            'attempts': attempts,
            'recovery_used': True,
            'error': last_error
        }
    
    def with_timeout(self, action: Callable, timeout: float = 30.0) -> Any:
        """Execute action with timeout"""
        import threading
        
        result = {'value': None, 'error': None, 'completed': False}
        
        def wrapper():
            try:
                result['value'] = action()
                result['completed'] = True
            except Exception as e:
                result['error'] = str(e)
        
        thread = threading.Thread(target=wrapper)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if not result['completed']:
            raise TimeoutError(f"Action timed out after {timeout}s")
        
        if result['error']:
            raise Exception(result['error'])
        
        return result['value']
    
    def safe_execute(self, action: Callable, 
                    default_value: Any = None,
                    log_errors: bool = True) -> Any:
        """
        Execute action safely, returning default value on error
        """
        try:
            return action()
        except Exception as e:
            if log_errors:
                self.error_history.append({
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
            return default_value
    
    def classify_error(self, error: Exception) -> ErrorSeverity:
        """Classify error severity"""
        error_str = str(error).lower()
        
        if 'timeout' in error_str or 'not found' in error_str:
            return ErrorSeverity.MEDIUM
        
        if 'network' in error_str or 'connection' in error_str:
            return ErrorSeverity.HIGH
        
        if 'critical' in error_str or 'fatal' in error_str:
            return ErrorSeverity.CRITICAL
        
        return ErrorSeverity.LOW
    
    def get_recovery_stats(self) -> Dict:
        """Get recovery statistics"""
        total = self.recovery_stats['total_errors']
        if total == 0:
            success_rate = 1.0
        else:
            success_rate = self.recovery_stats['recovered'] / total
        
        return {
            **self.recovery_stats,
            'success_rate': success_rate
        }
    
    def get_recent_errors(self, count: int = 10) -> List[Dict]:
        """Get recent errors"""
        return self.error_history[-count:]
    
    def clear_history(self):
        """Clear error history"""
        self.error_history.clear()


# Specific recovery helpers
class ActionRecovery:
    """Recovery helpers for specific action types"""
    
    @staticmethod
    def recover_click(janus_os, x: int, y: int, retries: int = 3) -> bool:
        """Recover from failed click"""
        for attempt in range(retries):
            try:
                # Try clicking with slight position variations
                offset = attempt * 5
                janus_os.click(x + offset, y + offset)
                time.sleep(0.5)
                return True
            except:
                if attempt < retries - 1:
                    time.sleep(1)
        return False
    
    @staticmethod
    def recover_window_switch(window_manager, title: str, retries: int = 3) -> bool:
        """Recover from failed window switch"""
        for attempt in range(retries):
            try:
                # Try finding window with partial match
                windows = window_manager.get_all_windows()
                for window in windows:
                    if title.lower() in window.title.lower():
                        window_manager.activate_window(window.hwnd)
                        time.sleep(1)
                        return True
                
                if attempt < retries - 1:
                    time.sleep(2)
            except:
                if attempt < retries - 1:
                    time.sleep(2)
        return False
    
    @staticmethod
    def recover_screen_capture(janus_os, retries: int = 3) -> Optional[tuple]:
        """Recover from failed screen capture"""
        for attempt in range(retries):
            try:
                result = janus_os.capture_screen()
                if result and len(result) == 3:
                    return result
            except:
                pass
            
            if attempt < retries - 1:
                time.sleep(0.5)
        
        return None


# Decorator for automatic recovery
def with_recovery(error_type: str = 'default', max_retries: int = 3):
    """Decorator to add automatic recovery to functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            recovery = ErrorRecovery()
            
            def action():
                return func(*args, **kwargs)
            
            result = recovery.execute_with_recovery(
                action,
                error_type=error_type
            )
            
            if not result['success']:
                raise Exception(result['error'])
            
            return result['result']
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test error recovery
    recovery = ErrorRecovery()
    
    print("Error Recovery System Test")
    print("=" * 60)
    
    # Test 1: Successful action
    print("\n1. Testing successful action...")
    def success_action():
        return "Success!"
    
    result = recovery.execute_with_recovery(success_action)
    print(f"   Result: {result}")
    
    # Test 2: Recoverable error
    print("\n2. Testing recoverable error...")
    attempt_count = [0]
    
    def recoverable_action():
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise Exception("Temporary error")
        return "Recovered!"
    
    result = recovery.execute_with_recovery(recoverable_action)
    print(f"   Result: {result}")
    
    # Test 3: Unrecoverable error
    print("\n3. Testing unrecoverable error...")
    def failing_action():
        raise Exception("Permanent error")
    
    result = recovery.execute_with_recovery(failing_action)
    print(f"   Result: {result}")
    
    # Show stats
    print("\n4. Recovery statistics:")
    stats = recovery.get_recovery_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Error recovery system ready!")
