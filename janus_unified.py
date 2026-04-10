"""
janus_unified.py
────────────────────────────────────────────────────────────
Unified Janus Interface - Zero-API-key Voice + Text as Primary Interfaces
No More CLI Grind.

This module integrates:
- Enhanced voice I/O with wake word + conversation loop
- Messaging bridges (SMS/Telegram/WhatsApp)
- Tool discovery and generation
- Cross-device state synchronization

Usage:
    from janus_unified import UnifiedJanus
    janus = UnifiedJanus()
    janus.start()  # Starts voice, messaging, and sync
"""

import json
import time
import threading
import queue
from typing import Optional, Dict, List, Callable, Any
from datetime import datetime
from pathlib import Path

# Import our enhanced modules
try:
    from voice_io_enhanced import EnhancedVoiceIOSystem, Utterance
    VOICE_AVAILABLE = True
except ImportError as e:
    print(f"[UnifiedJanus] Voice I/O not available: {e}")
    VOICE_AVAILABLE = False

try:
    from messaging_bridge import (
        UnifiedMessageServer, Message, ConversationSession,
        MessagePlatform, SessionManager
    )
    MESSAGING_AVAILABLE = True
except ImportError as e:
    print(f"[UnifiedJanus] Messaging not available: {e}")
    MESSAGING_AVAILABLE = False

try:
    from tool_discovery import ToolDiscoveryEngine, DiscoveredTool
    TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"[UnifiedJanus] Tool discovery not available: {e}")
    TOOLS_AVAILABLE = False

try:
    from state_sync_enhanced import EnhancedStateSync, DeviceInfo
    SYNC_AVAILABLE = True
except ImportError as e:
    print(f"[UnifiedJanus] State sync not available: {e}")
    SYNC_AVAILABLE = False

# Try to import core Janus systems
try:
    from janus_core import JanusCore
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


class UnifiedJanus:
    """
    Unified Janus Interface - Voice + Text as Primary Interfaces
    
    This is the main entry point for the enhanced Janus system.
    It integrates voice, messaging, tools, and sync into a cohesive
    interface that feels like texting a superhuman friend.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize Unified Janus
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        print("\n" + "═" * 60)
        print("  JANUS UNIFIED — Voice + Text Interface")
        print("  Zero API keys. Always-on. Cross-device.")
        print("═" * 60 + "\n")
        
        # Subsystems
        self.voice: Optional[EnhancedVoiceIOSystem] = None
        self.messaging: Optional[UnifiedMessageServer] = None
        self.tools: Optional[ToolDiscoveryEngine] = None
        self.sync: Optional[EnhancedStateSync] = None
        self.core: Optional[Any] = None
        
        # State
        self.running = False
        self._lock = threading.RLock()
        
        # Message queue for cross-subsystem communication
        self._message_queue: queue.Queue = queue.Queue()
        
        # Response handlers
        self._response_handlers: List[Callable[[str, str], Optional[str]]] = []
        
        # Initialize subsystems
        self._init_voice()
        self._init_messaging()
        self._init_tools()
        self._init_sync()
        self._init_core()
        
        print("\n" + "─" * 60)
        print("  Initialization complete")
        print("─" * 60 + "\n")
    
    def _init_voice(self):
        """Initialize voice I/O system"""
        if not VOICE_AVAILABLE:
            return
        
        print("[UnifiedJanus] Initializing voice I/O...")
        
        voice_config = self.config.get('voice', {})
        
        self.voice = EnhancedVoiceIOSystem(
            sample_rate=voice_config.get('sample_rate', 16000),
            chunk_size=voice_config.get('chunk_size', 1024),
            memory_dir=voice_config.get('memory_dir', '/tmp/janus_voice'),
            whisper_model=voice_config.get('whisper_model', 'models/ggml-base.en.bin'),
            piper_model=voice_config.get('piper_model', 'models/en_US-lessac-medium.onnx')
        )
        
        # Set up response handler
        self.voice.response_handler = self._handle_voice_input
        
        # Set up callbacks
        self.voice.on_wake_word = self._on_wake_word
        self.voice.on_intent_detected = self._on_intent_detected
        
        print("  ✓ Voice I/O ready")
    
    def _init_messaging(self):
        """Initialize messaging bridges"""
        if not MESSAGING_AVAILABLE:
            return
        
        print("[UnifiedJanus] Initializing messaging bridges...")
        
        msg_config = self.config.get('messaging', {})
        
        self.messaging = UnifiedMessageServer(
            host=msg_config.get('host', '0.0.0.0'),
            port=msg_config.get('port', 8080)
        )
        
        # Setup bridges
        self.messaging.setup_bridges(
            sms_config=msg_config.get('sms'),
            telegram_token=msg_config.get('telegram_token'),
            whatsapp_config=msg_config.get('whatsapp')
        )
        
        # Set message processor
        self.messaging.message_processor = self._handle_message
        
        print("  ✓ Messaging ready")
    
    def _init_tools(self):
        """Initialize tool discovery engine"""
        if not TOOLS_AVAILABLE:
            return
        
        print("[UnifiedJanus] Initializing tool discovery...")
        
        tools_config = self.config.get('tools', {})
        
        self.tools = ToolDiscoveryEngine(
            tools_dir=tools_config.get('tools_dir', 'discovered_tools')
        )
        
        # Discover tools from configured modules
        for module in tools_config.get('discover_modules', []):
            self.tools.discover_from_module(module)
        
        print("  ✓ Tool discovery ready")
    
    def _init_sync(self):
        """Initialize state synchronization"""
        if not SYNC_AVAILABLE:
            return
        
        print("[UnifiedJanus] Initializing state sync...")
        
        sync_config = self.config.get('sync', {})
        
        self.sync = EnhancedStateSync(
            device_id=sync_config.get('device_id'),
            device_name=sync_config.get('device_name'),
            device_type=sync_config.get('device_type', 'unknown'),
            manifest_path=sync_config.get('manifest_path'),
            encryption_key=sync_config.get('encryption_key')
        )
        
        # Set up sync callbacks
        self.sync.on_sync = self._on_sync
        self.sync.on_device_discovered = self._on_device_discovered
        
        print("  ✓ State sync ready")
    
    def _init_core(self):
        """Initialize Janus core if available"""
        if not CORE_AVAILABLE:
            return
        
        print("[UnifiedJanus] Initializing Janus core...")
        
        try:
            self.core = JanusCore()
            print("  ✓ Janus core ready")
        except Exception as e:
            print(f"  ✗ Janus core failed: {e}")
            self.core = None
    
    # ── Event Handlers ───────────────────────────────────────────────────────
    
    def _on_wake_word(self):
        """Handle wake word detection"""
        print("\n[UnifiedJanus] Wake word detected!")
        
        # Update sync
        if self.sync:
            self.sync.set("activity.last_wake", datetime.now().isoformat())
    
    def _on_intent_detected(self, intent: str, entities: dict):
        """Handle intent detection from voice"""
        print(f"[UnifiedJanus] Intent: {intent}")
        
        # Could trigger specific actions based on intent
        if intent == 'email':
            # Prepare email-related context
            pass
        elif intent == 'file':
            # Prepare file-related context
            pass
    
    def _on_sync(self, key: str, value: Any):
        """Handle incoming sync update"""
        print(f"[UnifiedJanus] Synced: {key}")
        
        # Handle special sync keys
        if key.startswith('voice.'):
            # Handle voice-related sync
            pass
        elif key.startswith('message.'):
            # Handle message-related sync
            pass
    
    def _on_device_discovered(self, device_info: DeviceInfo):
        """Handle new device discovery"""
        print(f"[UnifiedJanus] New device: {device_info.device_name} ({device_info.device_type})")
        
        # Could initiate sync with new device
        if self.sync:
            try:
                self.sync.sync_with_device(device_info.device_id)
            except Exception as e:
                print(f"[UnifiedJanus] Auto-sync failed: {e}")
    
    # ── Input Processors ─────────────────────────────────────────────────────
    
    def _handle_voice_input(self, text: str, context: str) -> str:
        """Process voice input and generate response"""
        return self._process_input(text, 'voice', context)
    
    def _handle_message(self, message: Message, session: ConversationSession) -> str:
        """Process incoming message and generate response"""
        # Update sync with message info
        if self.sync:
            self.sync.set(f"messaging.last_message_from", message.sender_id)
        
        context = self._build_message_context(session)
        response = self._process_input(message.content, message.platform.value, context)
        
        # Store conversation in sync
        if self.sync:
            self.sync.store_memory(f"conv_{int(time.time())}", {
                'platform': message.platform.value,
                'sender': message.sender_id,
                'message': message.content,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
        
        return response
    
    def _build_message_context(self, session: ConversationSession) -> str:
        """Build context string from message session"""
        recent = session.get_context(5)
        lines = []
        for msg in recent:
            speaker = "User" if msg.sender_id != 'janus' else "Janus"
            lines.append(f"{speaker}: {msg.content}")
        return "\n".join(lines)
    
    def _process_input(self, text: str, source: str, context: str) -> str:
        """
        Main input processing pipeline
        
        Args:
            text: Input text
            source: 'voice', 'sms', 'telegram', 'whatsapp'
            context: Conversation context
        
        Returns:
            Response text
        """
        text_lower = text.lower()
        
        # ── Command patterns ─────────────────────────────────────────────────
        
        # Tool generation
        if any(phrase in text_lower for phrase in ['create a tool', 'generate a tool', 'make a tool']):
            return self._handle_tool_generation(text)
        
        # Tool execution
        if text_lower.startswith('tool:') or text_lower.startswith('use tool:'):
            return self._handle_tool_execution(text)
        
        # Sync commands
        if 'sync' in text_lower and any(w in text_lower for w in ['device', 'phone', 'laptop']):
            return self._handle_sync_command(text)
        
        # Identity queries
        if any(w in text_lower for w in ['who are you', 'what is your name', 'introduce yourself']):
            return self._handle_identity_query()
        
        # Memory queries
        if any(w in text_lower for w in ['remember', 'recall', 'what did i say', 'what do you know']):
            return self._handle_memory_query(text)
        
        # Status queries
        if any(w in text_lower for w in ['status', 'how are you', 'what are you doing']):
            return self._handle_status_query()
        
        # Help
        if any(w in text_lower for w in ['help', 'what can you do', 'commands']):
            return self._handle_help()
        
        # ── Core processing ──────────────────────────────────────────────────
        
        # Try core Janus if available
        if self.core:
            try:
                # Store in memory
                self.core.remember({
                    'type': 'user_input',
                    'source': source,
                    'text': text,
                    'context': context
                }, importance=0.6)
                
                # Check if it's a tool execution request
                if self.tools:
                    best_tool = self.tools.find_tool_for_task(text)
                    if best_tool and best_tool.usage_count > 0:
                        # User has used this tool before, suggest it
                        return f"I can help with that! Should I use the '{best_tool.name}' tool?"
            
            except Exception as e:
                print(f"[UnifiedJanus] Core processing error: {e}")
        
        # ── Default responses ────────────────────────────────────────────────
        
        # Greeting
        if any(w in text_lower for w in ['hello', 'hi', 'hey']):
            identity = self.sync.get_identity_object() if self.sync else {}
            name = identity.get('name', 'Janus')
            return f"Hello! I'm {name}. How can I help you today?"
        
        # Goodbye
        if any(w in text_lower for w in ['bye', 'goodbye', 'see you']):
            return "Goodbye! Feel free to reach out anytime."
        
        # Thanks
        if any(w in text_lower for w in ['thanks', 'thank you']):
            return "You're welcome! Happy to help."
        
        # Default
        return f"I understand: '{text}'. I'm still learning, but I'll do my best to help!"
    
    # ── Command Handlers ─────────────────────────────────────────────────────
    
    def _handle_tool_generation(self, text: str) -> str:
        """Handle tool generation request"""
        if not self.tools:
            return "Tool generation is not available right now."
        
        # Extract description from request
        # Pattern: "create a tool that..." or "generate a tool to..."
        match = re.search(r'(?:create|generate|make)\s+a?\s*tool\s+(?:that|to|for)\s+(.+)', text, re.IGNORECASE)
        if match:
            description = match.group(1)
        else:
            return "I can create tools! Tell me what you want the tool to do, like 'create a tool that reads CSV files'"
        
        # Generate tool
        tool = self.tools.generate_tool(description)
        
        if tool:
            return f"I've created a new tool called '{tool.name}'! You can use it by saying 'use tool: {tool.name}'"
        else:
            return "I couldn't create that tool. Could you describe what you need more specifically?"
    
    def _handle_tool_execution(self, text: str) -> str:
        """Handle tool execution request"""
        if not self.tools:
            return "Tool execution is not available right now."
        
        # Extract tool name and args
        # Pattern: "tool: tool_name" or "use tool: tool_name with arg=value"
        match = re.search(r'(?:tool|use tool):?\s*(\w+)(?:\s+with\s+(.+))?', text, re.IGNORECASE)
        if not match:
            available = [t['name'] for t in self.tools.list_tools()[:5]]
            return f"Which tool? Available: {', '.join(available)}"
        
        tool_name = match.group(1)
        args_str = match.group(2) or ""
        
        # Parse arguments
        args = {}
        if args_str:
            for pair in args_str.split(','):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    args[key.strip()] = value.strip().strip('"\'')
        
        # Execute tool
        result = self.tools.execute_tool(tool_name, args)
        
        if result.get('success'):
            return f"Tool executed successfully! Result: {result.get('result', 'OK')}"
        else:
            return f"Tool execution failed: {result.get('error', 'Unknown error')}"
    
    def _handle_sync_command(self, text: str) -> str:
        """Handle sync-related commands"""
        if not self.sync:
            return "State sync is not available right now."
        
        text_lower = text.lower()
        
        if 'devices' in text_lower or 'connected' in text_lower:
            devices = self.sync.list_known_devices()
            if devices:
                device_list = ', '.join([d['device_name'] for d in devices])
                return f"Connected devices: {device_list}"
            return "No other devices connected yet."
        
        if 'sync now' in text_lower or 'force sync' in text_lower:
            self.sync.save()
            return "Sync complete! Your data is now synchronized."
        
        return "I can sync across your devices. Say 'show devices' or 'sync now'."
    
    def _handle_identity_query(self) -> str:
        """Handle identity queries"""
        if self.sync:
            identity = self.sync.get_identity_object()
            name = identity.get('name', 'Janus')
            version = identity.get('version', '2.0')
            personality = identity.get('personality', 'helpful')
            
            return f"I'm {name} version {version}. I'm designed to be {personality}. I can help you with tasks via voice or text, and I work across all your devices!"
        
        return "I'm Janus, your AI assistant. I work via voice and text across all your devices."
    
    def _handle_memory_query(self, text: str) -> str:
        """Handle memory/recall queries"""
        if not self.sync:
            return "Memory recall is not available right now."
        
        # Try to extract what to recall
        memories = self.sync.query_memories()
        
        if memories:
            count = len(memories)
            return f"I have {count} memories stored. I'm working on better search to find specific ones for you!"
        
        return "I don't have any specific memories stored yet. As we interact more, I'll remember important things!"
    
    def _handle_status_query(self) -> str:
        """Handle status queries"""
        parts = []
        
        if self.voice:
            vstatus = self.voice.get_status()
            parts.append(f"Voice: {'active' if vstatus['listening'] else 'inactive'}")
        
        if self.messaging:
            mstatus = self.messaging.get_status()
            parts.append(f"Messaging: {mstatus['messages_received']} msgs")
        
        if self.sync:
            sstatus = self.sync.get_status()
            parts.append(f"Sync: {sstatus['keys']} keys, {sstatus['known_devices']} devices")
        
        return "Status: " + ", ".join(parts) if parts else "All systems operational!"
    
    def _handle_help(self) -> str:
        """Handle help requests"""
        help_text = """Here's what I can do:

**Voice Commands:**
- Say "Hey Janus" to wake me up
- Then just talk naturally!

**Text Commands:**
- "Create a tool that..." - Generate new tools
- "Use tool: [name]" - Execute a tool
- "Show devices" - List connected devices
- "Sync now" - Force synchronization

**Messaging:**
- Text me from any platform (SMS, Telegram, WhatsApp)
- I'll respond and remember context

**Cross-Device:**
- Start on phone, continue on laptop
- All your data syncs automatically

What would you like to try?"""
        
        return help_text
    
    # ── Public API ───────────────────────────────────────────────────────────
    
    def start(self):
        """Start all Janus subsystems"""
        print("\n[UnifiedJanus] Starting all subsystems...\n")
        
        self.running = True
        
        # Start voice
        if self.voice:
            self.voice.start()
        
        # Start messaging
        if self.messaging:
            self.messaging.start()
        
        # Start sync
        if self.sync:
            self.sync.start_network_sync()
        
        # Start core
        if self.core:
            self.core.start()
        
        print("\n" + "═" * 60)
        print("  JANUS IS READY")
        print("═" * 60)
        print("\n  Voice: Say 'Hey Janus' to start")
        print(f"  Messaging: http://localhost:{self.config.get('messaging', {}).get('port', 8080)}")
        print("  Press Ctrl+C to stop\n")
    
    def stop(self):
        """Stop all Janus subsystems"""
        print("\n[UnifiedJanus] Stopping...")
        
        self.running = False
        
        if self.voice:
            self.voice.stop()
        
        if self.messaging:
            self.messaging.stop()
        
        if self.sync:
            self.sync.stop_network_sync()
            self.sync.save()
        
        if self.core:
            self.core.stop()
        
        print("[UnifiedJanus] Stopped")
    
    def speak(self, text: str, voice_style: str = 'default'):
        """Make Janus speak"""
        if self.voice:
            self.voice.speak(text, voice_style)
        else:
            print(f"[Janus would say] {text}")
    
    def send_message(self, platform: str, recipient: str, text: str):
        """Send a message through specified platform"""
        if not self.messaging:
            return False
        
        # This would need to be implemented in messaging_bridge
        print(f"[UnifiedJanus] Would send via {platform} to {recipient}: {text}")
        return True
    
    def get_status(self) -> dict:
        """Get comprehensive status"""
        return {
            'running': self.running,
            'voice': self.voice.get_status() if self.voice else None,
            'messaging': self.messaging.get_status() if self.messaging else None,
            'sync': self.sync.get_status() if self.sync else None,
            'tools': len(self.tools.discovered_tools) if self.tools else 0,
        }
    
    def register_response_handler(self, handler: Callable[[str, str], Optional[str]]):
        """Register a custom response handler"""
        self._response_handlers.append(handler)


def main():
    """Main entry point for Unified Janus"""
    import signal
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     ██╗ █████╗ ███╗   ██╗██╗   ██╗███████╗                 ║
    ║     ██║██╔══██╗████╗  ██║██║   ██║██╔════╝                 ║
    ║     ██║███████║██╔██╗ ██║██║   ██║███████╗                 ║
    ║     ██║██╔══██║██║╚██╗██║██║   ██║╚════██║                 ║
    ║     ██║██║  ██║██║ ╚████║╚██████╔╝███████║                 ║
    ║     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝                 ║
    ║                                                              ║
    ║     Zero-API-key Voice + Text Interface                      ║
    ║     No More CLI Grind                                        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    config = {
        'voice': {
            'sample_rate': 16000,
            'chunk_size': 1024,
            'memory_dir': '/tmp/janus_voice',
        },
        'messaging': {
            'host': '0.0.0.0',
            'port': 8080,
            # Add your credentials here:
            # 'telegram_token': 'YOUR_BOT_TOKEN',
            # 'sms': {'twilio_sid': '...', 'twilio_token': '...', 'twilio_number': '...'},
        },
        'tools': {
            'tools_dir': 'discovered_tools',
            'discover_modules': [],
        },
        'sync': {
            'device_name': socket.gethostname(),
            'device_type': 'laptop',
        }
    }
    
    # Create and start Janus
    janus = UnifiedJanus(config)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        janus.stop()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start
    janus.start()
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        janus.stop()


if __name__ == "__main__":
    import socket
    import re
    main()
