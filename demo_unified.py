"""
demo_unified.py
────────────────────────────────────────────────────────────
Demonstration of Janus Unified features without requiring
full setup (Whisper.cpp, Piper, etc.)

This demo shows:
- Tool discovery and generation
- State synchronization
- Messaging bridge structure
- Voice I/O architecture

Run: python demo_unified.py
"""

import time
import json
from pathlib import Path


def demo_tool_discovery():
    """Demo: Tool Discovery & Generation"""
    print("\n" + "═" * 60)
    print("  DEMO 1: Tool Discovery & Generation")
    print("═" * 60 + "\n")
    
    try:
        from tool_discovery import ToolDiscoveryEngine
        
        discovery = ToolDiscoveryEngine(tools_dir='/tmp/janus_demo_tools')
        
        # Generate a file tool
        print("1. Generating 'file_operations' tool...")
        file_tool = discovery.generate_tool(
            "Read and write text files to the local filesystem",
            name="file_operations",
            tool_type="file"
        )
        
        if file_tool:
            print(f"   ✓ Generated: {file_tool.name}")
            print(f"   ✓ Saved to: {file_tool.file_path}")
            print(f"\n   Code preview:")
            print("   " + "-" * 50)
            for line in file_tool.code.split('\n')[:15]:
                print(f"   {line}")
            print("   " + "-" * 50)
        
        # Generate an API tool
        print("\n2. Generating 'api_request' tool...")
        api_tool = discovery.generate_tool(
            "Make HTTP requests to REST APIs",
            name="api_request",
            tool_type="api"
        )
        
        if api_tool:
            print(f"   ✓ Generated: {api_tool.name}")
        
        # Test the file tool
        print("\n3. Testing file_operations tool...")
        result = discovery.execute_tool(
            'file_operations',
            {'path': '/tmp/janus_demo_test.txt', 'content': 'Hello from Janus Unified!'}
        )
        print(f"   Write result: {result}")
        
        result = discovery.execute_tool(
            'file_operations',
            {'path': '/tmp/janus_demo_test.txt'}
        )
        print(f"   Read result: {result}")
        
        # List all tools
        print("\n4. All discovered tools:")
        for tool in discovery.list_tools():
            print(f"   - {tool['name']}: {tool['description'][:40]}...")
        
        # Find tool for task
        print("\n5. Finding best tool for 'read a file'...")
        best = discovery.find_tool_for_task("I need to read a file")
        if best:
            print(f"   ✓ Best match: {best.name}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")


def demo_state_sync():
    """Demo: State Synchronization"""
    print("\n" + "═" * 60)
    print("  DEMO 2: State Synchronization")
    print("═" * 60 + "\n")
    
    try:
        from state_sync_enhanced import EnhancedStateSync
        
        # Create two sync instances (simulating two devices)
        print("1. Creating sync instances (Device A and Device B)...")
        sync_a = EnhancedStateSync(
            device_id='device_a',
            device_name='Laptop-A',
            device_type='laptop',
            manifest_path='/tmp/janus_sync_a.json'
        )
        
        sync_b = EnhancedStateSync(
            device_id='device_b',
            device_name='Phone-B',
            device_type='phone',
            manifest_path='/tmp/janus_sync_b.json'
        )
        
        # Set identity on Device A
        print("\n2. Setting identity on Device A...")
        sync_a.set_identity('name', 'Janus')
        sync_a.set_identity('version', '2.0')
        sync_a.set_identity('personality', 'helpful and friendly')
        print("   ✓ Identity set on Device A")
        
        # Store memory on Device A
        print("\n3. Storing memory on Device A...")
        sync_a.store_memory('preference_001', {
            'content': 'User prefers dark mode',
            'tags': ['preference', 'ui'],
            'importance': 0.8
        })
        print("   ✓ Memory stored")
        
        # Save Device A state
        print("\n4. Saving Device A state...")
        sync_a.save()
        print("   ✓ State saved")
        
        # Merge to Device B (simulating sync)
        print("\n5. Merging Device A state to Device B...")
        report = sync_b.merge_from_file('/tmp/janus_sync_a.json')
        print(f"   ✓ Merge complete:")
        print(f"     - Merged: {report.get('merged', 0)} keys")
        print(f"     - Kept remote: {report.get('kept_remote', 0)}")
        
        # Verify sync on Device B
        print("\n6. Verifying sync on Device B...")
        identity = sync_b.get_identity_object()
        print(f"   ✓ Identity synced: {identity.get('name')}")
        
        memories = sync_b.query_memories()
        print(f"   ✓ Memories synced: {len(memories)} memories")
        
        # Status
        print("\n7. Device B status:")
        status = sync_b.get_status()
        print(f"   - Device: {status['device_name']}")
        print(f"   - Keys: {status['keys']}")
        print(f"   - Vector clock: {status['vector']}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")


def demo_messaging():
    """Demo: Messaging Bridge Structure"""
    print("\n" + "═" * 60)
    print("  DEMO 3: Messaging Bridge")
    print("═" * 60 + "\n")
    
    try:
        from messaging_bridge import (
            UnifiedMessageServer, Message, ConversationSession,
            MessagePlatform, SessionManager
        )
        
        print("1. Creating messaging server...")
        server = UnifiedMessageServer(host='localhost', port=8080)
        print("   ✓ Server created")
        
        print("\n2. Setting up bridges (without credentials for demo)...")
        server.setup_bridges(
            sms_config=None,
            telegram_token=None,
            whatsapp_config=None
        )
        print("   ✓ Bridges configured (demo mode)")
        
        print("\n3. Creating sample message...")
        message = Message(
            message_id='msg_001',
            platform=MessagePlatform.TELEGRAM,
            sender_id='123456789',
            sender_name='Demo User',
            content='Hello Janus!',
            timestamp='2024-01-15T10:30:00',
            metadata={'chat_id': '123456789'}
        )
        print(f"   ✓ Message created:")
        print(f"     - From: {message.sender_name}")
        print(f"     - Platform: {message.platform.value}")
        print(f"     - Content: {message.content}")
        
        print("\n4. Session management...")
        session = server.sessions.get_or_create_session(
            message.sender_id,
            message.platform
        )
        session.add_message(message)
        print(f"   ✓ Session created: {session.session_id}")
        print(f"   ✓ Messages in session: {len(session.messages)}")
        
        print("\n5. Server status:")
        status = server.get_status()
        print(f"   - Running: {status['running']}")
        print(f"   - Port: {status['port']}")
        print(f"   - Bridges: SMS={status['bridges']['sms']}, "
              f"Telegram={status['bridges']['telegram']}, "
              f"WhatsApp={status['bridges']['whatsapp']}")
        
        print("\n   To test the server:")
        print("   curl http://localhost:8080/")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")


def demo_voice_io():
    """Demo: Voice I/O Architecture"""
    print("\n" + "═" * 60)
    print("  DEMO 4: Voice I/O Architecture")
    print("═" * 60 + "\n")
    
    try:
        from voice_io_enhanced import (
            EnhancedVoiceIOSystem, WhisperSTT, PiperTTS,
            WebRTCVAD, ConversationContext
        )
        
        print("1. Voice I/O Components:")
        print("   - WhisperSTT: Local speech-to-text")
        print("   - PiperTTS: Local text-to-speech")
        print("   - WebRTCVAD: Voice activity detection")
        print("   - ConversationContext: Multi-turn dialogue")
        
        print("\n2. Creating Voice I/O system (without audio hardware)...")
        # This won't actually start audio, just shows structure
        voice = EnhancedVoiceIOSystem(
            sample_rate=16000,
            memory_dir='/tmp/janus_voice_demo'
        )
        print("   ✓ Voice system created")
        
        print("\n3. Wake words configured:")
        for word in voice.stt.wake_words:
            print(f"   - '{word}'")
        
        print("\n4. Voice styles available:")
        for style in voice.tts.voices.keys():
            print(f"   - '{style}'")
        
        print("\n5. Conversation features:")
        print("   - Wake word detection")
        print("   - Continuous conversation (30s timeout)")
        print("   - Intent detection")
        print("   - Context preservation")
        
        print("\n   To use voice:")
        print("   1. Connect microphone")
        print("   2. Install Whisper.cpp and Piper")
        print("   3. Run: python janus_unified.py")
        print("   4. Say 'Hey Janus'")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")


def demo_integration():
    """Demo: How It All Fits Together"""
    print("\n" + "═" * 60)
    print("  DEMO 5: Integration Architecture")
    print("═" * 60 + "\n")
    
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    UNIFIED JANUS                            │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│                                                             │")
    print("│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │")
    print("│  │    VOICE     │  │  MESSAGING   │  │    TOOLS     │      │")
    print("│  │  Wake Word   │  │  SMS/TG/WA   │  │  Auto-Gen    │      │")
    print("│  │  Whisper.cpp │  │  Bridges     │  │  Discovery   │      │")
    print("│  │  Piper TTS   │  │  Web Server  │  │  Sandbox     │      │")
    print("│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │")
    print("│         │                 │                 │              │")
    print("│         └─────────────────┼─────────────────┘              │")
    print("│                           │                                │")
    print("│              ┌────────────┴────────────┐                  │")
    print("│              │    Unified Processor    │                  │")
    print("│              │  - Intent Recognition   │                  │")
    print("│              │  - Context Management   │                  │")
    print("│              │  - Response Generation  │                  │")
    print("│              └────────────┬────────────┘                  │")
    print("│                           │                                │")
    print("│              ┌────────────┴────────────┐                  │")
    print("│              │   State Sync (CRDT)     │                  │")
    print("│              │  - Cross-device merge   │                  │")
    print("│              │  - Identity persistence │                  │")
    print("│              │  - Memory sync          │                  │")
    print("│              └─────────────────────────┘                  │")
    print("│                                                             │")
    print("└─────────────────────────────────────────────────────────────┘")
    
    print("\nKey Features:")
    print("  ✓ Zero API keys required")
    print("  ✓ Local processing (privacy)")
    print("  ✓ Cross-device synchronization")
    print("  ✓ Extensible tool system")
    print("  ✓ Multiple input methods (voice + text)")
    
    print("\nUse Cases:")
    print("  • 'Hey Janus, summarize my emails'")
    print("  • [Text] 'Book my flight to NYC'")
    print("  • 'Create a tool that converts PDF to text'")
    print("  • Start on phone, continue on laptop")


def main():
    """Run all demos"""
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
    ║     UNIFIED DEMO — Zero-API-key Voice + Text               ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("This demo shows Janus Unified features without requiring")
    print("full setup (Whisper.cpp, Piper, etc.)")
    print("")
    
    input("Press Enter to start demo...")
    
    # Run demos
    demo_tool_discovery()
    time.sleep(1)
    
    demo_state_sync()
    time.sleep(1)
    
    demo_messaging()
    time.sleep(1)
    
    demo_voice_io()
    time.sleep(1)
    
    demo_integration()
    
    print("\n" + "═" * 60)
    print("  DEMO COMPLETE")
    print("═" * 60)
    print("\nNext steps:")
    print("  1. Run setup: bash setup_enhanced.sh")
    print("  2. Configure: edit config.json")
    print("  3. Run: python janus_unified.py")
    print("\nFor more info: see UNIFIED_README.md")
    print("")


if __name__ == "__main__":
    main()
