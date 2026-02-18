Janus Unified — Zero-API-key Voice + Text Interface

Goal: 100% of daily computer tasks (email, files, research, coding) become zero-API-key Voice + Text as Primary Interfaces (No More CLI Grind)

Janus Unified expands the original Janus project with always-on wake-word detection, messaging bridges, tool discovery, and cross-device synchronization — all without requiring any API keys.
⸻
✨ Features

🎙️ Always-On Voice Interface
- Wake Word Detection: Say "Hey Janus", "OK Janus", or just "Janus" to activate
- Continuous Conversation: Natural back-and-forth dialogue
- Local STT: Whisper.cpp integration (no cloud, no API keys)
- Local TTS: Piper integration (natural-sounding voice)
- Voice Activity Detection: WebRTC VAD for accurate speech detection

💬 Messaging Bridges
Text Janus from anywhere:
- SMS: Via Twilio or local GSM modem
- Telegram: Full bot integration with polling
- WhatsApp: Business API support
- Missed Call Handling: Voicemail transcription and processing

Janus runs a local server → you text "Hey Janus, book my flight" or "Reply to mom's email". No app-switching; feels like texting a superhuman friend.

🔧 Tool Discovery & Generation
- Auto-Discover Tools: Find tools in Python modules
- Generate Tools: Create new tools from natural language descriptions
- On-the-Fly Creation: "Create a tool that reads CSV files" → Janus generates the code
- Sandboxed Execution: Safe tool execution with validation

🔄 Cross-Device State Sync
- CRDT-Based Sync: Conflict-free replicated data types
- Vector Clocks: Causality tracking across devices
- Automatic Discovery: Find other Janus instances on local network
- Encrypted Sync: Optional encryption for sensitive data
- Persistent Identity: Your Janus remembers you across all devices

You text from phone → Janus picks up on laptop/server seamlessly.
⸻
📁 New Files

Janus/
├── voice_io_enhanced.py      # Always-on wake word + conversation loop
├── messaging_bridge.py        # SMS/Telegram/WhatsApp bridges
├── tool_discovery.py          # Dynamic tool discovery & generation
├── state_sync_enhanced.py     # Cross-device state synchronization
├── janus_unified.py           # Main unified interface
├── requirements_enhanced.txt  # Python dependencies
└── UNIFIED_README.md          # This file

⸻
🚀 Quick Start

1. Install Dependencies

# Python dependencies
pip install -r requirements_enhanced.txt

# Install Whisper.cpp (local STT)
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make
# Download model
bash models/download-ggml-model.sh base.en

# Install Piper (local TTS)
# Download from: https://github.com/rhasspy/piper/releases
# Or build from source
git clone https://github.com/rhasspy/piper.git
cd piper
mkdir build && cd build
cmake .. && make


2. Configure Janus

Create a config.json:

{
  "voice": {
    "sample_rate": 16000,
    "whisper_model": "models/ggml-base.en.bin",
    "piper_model": "models/en_US-lessac-medium.onnx"
  },
  "messaging": {
    "port": 8080,
    "telegram_token": "YOUR_BOT_TOKEN",  // Optional
    "sms": {  // Optional - use Twilio
      "twilio_sid": "YOUR_SID",
      "twilio_token": "YOUR_TOKEN",
      "twilio_number": "YOUR_NUMBER"
    }
  },
  "sync": {
    "device_name": "My-Laptop",
    "device_type": "laptop"
  }
}


3. Run Janus Unified

python janus_unified.py


You'll see:
════════════════════════════════════════════════════════════
  JANUS UNIFIED — Voice + Text Interface
  Zero API keys. Always-on. Cross-device.
════════════════════════════════════════════════════════════

[UnifiedJanus] Initializing voice I/O...
  ✓ Voice I/O ready
[UnifiedJanus] Initializing messaging bridges...
  ✓ Messaging ready
...

════════════════════════════════════════════════════════════
  JANUS IS READY
════════════════════════════════════════════════════════════

  Voice: Say 'Hey Janus' to start
  Messaging: http://localhost:8080
  Press Ctrl+C to stop

⸻
🎙️ Voice Commands

Wake Words
- "Hey Janus"
- "OK Janus"
- "Janus"
- "Hi Janus"

Example Interactions

You: "Hey Janus" 
Janus: "Yes? I'm listening."

You: "What's the weather like?" 
Janus: "I don't have access to weather data yet, but I can help you check a weather website."

You: "Create a tool that reads CSV files" 
Janus: "I've created a new tool called 'read_csv_files'! You can use it by saying 'use tool: read_csv_files'"

You: "Goodbye" 
Janus: "Goodbye! Feel free to reach out anytime."
⸻
💬 Messaging

Telegram Setup

1. Message @BotFather on Telegram
2. Create a new bot with /newbot
3. Copy the token
4. Add to your config:

{
  "messaging": {
    "telegram_token": "123456789:ABCdefGHIjklMNOpqrSTUvwxyz"
  }
}


5. Text your bot: "Hello Janus!"

SMS Setup (Twilio)

1. Sign up at Twilio
2. Get a phone number
3. Add credentials to config:

{
  "messaging": {
    "sms": {
      "twilio_sid": "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      "twilio_token": "your_auth_token",
      "twilio_number": "+1234567890"
    }
  }
}


4. Configure webhook URL in Twilio console: http://your-server:8080/sms

Webhook Endpoints

- GET / — Status
- POST /sms — SMS webhook
- POST /whatsapp — WhatsApp webhook
- POST /call — Voicemail upload
⸻
🔧 Tool Generation

Natural Language to Tool

You: "Create a tool that fetches weather from OpenWeatherMap"

Janus generates:

def fetch_weather(city: str, api_key: str) -> dict:
    """
    Fetch weather from OpenWeatherMap
    """
    import requests
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        response = requests.get(url, params={
            'q': city,
            'appid': api_key,
            'units': 'metric'
        })
        
        return {
            "success": response.status_code  200,
            "data": response.json() if response.status_code  200 else None
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


Using Generated Tools

You: "Use tool: fetch_weather with city=London, api_key=YOUR_KEY"
⸻
🔄 Cross-Device Sync

How It Works

1. Automatic Discovery: Janus instances find each other on the local network via UDP broadcast
2. Vector Clocks: Track causality to handle concurrent updates
3. CRDT Merge: Conflict-free merging of state changes
4. Encrypted Transfer: Optional encryption for sensitive data

Synced Data

- Identity: Name, personality, preferences
- Memories: Important facts, user preferences
- Conversation History: Context across devices
- Tool Registry: Generated and discovered tools
- Settings: All configuration

Example

1. On Phone: "Hey Janus, remember that I prefer dark mode"
2. On Laptop: Janus already knows you prefer dark mode
⸻
📊 Architecture

┌─────────────────────────────────────────────────────────────┐
│                    UNIFIED JANUS                            │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Voice     │  │  Messaging   │  │    Tools     │      │
│  │   (Wake +    │  │  (SMS/TG/WA) │  │ (Discovery + │      │
│  │Conversation) │  │              │  │ Generation)  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘              │
│                           │                                │
│              ┌────────────┴────────────┐                  │
│              │    Unified Processor    │                  │
│              │  (Intent → Response)    │                  │
│              └────────────┬────────────┘                  │
│                           │                                │
│              ┌────────────┴────────────┐                  │
│              │   State Sync (CRDT)     │                  │
│              │  (Cross-device merge)   │                  │
│              └─────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘

⸻
🔌 API Reference

UnifiedJanus Class

from janus_unified import UnifiedJanus

# Create instance
janus = UnifiedJanus(config)

# Start all subsystems
janus.start()

# Make Janus speak
janus.speak("Hello!", voice_style='friendly')

# Send message
janus.send_message('telegram', 'user_id', 'Hello!')

# Get status
status = janus.get_status()

# Stop
janus.stop()


Voice I/O

from voice_io_enhanced import EnhancedVoiceIOSystem

voice = EnhancedVoiceIOSystem()
voice.start()

# Set response handler
def handle_input(text, context):
    return f"You said: {text}"

voice.response_handler = handle_input

# Or manually trigger
voice.speak("Hello!")
voice.start_conversation()


Messaging

from messaging_bridge import UnifiedMessageServer

server = UnifiedMessageServer(port=8080)
server.setup_bridges(telegram_token='...')
server.message_processor = my_handler
server.start()


Tool Discovery

from tool_discovery import ToolDiscoveryEngine

discovery = ToolDiscoveryEngine()

# Generate tool
tool = discovery.generate_tool("Read CSV files and return data")

# Execute
result = discovery.execute_tool(tool.name, {'path': 'data.csv'})

# Find for task
best_tool = discovery.find_tool_for_task("I need to read a file")


State Sync

from state_sync_enhanced import EnhancedStateSync

sync = EnhancedStateSync(device_name='My-Laptop')

# Set values (auto-propagates)
sync.set('preference.theme', 'dark')

# Get values
theme = sync.get('preference.theme', 'light')

# Store memory
sync.store_memory('mem_001', {'content': 'User likes cats'})

# Start network sync
sync.start_network_sync()

# Sync with specific device
sync.sync_with_device('other-device-id')

⸻
🛠️ Advanced Configuration

Custom Wake Words

Edit voice_io_enhanced.py:

self.wake_words = [
    'janus', 'hey janus', 'ok janus',
    'computer', 'hey computer',  # Add your own
]


Custom Voice Styles

voice.speak("Hello!", voice_style='excited')  # friendly, serious, calm, excited


Tool Templates

Add new templates in tool_discovery.py:

self.templates['database'] = '''
def {name}(query: str, connection_string: str) -> dict:
    """{description}"""
    import sqlite3
    # ... implementation
'''

⸻
🔒 Security

Encryption

Enable encryption for state sync:

sync = EnhancedStateSync(encryption_key='your-secret-key')


Sandboxed Tools

High-risk tools run in isolated subprocesses with timeout protection.

Rate Limiting

Tool execution is rate-limited to prevent abuse.
⸻
🐛 Troubleshooting

Voice Not Working

# Check audio devices
python -c "import pyaudio; p = pyaudio.PyAudio(); print([p.get_device_info_by_index(i) for i in range(p.get_device_count())])"

# Test recording
arecord -l


Whisper Not Found

# Build whisper.cpp
cd whisper.cpp
make
./main -h  # Test


Piper Not Found

# Download pre-built binary
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz
tar -xzf piper_amd64.tar.gz
sudo mv piper /usr/local/bin/


Sync Not Working

- Check firewall settings (ports 37020-37021)
- Ensure devices are on same network
- Check sync_manifest.json for errors
⸻
🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request
⸻
📜 License

Same as original Janus project.
⸻
🙏 Acknowledgments

- Whisper.cpp by Georgi Gerganov
- Piper by Rhasspy
- Original Janus project by ThoseIdiots
⸻
Janus Unified: Your AI companion that works how you work — by voice, by text, across all your devices.

            Generated by Kimi.ai