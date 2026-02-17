"""
messaging_bridge.py
────────────────────────────────────────────────────────────
Unified messaging bridge for SMS, Telegram, and WhatsApp.
Janus runs a local server → you text "Hey Janus, book my flight"
or "Reply to mom's email". No app-switching; feels like texting
a superhuman friend.

Features:
- SMS bridge via Twilio or local modem
- Telegram bot integration
- WhatsApp Business API bridge
- Unified message handler
- Conversation context across platforms
"""

import json
import time
import threading
import re
import queue
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Callable, Any
from datetime import datetime
from pathlib import Path
from enum import Enum
import hashlib
import uuid

# Web framework for local server
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse


class MessagePlatform(Enum):
    SMS = "sms"
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
    CALL = "call"  # For missed call transcription


@dataclass
class Message:
    """Unified message structure"""
    message_id: str
    platform: MessagePlatform
    sender_id: str  # Phone number, chat ID, etc.
    sender_name: Optional[str]
    content: str
    timestamp: str
    media_urls: List[str] = None
    reply_to: Optional[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.media_urls is None:
            self.media_urls = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MessageResponse:
    """Response to be sent back"""
    response_id: str
    original_message_id: str
    content: str
    platform: MessagePlatform
    recipient_id: str
    media_paths: List[str] = None
    
    def __post_init__(self):
        if self.media_paths is None:
            self.media_paths = []


class ConversationSession:
    """Manages a conversation session with a user across platforms"""
    
    def __init__(self, user_id: str, platform: MessagePlatform):
        self.user_id = user_id
        self.platform = platform
        self.session_id = f"{platform.value}_{user_id}_{int(time.time())}"
        self.messages: List[Message] = []
        self.created_at = datetime.now().isoformat()
        self.last_activity = time.time()
        self.context: Dict[str, Any] = {}
        self.pending_action: Optional[str] = None
    
    def add_message(self, message: Message):
        """Add a message to the session"""
        self.messages.append(message)
        self.last_activity = time.time()
    
    def get_context(self, n: int = 10) -> List[Message]:
        """Get recent message context"""
        return self.messages[-n:] if self.messages else []
    
    def is_expired(self, timeout_seconds: float = 1800) -> bool:
        """Check if session has expired (30 min default)"""
        return time.time() - self.last_activity > timeout_seconds
    
    def update_context(self, key: str, value: Any):
        """Update session context"""
        self.context[key] = value
    
    def get_context_value(self, key: str, default=None):
        """Get context value"""
        return self.context.get(key, default)


class SessionManager:
    """Manages conversation sessions across all platforms"""
    
    def __init__(self, session_timeout: float = 1800):
        self.sessions: Dict[str, ConversationSession] = {}
        self.timeout = session_timeout
        self._lock = threading.Lock()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def get_or_create_session(self, user_id: str, platform: MessagePlatform) -> ConversationSession:
        """Get existing session or create new one"""
        session_key = f"{platform.value}_{user_id}"
        
        with self._lock:
            if session_key in self.sessions:
                session = self.sessions[session_key]
                if not session.is_expired(self.timeout):
                    return session
            
            # Create new session
            session = ConversationSession(user_id, platform)
            self.sessions[session_key] = session
            return session
    
    def get_session(self, user_id: str, platform: MessagePlatform) -> Optional[ConversationSession]:
        """Get existing session if not expired"""
        session_key = f"{platform.value}_{user_id}"
        
        with self._lock:
            session = self.sessions.get(session_key)
            if session and not session.is_expired(self.timeout):
                return session
            return None
    
    def end_session(self, user_id: str, platform: MessagePlatform):
        """End a conversation session"""
        session_key = f"{platform.value}_{user_id}"
        
        with self._lock:
            if session_key in self.sessions:
                del self.sessions[session_key]
    
    def _cleanup_loop(self):
        """Periodically clean up expired sessions"""
        while True:
            time.sleep(60)  # Check every minute
            with self._lock:
                expired = [
                    key for key, session in self.sessions.items()
                    if session.is_expired(self.timeout)
                ]
                for key in expired:
                    del self.sessions[key]
                    print(f"[SessionManager] Expired session: {key}")


class SMSBridge:
    """
    SMS bridge using Twilio or local GSM modem
    """
    
    def __init__(self, 
                 twilio_sid: Optional[str] = None,
                 twilio_token: Optional[str] = None,
                 twilio_number: Optional[str] = None,
                 modem_port: Optional[str] = None):
        self.twilio_sid = twilio_sid
        self.twilio_token = twilio_token
        self.twilio_number = twilio_number
        self.modem_port = modem_port
        
        self.use_twilio = all([twilio_sid, twilio_token, twilio_number])
        self.use_modem = modem_port is not None
        
        self.message_handler: Optional[Callable[[Message], None]] = None
        
        if self.use_twilio:
            try:
                from twilio.rest import Client
                self.twilio_client = Client(twilio_sid, twilio_token)
                print("[SMSBridge] Twilio client initialized")
            except ImportError:
                print("[SMSBridge] Twilio not installed, run: pip install twilio")
                self.use_twilio = False
        
        if self.use_modem:
            try:
                import serial
                self.modem = serial.Serial(modem_port, 115200, timeout=1)
                print(f"[SMSBridge] GSM modem on {modem_port}")
            except ImportError:
                print("[SMSBridge] pyserial not installed, run: pip install pyserial")
                self.use_modem = False
            except Exception as e:
                print(f"[SMSBridge] Modem error: {e}")
                self.use_modem = False
    
    def send_message(self, to_number: str, content: str, media_urls: List[str] = None) -> bool:
        """Send SMS message"""
        try:
            if self.use_twilio:
                message = self.twilio_client.messages.create(
                    body=content,
                    from_=self.twilio_number,
                    to=to_number,
                    media_url=media_urls
                )
                print(f"[SMSBridge] Sent SMS to {to_number}: {message.sid}")
                return True
            
            elif self.use_modem:
                # AT command to send SMS
                self.modem.write(b'AT+CMGF=1\r')
                time.sleep(0.1)
                self.modem.write(f'AT+CMGS="{to_number}"\r'.encode())
                time.sleep(0.1)
                self.modem.write(content.encode() + b'\x1A')
                time.sleep(1)
                response = self.modem.read_all()
                print(f"[SMSBridge] Modem response: {response}")
                return True
            
            else:
                print(f"[SMSBridge] Would send to {to_number}: {content[:50]}...")
                return True
                
        except Exception as e:
            print(f"[SMSBridge] Send error: {e}")
            return False
    
    def parse_incoming(self, request_data: dict) -> Optional[Message]:
        """Parse incoming SMS from webhook"""
        # Twilio format
        from_number = request_data.get('From', request_data.get('from'))
        body = request_data.get('Body', request_data.get('body'))
        message_sid = request_data.get('MessageSid', request_data.get('id', str(uuid.uuid4())))
        
        if not from_number or not body:
            return None
        
        return Message(
            message_id=message_sid,
            platform=MessagePlatform.SMS,
            sender_id=from_number,
            sender_name=request_data.get('FromName'),
            content=body,
            timestamp=datetime.now().isoformat(),
            media_urls=request_data.get('MediaUrl', []),
            metadata={'twilio_data': request_data}
        )


class TelegramBridge:
    """
    Telegram Bot API bridge
    """
    
    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.message_handler: Optional[Callable[[Message], None]] = None
        self.offset = 0
        
        try:
            import requests
            self.requests = requests
            
            # Verify bot
            response = self.requests.get(f"{self.base_url}/getMe")
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    self.bot_info = data['result']
                    print(f"[TelegramBridge] Bot @{self.bot_info['username']} connected")
                else:
                    print("[TelegramBridge] Invalid bot token")
            else:
                print(f"[TelegramBridge] Connection error: {response.status_code}")
        except ImportError:
            print("[TelegramBridge] requests not installed, run: pip install requests")
        except Exception as e:
            print(f"[TelegramBridge] Error: {e}")
    
    def start_polling(self):
        """Start polling for messages"""
        self.polling_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.polling_thread.start()
    
    def _poll_loop(self):
        """Poll for new messages"""
        while True:
            try:
                response = self.requests.get(
                    f"{self.base_url}/getUpdates",
                    params={'offset': self.offset, 'limit': 100},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('ok'):
                        for update in data.get('result', []):
                            self.offset = max(self.offset, update['update_id'] + 1)
                            message = self._parse_update(update)
                            if message and self.message_handler:
                                self.message_handler(message)
                
                time.sleep(1)
                
            except Exception as e:
                print(f"[TelegramBridge] Polling error: {e}")
                time.sleep(5)
    
    def _parse_update(self, update: dict) -> Optional[Message]:
        """Parse Telegram update to Message"""
        if 'message' not in update:
            return None
        
        msg = update['message']
        chat = msg.get('chat', {})
        from_user = msg.get('from', {})
        
        # Get text or caption
        content = msg.get('text', '') or msg.get('caption', '')
        
        # Get media
        media_urls = []
        if 'photo' in msg:
            # Get largest photo
            photo = max(msg['photo'], key=lambda p: p.get('file_size', 0))
            media_urls.append(photo['file_id'])
        if 'voice' in msg:
            media_urls.append(msg['voice']['file_id'])
        
        return Message(
            message_id=str(msg['message_id']),
            platform=MessagePlatform.TELEGRAM,
            sender_id=str(from_user.get('id')),
            sender_name=from_user.get('first_name'),
            content=content,
            timestamp=datetime.fromtimestamp(msg.get('date', time.time())).isoformat(),
            media_urls=media_urls,
            reply_to=str(msg.get('reply_to_message', {}).get('message_id')) if msg.get('reply_to_message') else None,
            metadata={'chat_id': chat.get('id'), 'raw_update': update}
        )
    
    def send_message(self, chat_id: str, content: str, 
                     reply_to: Optional[str] = None,
                     parse_mode: str = 'Markdown') -> bool:
        """Send Telegram message"""
        try:
            params = {
                'chat_id': chat_id,
                'text': content,
                'parse_mode': parse_mode
            }
            if reply_to:
                params['reply_to_message_id'] = reply_to
            
            response = self.requests.post(
                f"{self.base_url}/sendMessage",
                json=params
            )
            
            if response.status_code == 200:
                print(f"[TelegramBridge] Sent message to {chat_id}")
                return True
            else:
                print(f"[TelegramBridge] Send error: {response.text}")
                return False
                
        except Exception as e:
            print(f"[TelegramBridge] Send error: {e}")
            return False
    
    def get_file_url(self, file_id: str) -> Optional[str]:
        """Get file URL from file_id"""
        try:
            response = self.requests.get(
                f"{self.base_url}/getFile",
                params={'file_id': file_id}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    file_path = data['result']['file_path']
                    return f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
            return None
        except Exception as e:
            print(f"[TelegramBridge] Get file error: {e}")
            return None


class WhatsAppBridge:
    """
    WhatsApp Business API bridge
    Supports both official API and unofficial libraries
    """
    
    def __init__(self, 
                 phone_id: Optional[str] = None,
                 access_token: Optional[str] = None,
                 use_unofficial: bool = False):
        self.phone_id = phone_id
        self.access_token = access_token
        self.use_unofficial = use_unofficial
        
        self.base_url = f"https://graph.facebook.com/v18.0/{phone_id}"
        self.message_handler: Optional[Callable[[Message], None]] = None
        
        if use_unofficial:
            try:
                # Using whatsapp-web.js or similar unofficial library
                # This would require Node.js bridge
                print("[WhatsAppBridge] Using unofficial WhatsApp Web")
                self.unofficial_available = True
            except:
                self.unofficial_available = False
        else:
            self.unofficial_available = False
            if phone_id and access_token:
                try:
                    import requests
                    self.requests = requests
                    print("[WhatsAppBridge] Official API initialized")
                except ImportError:
                    print("[WhatsAppBridge] requests not installed")
    
    def send_message(self, to_number: str, content: str) -> bool:
        """Send WhatsApp message"""
        try:
            if self.access_token:
                headers = {
                    'Authorization': f'Bearer {self.access_token}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    'messaging_product': 'whatsapp',
                    'recipient_type': 'individual',
                    'to': to_number,
                    'type': 'text',
                    'text': {'body': content}
                }
                
                response = self.requests.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    print(f"[WhatsAppBridge] Sent message to {to_number}")
                    return True
                else:
                    print(f"[WhatsAppBridge] Error: {response.text}")
                    return False
            else:
                print(f"[WhatsAppBridge] Would send to {to_number}: {content[:50]}...")
                return True
                
        except Exception as e:
            print(f"[WhatsAppBridge] Send error: {e}")
            return False
    
    def parse_webhook(self, request_data: dict) -> Optional[Message]:
        """Parse incoming WhatsApp webhook"""
        try:
            entries = request_data.get('entry', [])
            for entry in entries:
                changes = entry.get('changes', [])
                for change in changes:
                    value = change.get('value', {})
                    messages = value.get('messages', [])
                    
                    for msg in messages:
                        from_number = msg.get('from')
                        message_type = msg.get('type')
                        
                        # Get content based on type
                        if message_type == 'text':
                            content = msg.get('text', {}).get('body', '')
                        elif message_type == 'audio':
                            content = "[Voice message]"
                        elif message_type == 'image':
                            content = msg.get('caption', '[Image]')
                        else:
                            content = f"[{message_type}]"
                        
                        return Message(
                            message_id=msg.get('id'),
                            platform=MessagePlatform.WHATSAPP,
                            sender_id=from_number,
                            sender_name=value.get('contacts', [{}])[0].get('profile', {}).get('name'),
                            content=content,
                            timestamp=datetime.now().isoformat(),
                            metadata={'whatsapp_data': msg}
                        )
            return None
        except Exception as e:
            print(f"[WhatsAppBridge] Parse error: {e}")
            return None


class CallHandler:
    """
    Handle missed calls - transcribe voicemail and process
    """
    
    def __init__(self, voicemail_dir: str = "/tmp/janus_voicemail"):
        self.voicemail_dir = Path(voicemail_dir)
        self.voicemail_dir.mkdir(exist_ok=True, parents=True)
        self.call_handler: Optional[Callable[[Message, bytes], None]] = None
    
    def process_voicemail(self, caller_number: str, audio_data: bytes, 
                          caller_name: Optional[str] = None) -> Message:
        """Process a voicemail recording"""
        message_id = f"call_{int(time.time())}_{caller_number}"
        
        # Save voicemail
        voicemail_path = self.voicemail_dir / f"{message_id}.wav"
        with open(voicemail_path, 'wb') as f:
            f.write(audio_data)
        
        # Create message
        message = Message(
            message_id=message_id,
            platform=MessagePlatform.CALL,
            sender_id=caller_number,
            sender_name=caller_name,
            content="[Missed call with voicemail]",
            timestamp=datetime.now().isoformat(),
            media_urls=[str(voicemail_path)],
            metadata={'voicemail_path': str(voicemail_path)}
        )
        
        if self.call_handler:
            self.call_handler(message, audio_data)
        
        return message
    
    def transcribe_voicemail(self, audio_path: str) -> str:
        """Transcribe voicemail audio"""
        # This would integrate with WhisperSTT from voice_io_enhanced
        try:
            # Placeholder - actual implementation would use Whisper
            return "[Voicemail transcription would appear here]"
        except Exception as e:
            print(f"[CallHandler] Transcription error: {e}")
            return "[Could not transcribe voicemail]"


class UnifiedMessageServer:
    """
    Local HTTP server that receives messages from all platforms
    and routes them to Janus
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        self.host = host
        self.port = port
        self.server = None
        
        # Bridges
        self.sms: Optional[SMSBridge] = None
        self.telegram: Optional[TelegramBridge] = None
        self.whatsapp: Optional[WhatsAppBridge] = None
        self.calls: Optional[CallHandler] = None
        
        # Session management
        self.sessions = SessionManager()
        
        # Message handler (set by JanusCore)
        self.message_processor: Optional[Callable[[Message, ConversationSession], str]] = None
        
        # Response queue
        self.response_queue: queue.Queue = queue.Queue()
        
        # Stats
        self.messages_received = 0
        self.messages_sent = 0
    
    def setup_bridges(self,
                      sms_config: Optional[dict] = None,
                      telegram_token: Optional[str] = None,
                      whatsapp_config: Optional[dict] = None):
        """Setup all messaging bridges"""
        
        # SMS
        if sms_config:
            self.sms = SMSBridge(**sms_config)
            self.sms.message_handler = self._on_message
            print("[MessageServer] SMS bridge configured")
        
        # Telegram
        if telegram_token:
            self.telegram = TelegramBridge(telegram_token)
            self.telegram.message_handler = self._on_message
            self.telegram.start_polling()
            print("[MessageServer] Telegram bridge configured")
        
        # WhatsApp
        if whatsapp_config:
            self.whatsapp = WhatsAppBridge(**whatsapp_config)
            print("[MessageServer] WhatsApp bridge configured")
        
        # Call handler
        self.calls = CallHandler()
        self.calls.call_handler = self._on_call
        print("[MessageServer] Call handler configured")
    
    def _on_message(self, message: Message):
        """Handle incoming message from any platform"""
        self.messages_received += 1
        
        print(f"\n[Message] From {message.platform.value}: {message.sender_name or message.sender_id}")
        print(f"[Message] Content: {message.content[:100]}...")
        
        # Get or create session
        session = self.sessions.get_or_create_session(
            message.sender_id, message.platform
        )
        session.add_message(message)
        
        # Process message
        if self.message_processor:
            response = self.message_processor(message, session)
            
            if response:
                self.send_response(message, response)
    
    def _on_call(self, message: Message, audio_data: bytes):
        """Handle missed call with voicemail"""
        print(f"\n[Call] Missed call from {message.sender_id}")
        
        # Transcribe voicemail
        if self.calls:
            transcription = self.calls.transcribe_voicemail(
                message.metadata.get('voicemail_path', '')
            )
            message.content = f"[Voicemail]: {transcription}"
        
        # Process as message
        self._on_message(message)
    
    def send_response(self, original_message: Message, response_text: str):
        """Send response back through appropriate platform"""
        success = False
        
        if original_message.platform == MessagePlatform.SMS and self.sms:
            success = self.sms.send_message(
                original_message.sender_id,
                response_text
            )
        
        elif original_message.platform == MessagePlatform.TELEGRAM and self.telegram:
            chat_id = original_message.metadata.get('chat_id')
            if chat_id:
                success = self.telegram.send_message(
                    chat_id,
                    response_text,
                    reply_to=original_message.message_id
                )
        
        elif original_message.platform == MessagePlatform.WHATSAPP and self.whatsapp:
            success = self.whatsapp.send_message(
                original_message.sender_id,
                response_text
            )
        
        if success:
            self.messages_sent += 1
            print(f"[Message] Response sent")
        
        # Store in session
        session = self.sessions.get_session(
            original_message.sender_id,
            original_message.platform
        )
        if session:
            response_msg = Message(
                message_id=f"resp_{int(time.time())}",
                platform=original_message.platform,
                sender_id='janus',
                sender_name='Janus',
                content=response_text,
                timestamp=datetime.now().isoformat(),
                reply_to=original_message.message_id
            )
            session.add_message(response_msg)
    
    def start(self):
        """Start the HTTP server"""
        
        class RequestHandler(BaseHTTPRequestHandler):
            server_instance = self
            
            def do_POST(self):
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                
                try:
                    data = json.loads(post_data.decode('utf-8'))
                except:
                    data = urllib.parse.parse_qs(post_data.decode('utf-8'))
                    data = {k: v[0] if len(v) == 1 else v for k, v in data.items()}
                
                # Route to appropriate handler
                path = self.path
                
                if path == '/sms':
                    if self.server_instance.sms:
                        message = self.server_instance.sms.parse_incoming(data)
                        if message:
                            self.server_instance._on_message(message)
                
                elif path == '/whatsapp':
                    if self.server_instance.whatsapp:
                        message = self.server_instance.whatsapp.parse_webhook(data)
                        if message:
                            self.server_instance._on_message(message)
                
                elif path == '/call':
                    # Handle voicemail upload
                    caller = data.get('caller')
                    audio_b64 = data.get('audio')
                    if caller and audio_b64 and self.server_instance.calls:
                        import base64
                        audio_data = base64.b64decode(audio_b64)
                        self.server_instance.calls.process_voicemail(caller, audio_data)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'ok'}).encode())
            
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                status = {
                    'status': 'running',
                    'messages_received': self.server_instance.messages_received,
                    'messages_sent': self.server_instance.messages_sent,
                    'bridges': {
                        'sms': self.server_instance.sms is not None,
                        'telegram': self.server_instance.telegram is not None,
                        'whatsapp': self.server_instance.whatsapp is not None,
                    }
                }
                self.wfile.write(json.dumps(status).encode())
            
            def log_message(self, format, *args):
                # Suppress default logging
                pass
        
        RequestHandler.server_instance = self
        self.server = HTTPServer((self.host, self.port), RequestHandler)
        
        print(f"\n[MessageServer] HTTP server starting on http://{self.host}:{self.port}")
        print(f"[MessageServer] Endpoints:")
        print(f"  - GET  /       : Status")
        print(f"  - POST /sms    : SMS webhook")
        print(f"  - POST /whatsapp: WhatsApp webhook")
        print(f"  - POST /call   : Voicemail upload")
        
        server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        server_thread.start()
    
    def stop(self):
        """Stop the server"""
        if self.server:
            self.server.shutdown()
            print("[MessageServer] Server stopped")
    
    def get_status(self) -> dict:
        """Get server status"""
        return {
            'running': self.server is not None,
            'host': self.host,
            'port': self.port,
            'messages_received': self.messages_received,
            'messages_sent': self.messages_sent,
            'bridges': {
                'sms': self.sms is not None,
                'telegram': self.telegram is not None,
                'whatsapp': self.whatsapp is not None,
            },
            'active_sessions': len(self.sessions.sessions)
        }


def main():
    """Demo of messaging bridge"""
    print("=== Janus Messaging Bridge Demo ===\n")
    
    # Create server
    server = UnifiedMessageServer(host='localhost', port=8080)
    
    # Setup bridges (without real credentials for demo)
    server.setup_bridges(
        sms_config=None,  # {'twilio_sid': '...', 'twilio_token': '...', 'twilio_number': '...'}
        telegram_token=None,  # 'YOUR_BOT_TOKEN'
        whatsapp_config=None  # {'phone_id': '...', 'access_token': '...'}
    )
    
    # Message processor
    def process_message(message: Message, session: ConversationSession) -> str:
        """Process incoming message and generate response"""
        content_lower = message.content.lower()
        
        # Intent detection
        if any(w in content_lower for w in ['hello', 'hi', 'hey']):
            return f"Hello {message.sender_name or 'there'}! I'm Janus. How can I help you today?"
        
        elif 'book' in content_lower and 'flight' in content_lower:
            return "I'd be happy to help you book a flight! Where would you like to go and when?"
        
        elif 'reply' in content_lower and 'email' in content_lower:
            return "I can help you reply to emails. Which email would you like me to respond to?"
        
        elif 'remind' in content_lower:
            return "I'll set a reminder for you. What should I remind you about and when?"
        
        elif any(w in content_lower for w in ['bye', 'goodbye']):
            server.sessions.end_session(message.sender_id, message.platform)
            return "Goodbye! Feel free to message me anytime."
        
        else:
            # Context-aware response
            context = session.get_context(3)
            if context:
                return f"I understand: '{message.content}'. I'm processing your request with context from our conversation."
            return f"I received your message: '{message.content}'. How can I help with this?"
    
    server.message_processor = process_message
    
    # Start server
    server.start()
    
    print("\nServer is running. Test with:")
    print(f"  curl http://localhost:8080/")
    print(f"  curl -X POST http://localhost:8080/sms -d '{{\"From\": \"+1234567890\", \"Body\": \"Hello Janus\"}}'")
    print("\nPress Ctrl+C to stop.\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        server.stop()
        print(f"\nFinal status: {server.get_status()}")


if __name__ == "__main__":
    main()
