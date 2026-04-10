"""
janus_voip.py
=============
Self-hosted VoIP for Janus. Lets Janus call your phone.

Architecture:
  - Asterisk PBX runs on your EliteDesk (via Docker or WSL)
  - This module connects to Asterisk via AMI (Asterisk Manager Interface)
  - Janus generates speech with pyttsx3 (no API, runs locally)
  - Asterisk places the outbound call to your Verizon number
  - Your phone rings, you answer, Janus speaks

No Twilio. No monthly fees beyond a SIP trunk (~$1/month at VoIP.ms).

Requirements:
  pip install pyttsx3
  Docker Desktop (for Asterisk container)

Setup:
  1. Run: python janus_voip.py --setup
     This generates your Asterisk config and Docker compose file.
  2. Run: docker-compose -f asterisk-compose.yml up -d
  3. Configure your SIP trunk (VoIP.ms or similar)
  4. Run: python janus_voip.py --test

Usage:
  from janus_voip import JanusVoIP
  voip = JanusVoIP()
  voip.call_owner("Task completed. Revenue goal is 45 percent complete.")
"""

from __future__ import annotations

import os
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

_OWNER_PHONE   = os.environ.get("OWNER_PHONE", "").replace("+1", "").replace("-", "").replace(" ", "")
_SIP_TRUNK     = os.environ.get("SIP_TRUNK_HOST", "sip.voip.ms")
_SIP_USER      = os.environ.get("SIP_TRUNK_USER", "")
_SIP_PASS      = os.environ.get("SIP_TRUNK_PASS", "")
_AMI_HOST      = os.environ.get("ASTERISK_HOST", "127.0.0.1")
_AMI_PORT      = int(os.environ.get("ASTERISK_AMI_PORT", "5038"))
_AMI_USER      = os.environ.get("ASTERISK_AMI_USER", "janus")
_AMI_PASS      = os.environ.get("ASTERISK_AMI_PASS", "janus_ami_secret")
_AUDIO_DIR     = Path(os.environ.get("ASTERISK_AUDIO_DIR", "/tmp/janus_audio"))


# ── Text-to-Speech ────────────────────────────────────────────────────────────

class JanusTTS:
    """
    Converts text to a WAV file using pyttsx3 (fully local, no API).
    Falls back to espeak on Linux.
    """

    def __init__(self):
        self._engine = None
        self._lock   = threading.Lock()

    def _get_engine(self):
        if self._engine is None:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                # Tune voice — slower and slightly lower pitch sounds more natural
                self._engine.setProperty("rate",   145)
                self._engine.setProperty("volume", 0.9)
                # Pick a male voice if available
                voices = self._engine.getProperty("voices")
                for v in voices:
                    if "male" in v.name.lower() or "david" in v.name.lower():
                        self._engine.setProperty("voice", v.id)
                        break
            except ImportError:
                print("[VoIP] pyttsx3 not installed — run: pip install pyttsx3")
        return self._engine

    def synthesize(self, text: str, output_path: str) -> bool:
        """Convert text to WAV. Returns True on success."""
        with self._lock:
            engine = self._get_engine()
            if engine is None:
                return self._espeak_fallback(text, output_path)
            try:
                engine.save_to_file(text, output_path)
                engine.runAndWait()
                return Path(output_path).exists()
            except Exception as e:
                print(f"[VoIP] TTS failed: {e}")
                return self._espeak_fallback(text, output_path)

    def _espeak_fallback(self, text: str, output_path: str) -> bool:
        """Use espeak if pyttsx3 isn't available (Linux/WSL)."""
        try:
            subprocess.run(
                ["espeak", "-w", output_path, "-s", "145", text],
                capture_output=True, timeout=10
            )
            return Path(output_path).exists()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


# ── Asterisk Manager Interface (AMI) ─────────────────────────────────────────

class AsteriskAMI:
    """
    Connects to Asterisk via AMI to originate outbound calls.
    AMI is a simple TCP text protocol — no library needed.
    """

    def __init__(self, host: str, port: int, username: str, password: str):
        self.host     = host
        self.port     = port
        self.username = username
        self.password = password
        self._sock:   Optional[socket.socket] = None

    def connect(self) -> bool:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(10)
            self._sock.connect((self.host, self.port))
            # Read banner
            self._sock.recv(1024)
            # Login
            self._send(
                f"Action: Login\r\n"
                f"Username: {self.username}\r\n"
                f"Secret: {self.password}\r\n\r\n"
            )
            response = self._recv()
            return "Success" in response
        except Exception as e:
            print(f"[VoIP] AMI connect failed: {e}")
            return False

    def originate_call(
        self,
        phone_number: str,
        audio_file:   str,
        caller_id:    str = "Janus <0000000000>",
    ) -> bool:
        """
        Originate an outbound call. When answered, plays audio_file.
        audio_file should be the path without extension (Asterisk adds it).
        """
        if not self._sock:
            if not self.connect():
                return False

        # Strip extension from audio path for Asterisk
        audio_path = str(audio_file).replace(".wav", "").replace(".gsm", "")

        action = (
            f"Action: Originate\r\n"
            f"Channel: SIP/{phone_number}@{_SIP_TRUNK}\r\n"
            f"Context: janus-outbound\r\n"
            f"Exten: s\r\n"
            f"Priority: 1\r\n"
            f"CallerID: {caller_id}\r\n"
            f"Timeout: 30000\r\n"
            f"Variable: AUDIO_FILE={audio_path}\r\n"
            f"Async: true\r\n\r\n"
        )

        try:
            self._send(action)
            response = self._recv()
            return "Success" in response or "Queued" in response
        except Exception as e:
            print(f"[VoIP] Originate failed: {e}")
            return False

    def disconnect(self):
        if self._sock:
            try:
                self._send("Action: Logoff\r\n\r\n")
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _send(self, data: str):
        self._sock.sendall(data.encode("utf-8"))

    def _recv(self, timeout: float = 5.0) -> str:
        self._sock.settimeout(timeout)
        chunks = []
        try:
            while True:
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk.decode("utf-8", errors="replace"))
                if "\r\n\r\n" in chunks[-1]:
                    break
        except socket.timeout:
            pass
        return "".join(chunks)


# ── Main VoIP interface ───────────────────────────────────────────────────────

class JanusVoIP:
    """
    High-level VoIP interface for Janus.
    Generates speech and places calls via Asterisk.
    """

    def __init__(self):
        self._tts  = JanusTTS()
        self._ami  = AsteriskAMI(_AMI_HOST, _AMI_PORT, _AMI_USER, _AMI_PASS)
        _AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    def call_owner(self, message: str, phone: Optional[str] = None) -> bool:
        """
        Call the owner and speak a message.
        Returns True if call was successfully initiated.
        """
        target = (phone or _OWNER_PHONE).replace("+1", "").replace("-", "").replace(" ", "")
        if not target:
            print("[VoIP] OWNER_PHONE not configured")
            return False

        # Prepend a natural intro
        full_message = f"Hello, this is Janus. {message} End of message."

        # Generate audio
        audio_path = str(_AUDIO_DIR / f"call_{int(time.time())}.wav")
        print(f"[VoIP] Synthesizing speech: {message[:60]}...")
        ok = self._tts.synthesize(full_message, audio_path)
        if not ok:
            print("[VoIP] TTS failed — cannot place call")
            return False

        # Convert WAV to GSM (Asterisk's preferred format)
        gsm_path = audio_path.replace(".wav", ".gsm")
        self._convert_to_gsm(audio_path, gsm_path)
        final_audio = gsm_path if Path(gsm_path).exists() else audio_path

        # Place call
        print(f"[VoIP] Calling {target}...")
        ok = self._ami.originate_call(target, final_audio)
        if ok:
            print(f"[VoIP] Call initiated to {target}")
        else:
            print(f"[VoIP] Call failed — is Asterisk running?")
        return ok

    def _convert_to_gsm(self, wav_path: str, gsm_path: str):
        """Convert WAV to GSM using sox or ffmpeg if available."""
        for cmd in [
            ["sox", wav_path, "-r", "8000", "-c", "1", gsm_path],
            ["ffmpeg", "-y", "-i", wav_path, "-ar", "8000", "-ac", "1", gsm_path],
        ]:
            try:
                subprocess.run(cmd, capture_output=True, timeout=15)
                if Path(gsm_path).exists():
                    return
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

    def is_asterisk_running(self) -> bool:
        """Check if Asterisk AMI is reachable."""
        try:
            s = socket.socket()
            s.settimeout(2)
            s.connect((_AMI_HOST, _AMI_PORT))
            s.close()
            return True
        except Exception:
            return False


# ── Asterisk config generator ─────────────────────────────────────────────────

def generate_asterisk_config() -> dict[str, str]:
    """Generate minimal Asterisk config files for outbound calling."""

    sip_conf = f"""[general]
context=default
allowoverlap=no
udpbindaddr=0.0.0.0
tcpenable=no
transport=udp
srvlookup=yes

; SIP trunk to VoIP.ms (or your provider)
[voipms]
type=peer
host={_SIP_TRUNK}
username={_SIP_USER or 'YOUR_SIP_USERNAME'}
secret={_SIP_PASS or 'YOUR_SIP_PASSWORD'}
fromuser={_SIP_USER or 'YOUR_SIP_USERNAME'}
fromdomain={_SIP_TRUNK}
insecure=port,invite
qualify=yes
nat=force_rport,comedia
"""

    extensions_conf = f"""[janus-outbound]
; Janus outbound call context
; Plays the audio file then hangs up
exten => s,1,Answer()
exten => s,n,Wait(1)
exten => s,n,Playback(${{AUDIO_FILE}})
exten => s,n,Hangup()

[default]
exten => _X.,1,Dial(SIP/${{EXTEN}}@voipms,30)
exten => _X.,n,Hangup()
"""

    manager_conf = f"""[general]
enabled=yes
port=5038
bindaddr=127.0.0.1

[{_AMI_USER}]
secret={_AMI_PASS}
permit=127.0.0.1/255.255.255.0
read=all
write=all
"""

    docker_compose = f"""version: '3.8'
services:
  asterisk:
    image: andrius/asterisk:latest
    container_name: janus-asterisk
    restart: unless-stopped
    network_mode: host
    volumes:
      - ./asterisk-config/sip.conf:/etc/asterisk/sip.conf
      - ./asterisk-config/extensions.conf:/etc/asterisk/extensions.conf
      - ./asterisk-config/manager.conf:/etc/asterisk/manager.conf
      - {str(_AUDIO_DIR)}:/var/lib/asterisk/sounds/janus
    ports:
      - "5060:5060/udp"
      - "5038:5038"
      - "10000-10100:10000-10100/udp"
"""

    return {
        "asterisk-config/sip.conf":        sip_conf,
        "asterisk-config/extensions.conf": extensions_conf,
        "asterisk-config/manager.conf":    manager_conf,
        "asterisk-compose.yml":            docker_compose,
    }


def setup():
    """Generate all config files needed to run Asterisk."""
    configs = generate_asterisk_config()
    for path, content in configs.items():
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        print(f"  Created: {path}")

    print("\nSetup complete. Next steps:")
    print("1. Edit asterisk-config/sip.conf with your VoIP.ms credentials")
    print("   (Sign up free at voip.ms — pay-as-you-go, ~$0.01/min)")
    print("2. Add to .env.local:")
    print("   OWNER_PHONE=7578762492")
    print("   SIP_TRUNK_USER=your_voipms_username")
    print("   SIP_TRUNK_PASS=your_voipms_password")
    print("   ASTERISK_AMI_PASS=janus_ami_secret")
    print("3. Start Asterisk:")
    print("   docker-compose -f asterisk-compose.yml up -d")
    print("4. Test:")
    print("   python janus_voip.py --test")


# ── Module-level singleton ────────────────────────────────────────────────────

_voip: Optional[JanusVoIP] = None

def get_voip() -> JanusVoIP:
    global _voip
    if _voip is None:
        _voip = JanusVoIP()
    return _voip


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Janus VoIP")
    parser.add_argument("--setup",   action="store_true", help="Generate Asterisk config files")
    parser.add_argument("--test",    action="store_true", help="Place a test call")
    parser.add_argument("--status",  action="store_true", help="Check if Asterisk is running")
    parser.add_argument("--say",     type=str, help="Call owner and say this message")
    parser.add_argument("--tts",     type=str, help="Test TTS only (no call)")
    args = parser.parse_args()

    if args.setup:
        print("Generating Asterisk config...")
        setup()

    elif args.status:
        voip = JanusVoIP()
        ok   = voip.is_asterisk_running()
        print(f"Asterisk {'running ✓' if ok else 'not running ✗'} at {_AMI_HOST}:{_AMI_PORT}")

    elif args.tts:
        print(f"Synthesizing: {args.tts}")
        tts  = JanusTTS()
        path = "/tmp/janus_tts_test.wav"
        ok   = tts.synthesize(args.tts, path)
        print(f"Audio saved to {path}" if ok else "TTS failed")

    elif args.test:
        print("Placing test call...")
        voip = JanusVoIP()
        if not voip.is_asterisk_running():
            print("Asterisk is not running. Run: docker-compose -f asterisk-compose.yml up -d")
        else:
            ok = voip.call_owner("This is a test call from Janus. Everything is working correctly.")
            print("Call initiated ✓" if ok else "Call failed ✗")

    elif args.say:
        voip = JanusVoIP()
        ok   = voip.call_owner(args.say)
        print("Call initiated ✓" if ok else "Call failed ✗")

    else:
        parser.print_help()
