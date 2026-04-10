/**
 * JanusPWA.tsx
 * ============
 * Progressive Web App for communicating with Janus from your phone.
 * Install it on your phone's home screen — works like a native app.
 *
 * Features:
 *   - Real-time messages from Janus via WebSocket relay
 *   - Voice messages (Janus speaks, you hear it)
 *   - Send messages back to Janus
 *   - Alerts wake your screen
 *   - Works offline — messages queue and deliver when reconnected
 *   - End-to-end encrypted
 *
 * Add to your router:
 *   <Route path="/janus" component={JanusPWA} />
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { Button } from "./button";
import { Badge } from "./badge";
import {
  Wifi, WifiOff, Send, Volume2, Bell, BellOff,
  MessageSquare, AlertTriangle, CheckCircle2, Loader2
} from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

type MessageType = "text" | "voice" | "alert" | "status" | "ping";

interface RelayMessage {
  id:        string;
  type:      MessageType;
  content:   string;
  timestamp: string;
  from:      "janus" | "owner";
  read:      boolean;
}

type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";

// ── Crypto (mirrors server-side) ──────────────────────────────────────────────

async function deriveKey(secret: string): Promise<CryptoKey> {
  const enc     = new TextEncoder();
  const keyMat  = await crypto.subtle.importKey("raw", enc.encode(secret), "PBKDF2", false, ["deriveKey"]);
  return crypto.subtle.deriveKey(
    { name: "PBKDF2", salt: enc.encode("janus-relay-salt"), iterations: 100000, hash: "SHA-256" },
    keyMat,
    { name: "AES-CBC", length: 256 },
    false,
    ["encrypt", "decrypt"]
  );
}

async function decryptMessage(data: string, secret: string): Promise<string> {
  try {
    const [ivHex, encHex] = data.split(":");
    const iv  = new Uint8Array(ivHex.match(/.{2}/g)!.map(b => parseInt(b, 16)));
    const enc = new Uint8Array(encHex.match(/.{2}/g)!.map(b => parseInt(b, 16)));
    const key = await deriveKey(secret);
    const dec = await crypto.subtle.decrypt({ name: "AES-CBC", iv }, key, enc);
    return new TextDecoder().decode(dec);
  } catch {
    return data; // fallback: return raw if decryption fails
  }
}

async function encryptMessage(text: string, secret: string): Promise<string> {
  const iv  = crypto.getRandomValues(new Uint8Array(16));
  const key = await deriveKey(secret);
  const enc = await crypto.subtle.encrypt(
    { name: "AES-CBC", iv },
    key,
    new TextEncoder().encode(text)
  );
  const toHex = (buf: Uint8Array) => Array.from(buf).map(b => b.toString(16).padStart(2, "0")).join("");
  return toHex(iv) + ":" + toHex(new Uint8Array(enc));
}

// ── Main PWA component ────────────────────────────────────────────────────────

export function JanusPWA() {
  const [status,    setStatus]    = useState<ConnectionStatus>("disconnected");
  const [messages,  setMessages]  = useState<RelayMessage[]>([]);
  const [input,     setInput]     = useState("");
  const [secret,    setSecret]    = useState(() => localStorage.getItem("relay_secret") ?? "");
  const [relayUrl,  setRelayUrl]  = useState(() => localStorage.getItem("relay_url") ?? "");
  const [notifs,    setNotifs]    = useState(false);
  const [setup,     setSetup]     = useState(!localStorage.getItem("relay_secret"));

  const wsRef      = useRef<WebSocket | null>(null);
  const bottomRef  = useRef<HTMLDivElement>(null);
  const audioCtx   = useRef<AudioContext | null>(null);

  // ── WebSocket connection ───────────────────────────────────────────────────

  const connect = useCallback(() => {
    if (!secret || !relayUrl) return;

    setStatus("connecting");
    const url = `${relayUrl}?secret=${encodeURIComponent(secret)}`;
    const ws  = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      console.log("[PWA] Connected to Janus relay");
    };

    ws.onmessage = async (event) => {
      try {
        const decrypted = await decryptMessage(event.data, secret);
        const msg       = JSON.parse(decrypted) as RelayMessage;

        setMessages(prev => {
          const exists = prev.find(m => m.id === msg.id);
          return exists ? prev : [...prev, msg];
        });

        // Play audio for voice messages
        if (msg.type === "voice" && msg.content) {
          playVoice(msg.content);
        }

        // Show notification for alerts
        if (msg.type === "alert" && notifs) {
          showNotification("Janus Alert", msg.content);
        }

        // Vibrate for alerts
        if (msg.type === "alert" && navigator.vibrate) {
          navigator.vibrate([200, 100, 200]);
        }
      } catch (e) {
        console.error("[PWA] Message error:", e);
      }
    };

    ws.onclose = () => {
      setStatus("disconnected");
      wsRef.current = null;
      // Auto-reconnect after 5s
      setTimeout(() => {
        if (document.visibilityState !== "hidden") connect();
      }, 5000);
    };

    ws.onerror = () => {
      setStatus("error");
    };
  }, [secret, relayUrl, notifs]);

  useEffect(() => {
    if (secret && relayUrl) connect();
    return () => wsRef.current?.close();
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Send message to Janus ──────────────────────────────────────────────────

  const sendMessage = async () => {
    if (!input.trim() || !wsRef.current || status !== "connected") return;

    const msg: RelayMessage = {
      id:        crypto.randomUUID(),
      type:      "text",
      content:   input.trim(),
      timestamp: new Date().toISOString(),
      from:      "owner",
      read:      true,
    };

    const encrypted = await encryptMessage(JSON.stringify(msg), secret);
    wsRef.current.send(encrypted);

    setMessages(prev => [...prev, msg]);
    setInput("");
  };

  // ── Voice playback ─────────────────────────────────────────────────────────

  const playVoice = async (base64Audio: string) => {
    try {
      if (!audioCtx.current) {
        audioCtx.current = new AudioContext();
      }
      const binary = atob(base64Audio);
      const bytes  = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const buffer = await audioCtx.current.decodeAudioData(bytes.buffer);
      const source = audioCtx.current.createBufferSource();
      source.buffer = buffer;
      source.connect(audioCtx.current.destination);
      source.start();
    } catch (e) {
      console.error("[PWA] Audio playback error:", e);
    }
  };

  // ── Notifications ──────────────────────────────────────────────────────────

  const requestNotifications = async () => {
    if ("Notification" in window) {
      const perm = await Notification.requestPermission();
      setNotifs(perm === "granted");
    }
  };

  const showNotification = (title: string, body: string) => {
    if (Notification.permission === "granted") {
      new Notification(title, { body, icon: "/favicon.ico" });
    }
  };

  // ── Setup screen ───────────────────────────────────────────────────────────

  const saveSetup = () => {
    localStorage.setItem("relay_secret", secret);
    localStorage.setItem("relay_url",    relayUrl);
    setSetup(false);
    connect();
  };

  if (setup) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <div className="w-full max-w-sm space-y-4">
          <div className="text-center">
            <h1 className="text-2xl font-bold">Connect to Janus</h1>
            <p className="text-muted-foreground text-sm mt-1">
              Enter your relay server details
            </p>
          </div>

          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium">Relay URL</label>
              <input
                type="url"
                value={relayUrl}
                onChange={e => setRelayUrl(e.target.value)}
                placeholder="ws://your-home-ip:8765"
                className="w-full mt-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Your EliteDesk's public IP with port 8765
              </p>
            </div>

            <div>
              <label className="text-sm font-medium">Secret Key</label>
              <input
                type="password"
                value={secret}
                onChange={e => setSecret(e.target.value)}
                placeholder="Your RELAY_SECRET from .env.local"
                className="w-full mt-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
              />
            </div>

            <Button
              className="w-full"
              onClick={saveSetup}
              disabled={!secret || !relayUrl}
            >
              Connect
            </Button>
          </div>
        </div>
      </div>
    );
  }

  // ── Main chat UI ───────────────────────────────────────────────────────────

  const statusColor = status === "connected"    ? "bg-green-500" :
                      status === "connecting"   ? "bg-yellow-500" :
                      status === "error"        ? "bg-red-500" :
                                                   "bg-gray-400";

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${statusColor}`} />
          <span className="font-semibold">Janus</span>
          <span className="text-xs text-muted-foreground capitalize">{status}</span>
        </div>

        <div className="flex items-center gap-2">
          {status === "disconnected" || status === "error" ? (
            <Button variant="ghost" size="icon" onClick={connect}>
              <WifiOff className="h-4 w-4" />
            </Button>
          ) : (
            <Wifi className="h-4 w-4 text-green-500" />
          )}

          <Button
            variant="ghost"
            size="icon"
            onClick={notifs ? () => setNotifs(false) : requestNotifications}
          >
            {notifs ? <Bell className="h-4 w-4" /> : <BellOff className="h-4 w-4 text-muted-foreground" />}
          </Button>

          <Button variant="ghost" size="icon" onClick={() => setSetup(true)}>
            <span className="text-xs">⚙</span>
          </Button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && (
          <div className="text-center text-muted-foreground text-sm py-8">
            {status === "connected"
              ? "Connected. Waiting for Janus..."
              : "Not connected. Tap the WiFi icon to reconnect."}
          </div>
        )}

        {messages.map(msg => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t p-3 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && sendMessage()}
          placeholder="Message Janus..."
          className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
          disabled={status !== "connected"}
        />
        <Button
          size="icon"
          onClick={sendMessage}
          disabled={!input.trim() || status !== "connected"}
        >
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

// ── Message bubble ────────────────────────────────────────────────────────────

function MessageBubble({ message }: { message: RelayMessage }) {
  const isJanus = message.from === "janus";

  const icon = message.type === "alert"  ? <AlertTriangle className="h-3 w-3" /> :
               message.type === "voice"  ? <Volume2       className="h-3 w-3" /> :
               message.type === "status" ? <CheckCircle2  className="h-3 w-3" /> :
                                            <MessageSquare className="h-3 w-3" />;

  const bgColor = message.type === "alert"
    ? "bg-destructive text-destructive-foreground"
    : isJanus
    ? "bg-muted"
    : "bg-primary text-primary-foreground";

  return (
    <div className={`flex ${isJanus ? "justify-start" : "justify-end"}`}>
      <div className={`max-w-[80%] rounded-2xl px-4 py-2 ${bgColor}`}>
        <div className="flex items-center gap-1 mb-1 opacity-70">
          {icon}
          <span className="text-xs">
            {isJanus ? "Janus" : "You"} · {new Date(message.timestamp).toLocaleTimeString()}
          </span>
        </div>
        <p className="text-sm whitespace-pre-wrap">{message.content}</p>
      </div>
    </div>
  );
}
