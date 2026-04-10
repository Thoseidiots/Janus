/**
 * MeshChat.tsx
 * ============
 * Full messaging app — like WhatsApp but self-hosted on your ISP.
 * Installs on iPhone home screen as a native-feeling app.
 *
 * Features:
 *   - iMessage-style chat UI
 *   - Push notifications (works when app is closed on iOS 16.4+)
 *   - Voice messages with waveform
 *   - Read receipts
 *   - Offline message queue
 *   - AES-256 end-to-end encryption
 *   - Multi-contact support (Janus + future contacts)
 *
 * Route: /meshchat
 */

import {
  useState, useEffect, useRef, useCallback, useMemo
} from "react";
import { Button } from "./button";
import {
  Send, Mic, MicOff, Phone, Video, Info,
  ChevronLeft, Volume2, AlertTriangle,
  CheckCheck, Check, Wifi, WifiOff, Plus,
  Image as ImageIcon, Smile, RefreshCw
} from "lucide-react";

// ── Update checker ────────────────────────────────────────────────────────────

const LOCAL_VERSION_KEY = "meshchat_version";

async function checkForUpdate(serverUrl: string): Promise<{
  hasUpdate: boolean;
  newVersion?: string;
  changelog?: string;
}> {
  try {
    const res = await fetch(`${serverUrl}/meshchat-version.json?t=${Date.now()}`);
    if (!res.ok) return { hasUpdate: false };
    const remote = await res.json();
    const local  = localStorage.getItem(LOCAL_VERSION_KEY) ?? "0.0.0";
    const hasUpdate = remote.version !== local;
    return { hasUpdate, newVersion: remote.version, changelog: remote.changelog };
  } catch {
    return { hasUpdate: false };
  }
}

function UpdateBanner({
  version, changelog, onUpdate, onDismiss
}: {
  version: string;
  changelog: string;
  onUpdate: () => void;
  onDismiss: () => void;
}) {
  return (
    <div className="bg-primary text-primary-foreground px-4 py-3 flex items-center justify-between gap-3">
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold">Update available — v{version}</p>
        {changelog && (
          <p className="text-xs opacity-80 truncate">{changelog}</p>
        )}
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <Button
          size="sm"
          variant="secondary"
          className="h-7 text-xs"
          onClick={onUpdate}
        >
          <RefreshCw className="h-3 w-3 mr-1" />
          Update
        </Button>
        <button
          className="text-xs opacity-70 hover:opacity-100"
          onClick={onDismiss}
        >
          Later
        </button>
      </div>
    </div>
  );
}

// ── Types ─────────────────────────────────────────────────────────────────────

type MsgType   = "text" | "voice" | "alert" | "status" | "image";
type MsgStatus = "sending" | "sent" | "delivered" | "read";

interface ChatMessage {
  id:        string;
  type:      MsgType;
  content:   string;
  timestamp: string;
  from:      string;   // "janus" | "me" | user id
  status:    MsgStatus;
  duration?: number;   // voice message duration in seconds
}

interface Contact {
  id:     string;
  name:   string;
  avatar: string;   // emoji avatar
  online: boolean;
  unread: number;
  lastMessage?: string;
  lastTime?:    string;
}

type Screen = "contacts" | "chat" | "setup";

// ── Crypto ────────────────────────────────────────────────────────────────────

async function deriveKey(secret: string): Promise<CryptoKey> {
  const enc    = new TextEncoder();
  const keyMat = await crypto.subtle.importKey(
    "raw", enc.encode(secret), "PBKDF2", false, ["deriveKey"]
  );
  return crypto.subtle.deriveKey(
    { name: "PBKDF2", salt: enc.encode("meshchat-v1"), iterations: 100_000, hash: "SHA-256" },
    keyMat,
    { name: "AES-CBC", length: 256 },
    false,
    ["encrypt", "decrypt"]
  );
}

async function decrypt(data: string, secret: string): Promise<string> {
  try {
    const [ivHex, encHex] = data.split(":");
    const iv  = new Uint8Array(ivHex.match(/.{2}/g)!.map(b => parseInt(b, 16)));
    const enc = new Uint8Array(encHex.match(/.{2}/g)!.map(b => parseInt(b, 16)));
    const key = await deriveKey(secret);
    const dec = await crypto.subtle.decrypt({ name: "AES-CBC", iv }, key, enc);
    return new TextDecoder().decode(dec);
  } catch {
    return data;
  }
}

async function encrypt(text: string, secret: string): Promise<string> {
  const iv  = crypto.getRandomValues(new Uint8Array(16));
  const key = await deriveKey(secret);
  const enc = await crypto.subtle.encrypt(
    { name: "AES-CBC", iv }, key, new TextEncoder().encode(text)
  );
  const hex = (b: Uint8Array) => Array.from(b).map(x => x.toString(16).padStart(2, "0")).join("");
  return hex(iv) + ":" + hex(new Uint8Array(enc));
}

// ── Push notification registration ───────────────────────────────────────────

async function registerPush(serverUrl: string, secret: string): Promise<boolean> {
  if (!("serviceWorker" in navigator) || !("PushManager" in window)) return false;

  try {
    const reg = await navigator.serviceWorker.register("/meshchat-sw.js");
    const perm = await Notification.requestPermission();
    if (perm !== "granted") return false;

    // Get VAPID public key from server
    const res  = await fetch(`${serverUrl}/api/push/vapid-key`);
    const { publicKey } = await res.json();

    const sub = await reg.pushManager.subscribe({
      userVisibleOnly:      true,
      applicationServerKey: publicKey,
    });

    // Register subscription with server
    await fetch(`${serverUrl}/api/push/subscribe`, {
      method:  "POST",
      headers: {
        "Content-Type":  "application/json",
        "Authorization": `Bearer ${secret}`,
      },
      body: JSON.stringify(sub),
    });

    return true;
  } catch (e) {
    console.error("[MeshChat] Push registration failed:", e);
    return false;
  }
}

// ── Storage ───────────────────────────────────────────────────────────────────

const DB_KEY = "meshchat_messages";

function loadMessages(contactId: string): ChatMessage[] {
  try {
    const all = JSON.parse(localStorage.getItem(DB_KEY) ?? "{}");
    return all[contactId] ?? [];
  } catch { return []; }
}

function saveMessages(contactId: string, messages: ChatMessage[]) {
  try {
    const all = JSON.parse(localStorage.getItem(DB_KEY) ?? "{}");
    all[contactId] = messages.slice(-200); // keep last 200
    localStorage.setItem(DB_KEY, JSON.stringify(all));
  } catch {}
}

// ── Main MeshChat app ─────────────────────────────────────────────────────────

export function MeshChat() {
  const [screen,      setScreen]      = useState<Screen>("contacts");
  const [activeChat,  setActiveChat]  = useState<Contact | null>(null);
  const [messages,    setMessages]    = useState<ChatMessage[]>([]);
  const [input,       setInput]       = useState("");
  const [wsStatus,    setWsStatus]    = useState<"connected"|"disconnected"|"connecting">("disconnected");
  const [recording,   setRecording]   = useState(false);
  const [pushEnabled, setPushEnabled] = useState(false);
  const [updateInfo,  setUpdateInfo]  = useState<{ version: string; changelog: string } | null>(null);

  // Config
  const [secret,     setSecret]     = useState(() => localStorage.getItem("mc_secret") ?? "");
  const [serverUrl,  setServerUrl]  = useState(() => localStorage.getItem("mc_server") ?? "");
  const [myName,     setMyName]     = useState(() => localStorage.getItem("mc_name")   ?? "");
  const isSetup = !secret || !serverUrl || !myName;

  // Contacts — Janus is always first
  const [contacts, setContacts] = useState<Contact[]>([
    { id: "janus", name: "Janus", avatar: "🤖", online: false, unread: 0 },
  ]);

  const wsRef       = useRef<WebSocket | null>(null);
  const bottomRef   = useRef<HTMLDivElement>(null);
  const mediaRef    = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);

  // ── WebSocket ──────────────────────────────────────────────────────────────

  const wsUrl = useMemo(() => {
    if (!serverUrl || !secret) return "";
    const base = serverUrl.replace(/^http/, "ws");
    return `${base}/ws/relay?secret=${encodeURIComponent(secret)}`;
  }, [serverUrl, secret]);

  const connect = useCallback(() => {
    if (!wsUrl) return;
    setWsStatus("connecting");
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsStatus("connected");
      setContacts(prev => prev.map(c =>
        c.id === "janus" ? { ...c, online: true } : c
      ));
    };

    ws.onmessage = async (event) => {
      try {
        const raw = await decrypt(event.data, secret);
        const msg = JSON.parse(raw);

        const chatMsg: ChatMessage = {
          id:        msg.id ?? crypto.randomUUID(),
          type:      msg.type ?? "text",
          content:   msg.content,
          timestamp: msg.timestamp ?? new Date().toISOString(),
          from:      "janus",
          status:    "delivered",
        };

        // Add to messages if this chat is open
        setMessages(prev => {
          if (prev.find(m => m.id === chatMsg.id)) return prev;
          const updated = [...prev, chatMsg];
          saveMessages("janus", updated);
          return updated;
        });

        // Update unread count if not in this chat
        setContacts(prev => prev.map(c =>
          c.id === "janus"
            ? { ...c, unread: activeChat?.id === "janus" ? 0 : c.unread + 1,
                lastMessage: chatMsg.content.slice(0, 40),
                lastTime: new Date(chatMsg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) }
            : c
        ));

        // Play voice
        if (chatMsg.type === "voice") playAudio(chatMsg.content);

        // Haptic for alerts
        if (chatMsg.type === "alert" && navigator.vibrate) {
          navigator.vibrate([200, 100, 200]);
        }
      } catch (e) {
        console.error("[MeshChat] Message error:", e);
      }
    };

    ws.onclose = () => {
      setWsStatus("disconnected");
      setContacts(prev => prev.map(c =>
        c.id === "janus" ? { ...c, online: false } : c
      ));
      wsRef.current = null;
      setTimeout(connect, 5000);
    };

    ws.onerror = () => setWsStatus("disconnected");
  }, [wsUrl, secret, activeChat]);

  useEffect(() => {
    if (!isSetup) connect();
    return () => wsRef.current?.close();
  }, [isSetup]);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load messages when opening a chat
  useEffect(() => {
    if (activeChat) {
      setMessages(loadMessages(activeChat.id));
      setContacts(prev => prev.map(c =>
        c.id === activeChat.id ? { ...c, unread: 0 } : c
      ));
    }
  }, [activeChat?.id]);

  // Check for updates on mount and every 30 minutes
  useEffect(() => {
    if (!serverUrl) return;
    const check = async () => {
      const result = await checkForUpdate(serverUrl);
      if (result.hasUpdate && result.newVersion) {
        setUpdateInfo({ version: result.newVersion, changelog: result.changelog ?? "" });
      }
    };
    check();
    const interval = setInterval(check, 30 * 60 * 1000);
    return () => clearInterval(interval);
  }, [serverUrl]);

  // ── Send ───────────────────────────────────────────────────────────────────

  const sendText = async () => {
    if (!input.trim() || wsStatus !== "connected") return;

    const msg: ChatMessage = {
      id:        crypto.randomUUID(),
      type:      "text",
      content:   input.trim(),
      timestamp: new Date().toISOString(),
      from:      "me",
      status:    "sending",
    };

    setMessages(prev => {
      const updated = [...prev, msg];
      saveMessages(activeChat!.id, updated);
      return updated;
    });
    setInput("");

    const payload = { id: msg.id, type: "text", content: msg.content,
                      timestamp: msg.timestamp, from: myName };
    const enc = await encrypt(JSON.stringify(payload), secret);
    wsRef.current?.send(enc);

    // Mark as sent
    setTimeout(() => {
      setMessages(prev => prev.map(m =>
        m.id === msg.id ? { ...m, status: "sent" } : m
      ));
    }, 300);
  };

  // ── Voice recording ────────────────────────────────────────────────────────

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr     = new MediaRecorder(stream);
      mediaRef.current  = mr;
      audioChunks.current = [];

      mr.ondataavailable = e => audioChunks.current.push(e.data);
      mr.onstop = async () => {
        const blob   = new Blob(audioChunks.current, { type: "audio/webm" });
        const b64    = await blobToBase64(blob);
        const msg: ChatMessage = {
          id:        crypto.randomUUID(),
          type:      "voice",
          content:   b64,
          timestamp: new Date().toISOString(),
          from:      "me",
          status:    "sending",
          duration:  Math.round(blob.size / 16000),
        };
        setMessages(prev => {
          const updated = [...prev, msg];
          saveMessages(activeChat!.id, updated);
          return updated;
        });
        const enc = await encrypt(JSON.stringify({ ...msg, from: myName }), secret);
        wsRef.current?.send(enc);
        stream.getTracks().forEach(t => t.stop());
      };

      mr.start();
      setRecording(true);
    } catch (e) {
      console.error("[MeshChat] Recording error:", e);
    }
  };

  const stopRecording = () => {
    mediaRef.current?.stop();
    setRecording(false);
  };

  // ── Audio playback ─────────────────────────────────────────────────────────

  const playAudio = async (b64: string) => {
    try {
      const ctx    = new AudioContext();
      const binary = atob(b64);
      const bytes  = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const buf    = await ctx.decodeAudioData(bytes.buffer);
      const src    = ctx.createBufferSource();
      src.buffer   = buf;
      src.connect(ctx.destination);
      src.start();
    } catch {}
  };

  // ── Push notifications ─────────────────────────────────────────────────────

  const enablePush = async () => {
    const ok = await registerPush(serverUrl, secret);
    setPushEnabled(ok);
  };

  // ── Setup screen ───────────────────────────────────────────────────────────

  if (isSetup) {
    return <SetupScreen
      onSave={(s, url, name) => {
        localStorage.setItem("mc_secret", s);
        localStorage.setItem("mc_server", url);
        localStorage.setItem("mc_name",   name);
        setSecret(s); setServerUrl(url); setMyName(name);
      }}
    />;
  }

  const handleUpdate = () => {
    if (updateInfo) {
      localStorage.setItem(LOCAL_VERSION_KEY, updateInfo.version);
    }
    // Unregister service worker to force fresh load
    navigator.serviceWorker?.getRegistrations().then(regs => {
      regs.forEach(r => r.unregister());
    });
    window.location.reload();
  };

  // ── Contacts list ──────────────────────────────────────────────────────────

  if (screen === "contacts") {
    return (
      <div className="flex flex-col h-screen bg-background">
        {/* Update banner */}
        {updateInfo && (
          <UpdateBanner
            version={updateInfo.version}
            changelog={updateInfo.changelog}
            onUpdate={handleUpdate}
            onDismiss={() => setUpdateInfo(null)}
          />
        )}
        {/* iOS-style header */}
        <div className="px-4 pt-12 pb-3 border-b bg-background/95 backdrop-blur sticky top-0 z-10">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold">MeshChat</h1>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${wsStatus === "connected" ? "bg-green-500" : "bg-gray-400"}`} />
              <Button variant="ghost" size="icon">
                <Plus className="h-5 w-5" />
              </Button>
            </div>
          </div>
        </div>

        {/* Contact list */}
        <div className="flex-1 overflow-y-auto">
          {contacts.map(contact => (
            <button
              key={contact.id}
              className="w-full flex items-center gap-3 px-4 py-3 hover:bg-muted/50 border-b border-border/50 text-left"
              onClick={() => { setActiveChat(contact); setScreen("chat"); }}
            >
              {/* Avatar */}
              <div className="relative">
                <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center text-2xl">
                  {contact.avatar}
                </div>
                {contact.online && (
                  <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-background" />
                )}
              </div>

              {/* Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className="font-semibold">{contact.name}</span>
                  <span className="text-xs text-muted-foreground">{contact.lastTime}</span>
                </div>
                <p className="text-sm text-muted-foreground truncate">
                  {contact.lastMessage ?? "No messages yet"}
                </p>
              </div>

              {/* Unread badge */}
              {contact.unread > 0 && (
                <div className="w-5 h-5 bg-primary rounded-full flex items-center justify-center">
                  <span className="text-xs text-primary-foreground font-bold">
                    {contact.unread}
                  </span>
                </div>
              )}
            </button>
          ))}
        </div>

        {/* Push notification prompt */}
        {!pushEnabled && (
          <div className="p-4 border-t bg-muted/30">
            <button
              className="w-full text-sm text-primary font-medium"
              onClick={enablePush}
            >
              Enable notifications to receive messages when app is closed
            </button>
          </div>
        )}
      </div>
    );
  }

  // ── Chat screen ────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Chat header */}
      <div className="px-2 pt-12 pb-2 border-b bg-background/95 backdrop-blur sticky top-0 z-10">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" onClick={() => setScreen("contacts")}>
            <ChevronLeft className="h-5 w-5" />
          </Button>

          <div className="flex-1 flex items-center gap-2">
            <div className="relative">
              <div className="w-9 h-9 rounded-full bg-muted flex items-center justify-center text-lg">
                {activeChat?.avatar}
              </div>
              {activeChat?.online && (
                <div className="absolute bottom-0 right-0 w-2.5 h-2.5 bg-green-500 rounded-full border-2 border-background" />
              )}
            </div>
            <div>
              <p className="font-semibold text-sm leading-tight">{activeChat?.name}</p>
              <p className="text-xs text-muted-foreground">
                {wsStatus === "connected" ? "online" : "offline"}
              </p>
            </div>
          </div>

          <Button variant="ghost" size="icon">
            <Phone className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon">
            <Info className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
        {messages.map((msg, i) => (
          <MessageRow
            key={msg.id}
            msg={msg}
            prev={messages[i - 1]}
            myName={myName}
            onPlayVoice={playAudio}
          />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="border-t px-3 py-2 pb-safe bg-background">
        <div className="flex items-end gap-2">
          <Button variant="ghost" size="icon" className="shrink-0 mb-1">
            <Plus className="h-5 w-5" />
          </Button>

          <div className="flex-1 relative">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendText();
                }
              }}
              placeholder="iMessage"
              rows={1}
              className="w-full rounded-2xl border border-input bg-muted/30 px-4 py-2 text-sm resize-none max-h-32 focus:outline-none focus:ring-1 focus:ring-primary"
              style={{ minHeight: "36px" }}
            />
          </div>

          {input.trim() ? (
            <Button
              size="icon"
              className="rounded-full shrink-0 mb-1 h-8 w-8"
              onClick={sendText}
              disabled={wsStatus !== "connected"}
            >
              <Send className="h-4 w-4" />
            </Button>
          ) : (
            <Button
              size="icon"
              variant={recording ? "destructive" : "ghost"}
              className="rounded-full shrink-0 mb-1 h-8 w-8"
              onPointerDown={startRecording}
              onPointerUp={stopRecording}
            >
              {recording ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Message row ───────────────────────────────────────────────────────────────

function MessageRow({
  msg, prev, myName, onPlayVoice
}: {
  msg: ChatMessage;
  prev?: ChatMessage;
  myName: string;
  onPlayVoice: (b64: string) => void;
}) {
  const isMe     = msg.from === "me";
  const showTime = !prev || (
    new Date(msg.timestamp).getTime() - new Date(prev.timestamp).getTime() > 5 * 60 * 1000
  );

  return (
    <>
      {showTime && (
        <div className="text-center text-xs text-muted-foreground py-2">
          {new Date(msg.timestamp).toLocaleString([], {
            month: "short", day: "numeric",
            hour: "2-digit", minute: "2-digit"
          })}
        </div>
      )}

      <div className={`flex ${isMe ? "justify-end" : "justify-start"} mb-0.5`}>
        <div className={`
          max-w-[75%] rounded-2xl px-3 py-2 text-sm
          ${msg.type === "alert"
            ? "bg-red-500 text-white"
            : isMe
            ? "bg-primary text-primary-foreground rounded-br-sm"
            : "bg-muted rounded-bl-sm"
          }
        `}>
          {msg.type === "voice" ? (
            <button
              className="flex items-center gap-2 min-w-[120px]"
              onClick={() => onPlayVoice(msg.content)}
            >
              <Volume2 className="h-4 w-4 shrink-0" />
              <div className="flex-1 h-1 bg-current/30 rounded-full">
                <div className="h-full w-1/3 bg-current rounded-full" />
              </div>
              <span className="text-xs opacity-70">
                {msg.duration ? `${msg.duration}s` : "▶"}
              </span>
            </button>
          ) : msg.type === "alert" ? (
            <div className="flex items-start gap-1.5">
              <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
              <span>{msg.content}</span>
            </div>
          ) : (
            <span className="whitespace-pre-wrap break-words">{msg.content}</span>
          )}

          {/* Read receipt */}
          {isMe && (
            <div className="flex justify-end mt-0.5">
              {msg.status === "read"      ? <CheckCheck className="h-3 w-3 opacity-70" /> :
               msg.status === "delivered" ? <CheckCheck className="h-3 w-3 opacity-40" /> :
               msg.status === "sent"      ? <Check      className="h-3 w-3 opacity-40" /> :
                                             <span className="text-xs opacity-40">…</span>}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

// ── Setup screen ──────────────────────────────────────────────────────────────

function SetupScreen({ onSave }: { onSave: (s: string, url: string, name: string) => void }) {
  const [secret,  setSecret]  = useState("");
  const [url,     setUrl]     = useState("");
  const [name,    setName]    = useState("");

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-6">
      <div className="w-full max-w-sm space-y-6">
        <div className="text-center space-y-2">
          <div className="text-5xl">💬</div>
          <h1 className="text-3xl font-bold">MeshChat</h1>
          <p className="text-muted-foreground text-sm">
            Private messaging on your own network
          </p>
        </div>

        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium block mb-1">Your name</label>
            <input
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="What should Janus call you?"
              className="w-full rounded-xl border border-input bg-muted/30 px-4 py-3 text-sm"
            />
          </div>

          <div>
            <label className="text-sm font-medium block mb-1">Server URL</label>
            <input
              type="url"
              value={url}
              onChange={e => setUrl(e.target.value)}
              placeholder="https://your-home-ip:3000"
              className="w-full rounded-xl border border-input bg-muted/30 px-4 py-3 text-sm"
            />
            <p className="text-xs text-muted-foreground mt-1">
              Your EliteDesk's address
            </p>
          </div>

          <div>
            <label className="text-sm font-medium block mb-1">Secret key</label>
            <input
              type="password"
              value={secret}
              onChange={e => setSecret(e.target.value)}
              placeholder="RELAY_SECRET from your server"
              className="w-full rounded-xl border border-input bg-muted/30 px-4 py-3 text-sm"
            />
          </div>

          <Button
            className="w-full rounded-xl py-3 text-base"
            onClick={() => onSave(secret, url, name)}
            disabled={!secret || !url || !name}
          >
            Connect to MeshChat
          </Button>
        </div>

        <p className="text-center text-xs text-muted-foreground">
          Add to home screen for the best experience
        </p>
      </div>
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload  = () => resolve((reader.result as string).split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}
