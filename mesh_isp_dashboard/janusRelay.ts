/**
 * janusRelay.ts
 * =============
 * Self-hosted private relay server for Janus ↔ Owner communication.
 * No carriers. No third parties. Runs on your EliteDesk.
 *
 * How it works:
 *   - WebSocket server on your EliteDesk (always on)
 *   - Your phone's PWA connects to it over the internet
 *   - Janus pushes messages, voice, and alerts through the relay
 *   - End-to-end encrypted with a shared secret
 *   - Messages queue if your phone is offline, delivered when reconnected
 *
 * Mount in your Express app:
 *   import { attachJanusRelay } from './janusRelay';
 *   attachJanusRelay(httpServer);
 *
 * Janus Python calls:
 *   POST /api/relay/send  { type: "text"|"voice"|"alert", content: "..." }
 */

import { WebSocketServer, WebSocket } from "ws";
import type { Server } from "http";
import crypto from "crypto";
import { Router, type Request, type Response } from "express";

// ── Config ────────────────────────────────────────────────────────────────────

const RELAY_SECRET  = process.env.RELAY_SECRET  ?? "change-this-secret";
const RELAY_PORT    = parseInt(process.env.RELAY_WS_PORT ?? "8765");
const MAX_QUEUE     = 100;   // max queued messages while phone is offline

// ── Message types ─────────────────────────────────────────────────────────────

export type RelayMessageType = "text" | "voice" | "alert" | "status" | "ping";

export interface RelayMessage {
  id:        string;
  type:      RelayMessageType;
  content:   string;           // text or base64 audio
  timestamp: string;
  from:      "janus" | "owner";
  read:      boolean;
}

// ── Encryption ────────────────────────────────────────────────────────────────

function encrypt(text: string, secret: string): string {
  const iv  = crypto.randomBytes(16);
  const key = crypto.scryptSync(secret, "janus-relay-salt", 32);
  const cipher = crypto.createCipheriv("aes-256-cbc", key, iv);
  const encrypted = Buffer.concat([cipher.update(text, "utf8"), cipher.final()]);
  return iv.toString("hex") + ":" + encrypted.toString("hex");
}

function decrypt(data: string, secret: string): string {
  const [ivHex, encHex] = data.split(":");
  const iv  = Buffer.from(ivHex, "hex");
  const key = crypto.scryptSync(secret, "janus-relay-salt", 32);
  const decipher = crypto.createDecipheriv("aes-256-cbc", key, iv);
  const decrypted = Buffer.concat([
    decipher.update(Buffer.from(encHex, "hex")),
    decipher.final(),
  ]);
  return decrypted.toString("utf8");
}

// ── Relay server ──────────────────────────────────────────────────────────────

class JanusRelayServer {
  private wss:      WebSocketServer | null = null;
  private clients:  Set<WebSocket>         = new Set();
  private queue:    RelayMessage[]         = [];   // messages while phone offline
  private history:  RelayMessage[]         = [];   // last 50 messages

  start(server?: Server) {
    if (server) {
      // Attach to existing HTTP server on /ws/relay path
      this.wss = new WebSocketServer({ server, path: "/ws/relay" });
    } else {
      // Standalone WebSocket server
      this.wss = new WebSocketServer({ port: RELAY_PORT });
    }

    this.wss.on("connection", (ws, req) => {
      // Authenticate via query param: ?secret=...
      const url    = new URL(req.url ?? "/", "http://localhost");
      const secret = url.searchParams.get("secret");

      if (secret !== RELAY_SECRET) {
        ws.close(4001, "Unauthorized");
        return;
      }

      console.log("[Relay] Phone connected");
      this.clients.add(ws);

      // Flush queued messages
      if (this.queue.length > 0) {
        console.log(`[Relay] Flushing ${this.queue.length} queued messages`);
        for (const msg of this.queue) {
          this._sendToClient(ws, msg);
        }
        this.queue = [];
      }

      ws.on("message", (data) => {
        try {
          const raw     = data.toString();
          const decrypted = decrypt(raw, RELAY_SECRET);
          const msg     = JSON.parse(decrypted) as RelayMessage;
          console.log(`[Relay] Message from owner: ${msg.content.slice(0, 60)}`);
          // Forward to Janus Python via file queue
          this._forwardToJanus(msg);
        } catch (e) {
          console.error("[Relay] Message parse error:", e);
        }
      });

      ws.on("close", () => {
        this.clients.delete(ws);
        console.log("[Relay] Phone disconnected");
      });

      ws.on("error", (err) => {
        console.error("[Relay] WebSocket error:", err.message);
        this.clients.delete(ws);
      });
    });

    console.log(`[Relay] WebSocket relay started on /ws/relay`);
  }

  /** Send a message to the owner's phone. Queues if offline. */
  send(type: RelayMessageType, content: string): RelayMessage {
    const msg: RelayMessage = {
      id:        crypto.randomUUID(),
      type,
      content,
      timestamp: new Date().toISOString(),
      from:      "janus",
      read:      false,
    };

    this.history.push(msg);
    if (this.history.length > 50) this.history.shift();

    const delivered = this._broadcast(msg);
    if (!delivered) {
      // Phone offline — queue it
      this.queue.push(msg);
      if (this.queue.length > MAX_QUEUE) this.queue.shift();
      console.log(`[Relay] Phone offline — queued message (${this.queue.length} in queue)`);
    }

    return msg;
  }

  getHistory():  RelayMessage[] { return this.history; }
  getQueue():    RelayMessage[] { return this.queue; }
  isConnected(): boolean        { return this.clients.size > 0; }

  private _broadcast(msg: RelayMessage): boolean {
    if (this.clients.size === 0) return false;
    let sent = false;
    for (const client of this.clients) {
      if (client.readyState === WebSocket.OPEN) {
        this._sendToClient(client, msg);
        sent = true;
      }
    }
    return sent;
  }

  private _sendToClient(ws: WebSocket, msg: RelayMessage) {
    try {
      const encrypted = encrypt(JSON.stringify(msg), RELAY_SECRET);
      ws.send(encrypted);
    } catch (e) {
      console.error("[Relay] Send error:", e);
    }
  }

  private _forwardToJanus(msg: RelayMessage) {
    // Write to a file that janus_autonomous_loop.py watches
    const fs   = require("fs");
    const path = require("path");
    const file = path.resolve(__dirname, "../../janus_inbox.jsonl");
    const line = JSON.stringify(msg) + "\n";
    fs.appendFileSync(file, line);
  }
}

// ── Singleton ─────────────────────────────────────────────────────────────────

export const relay = new JanusRelayServer();

export function attachJanusRelay(server: Server) {
  relay.start(server);
}

// ── Express REST router (for Janus Python to call) ────────────────────────────

export const relayRouter = Router();

function isAuthorized(req: Request): boolean {
  const secret = process.env.NOTIFY_SECRET ?? RELAY_SECRET;
  const ip     = req.ip ?? req.socket.remoteAddress ?? "";
  const isLocal = ip === "127.0.0.1" || ip === "::1" || ip === "::ffff:127.0.0.1";
  if (isLocal) return true;
  const auth = req.headers.authorization ?? "";
  return auth.startsWith("Bearer ") && auth.slice(7) === secret;
}

// POST /api/relay/send
relayRouter.post("/send", (req: Request, res: Response) => {
  if (!isAuthorized(req)) {
    res.status(401).json({ ok: false, error: "Unauthorized" });
    return;
  }

  const { type = "text", content } = req.body ?? {};
  if (!content) {
    res.status(400).json({ ok: false, error: "content required" });
    return;
  }

  const msg = relay.send(type as RelayMessageType, content);
  res.json({
    ok:        true,
    messageId: msg.id,
    delivered: relay.isConnected(),
    queued:    !relay.isConnected(),
  });
});

// GET /api/relay/status
relayRouter.get("/status", (_req: Request, res: Response) => {
  res.json({
    connected:    relay.isConnected(),
    queuedCount:  relay.getQueue().length,
    historyCount: relay.getHistory().length,
  });
});

// GET /api/relay/history
relayRouter.get("/history", (req: Request, res: Response) => {
  if (!isAuthorized(req)) {
    res.status(401).json({ ok: false, error: "Unauthorized" });
    return;
  }
  res.json({ messages: relay.getHistory() });
});
