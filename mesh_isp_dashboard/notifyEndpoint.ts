/**
 * notifyEndpoint.ts
 * =================
 * Simple Express REST endpoint that Janus Python calls to send notifications.
 * Mount this in your Express app alongside the tRPC router.
 *
 * Usage in your Express app:
 *   import { notifyEndpointRouter } from './notifyEndpoint';
 *   app.use('/api', notifyEndpointRouter);
 *
 * Janus Python then calls:
 *   POST http://localhost:3000/api/notify
 *   { "message": "Task completed", "level": "info" }
 *
 * Protected by a shared secret in NOTIFY_SECRET env var.
 * If NOTIFY_SECRET is not set, only localhost requests are accepted.
 */

import { Router, type Request, type Response } from "express";
import { notifyOwner } from "./janusNotify";

export const notifyEndpointRouter = Router();

function isAuthorized(req: Request): boolean {
  const secret = process.env.NOTIFY_SECRET;

  // If no secret configured, only allow localhost
  if (!secret) {
    const ip = req.ip ?? req.socket.remoteAddress ?? "";
    return ip === "127.0.0.1" || ip === "::1" || ip === "::ffff:127.0.0.1";
  }

  // Check Authorization header: "Bearer <secret>"
  const auth = req.headers.authorization ?? "";
  if (auth.startsWith("Bearer ") && auth.slice(7) === secret) {
    return true;
  }

  // Also accept as query param for simple GET pings
  if (req.query.secret === secret) {
    return true;
  }

  return false;
}

// POST /api/notify
notifyEndpointRouter.post("/notify", async (req: Request, res: Response) => {
  if (!isAuthorized(req)) {
    res.status(401).json({ ok: false, error: "Unauthorized" });
    return;
  }

  const { message, level, sms, email } = req.body ?? {};

  if (!message || typeof message !== "string") {
    res.status(400).json({ ok: false, error: "message is required" });
    return;
  }

  try {
    await notifyOwner(message, {
      level: level ?? "info",
      sms:   sms,
      email: email,
    });
    res.json({ ok: true });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    res.status(500).json({ ok: false, error: msg });
  }
});

// GET /api/notify/ping  — quick health check
notifyEndpointRouter.get("/notify/ping", (req: Request, res: Response) => {
  res.json({ ok: true, time: new Date().toISOString() });
});
