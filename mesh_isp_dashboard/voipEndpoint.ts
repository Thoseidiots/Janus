/**
 * voipEndpoint.ts
 * ===============
 * Express REST endpoint that triggers Janus to call your phone.
 * Mount alongside notifyEndpoint in your Express app:
 *
 *   import { voipEndpointRouter } from './voipEndpoint';
 *   app.use('/api/voip', voipEndpointRouter);
 *
 * The endpoint calls the Python janus_voip.py bridge via child_process.
 * This keeps all the Asterisk/TTS logic in Python where it belongs.
 */

import { Router, type Request, type Response } from "express";
import { execFile } from "child_process";
import { promisify } from "util";
import path from "path";

const execFileAsync = promisify(execFile);

export const voipEndpointRouter = Router();

function isAuthorized(req: Request): boolean {
  const secret = process.env.NOTIFY_SECRET;
  if (!secret) {
    const ip = req.ip ?? req.socket.remoteAddress ?? "";
    return ip === "127.0.0.1" || ip === "::1" || ip === "::ffff:127.0.0.1";
  }
  const auth = req.headers.authorization ?? "";
  return auth.startsWith("Bearer ") && auth.slice(7) === secret;
}

// POST /api/voip/call
voipEndpointRouter.post("/call", async (req: Request, res: Response) => {
  if (!isAuthorized(req)) {
    res.status(401).json({ ok: false, error: "Unauthorized" });
    return;
  }

  const { message } = req.body ?? {};
  if (!message || typeof message !== "string") {
    res.status(400).json({ ok: false, error: "message is required" });
    return;
  }

  // Path to janus_voip.py (one level up from dashboard)
  const scriptPath = path.resolve(__dirname, "../../janus_voip.py");

  try {
    // Fire and forget — don't wait for the call to complete
    execFileAsync("python", [scriptPath, "--say", message.slice(0, 300)], {
      timeout: 10_000,
    }).catch(err => {
      console.error("[VoIP] Call script error:", err.message);
    });

    res.json({ ok: true, message: "Call initiated" });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    res.status(500).json({ ok: false, error: msg });
  }
});

// GET /api/voip/status
voipEndpointRouter.get("/status", async (_req: Request, res: Response) => {
  const scriptPath = path.resolve(__dirname, "../../janus_voip.py");
  try {
    const { stdout } = await execFileAsync("python", [scriptPath, "--status"], {
      timeout: 5_000,
    });
    const running = stdout.includes("running ✓");
    res.json({ ok: true, asteriskRunning: running, output: stdout.trim() });
  } catch {
    res.json({ ok: false, asteriskRunning: false });
  }
});
