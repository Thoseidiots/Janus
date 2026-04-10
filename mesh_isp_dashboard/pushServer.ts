/**
 * pushServer.ts
 * =============
 * Web Push notification server for MeshChat.
 * Sends push notifications to your iPhone even when the app is closed.
 *
 * iOS 16.4+ supports Web Push for PWAs added to home screen.
 *
 * Uses VAPID (Voluntary Application Server Identification) — a standard
 * that lets your server send pushes without any third-party service.
 * The keys are generated once and stored locally. Entirely self-hosted.
 *
 * Mount in Express:
 *   import { pushRouter, sendPushToAll } from './pushServer';
 *   app.use('/api/push', pushRouter);
 *
 * Then call sendPushToAll() when Janus sends a message.
 */

import { Router, type Request, type Response } from "express";
import crypto from "crypto";
import fs from "fs";
import path from "path";

// ── VAPID key management ──────────────────────────────────────────────────────

const VAPID_FILE = path.resolve(process.cwd(), "vapid_keys.json");

interface VapidKeys {
  publicKey:  string;
  privateKey: string;
}

function generateVapidKeys(): VapidKeys {
  // Generate ECDH key pair on P-256 curve (required for Web Push)
  const { privateKey, publicKey } = crypto.generateKeyPairSync("ec", {
    namedCurve: "prime256v1",
    publicKeyEncoding:  { type: "spki",  format: "der" },
    privateKeyEncoding: { type: "pkcs8", format: "der" },
  });

  // Convert to URL-safe base64 (uncompressed point format for public key)
  const pubKeyB64  = publicKey.subarray(27).toString("base64url");  // strip DER header
  const privKeyB64 = privateKey.subarray(36).toString("base64url"); // strip DER header

  return { publicKey: pubKeyB64, privateKey: privKeyB64 };
}

function loadOrCreateVapidKeys(): VapidKeys {
  if (fs.existsSync(VAPID_FILE)) {
    try {
      return JSON.parse(fs.readFileSync(VAPID_FILE, "utf8"));
    } catch {}
  }
  const keys = generateVapidKeys();
  fs.writeFileSync(VAPID_FILE, JSON.stringify(keys, null, 2));
  console.log("[Push] Generated new VAPID keys → vapid_keys.json");
  return keys;
}

export const vapidKeys = loadOrCreateVapidKeys();

// ── Subscription store ────────────────────────────────────────────────────────

const SUBS_FILE = path.resolve(process.cwd(), "push_subscriptions.json");

interface PushSubscription {
  endpoint: string;
  keys: { p256dh: string; auth: string };
  addedAt: string;
}

function loadSubscriptions(): PushSubscription[] {
  if (!fs.existsSync(SUBS_FILE)) return [];
  try {
    return JSON.parse(fs.readFileSync(SUBS_FILE, "utf8"));
  } catch { return []; }
}

function saveSubscriptions(subs: PushSubscription[]) {
  fs.writeFileSync(SUBS_FILE, JSON.stringify(subs, null, 2));
}

let subscriptions = loadSubscriptions();

// ── Web Push sender ───────────────────────────────────────────────────────────

/**
 * Send a Web Push notification to all subscribed devices.
 * Uses the Web Push Protocol (RFC 8030) with VAPID auth.
 */
export async function sendPushToAll(payload: {
  title: string;
  body:  string;
  type?: string;
}): Promise<void> {
  if (subscriptions.length === 0) return;

  const data    = JSON.stringify(payload);
  const failed: string[] = [];

  for (const sub of subscriptions) {
    try {
      await sendWebPush(sub, data);
    } catch (e: any) {
      // 410 Gone = subscription expired, remove it
      if (e.statusCode === 410) {
        failed.push(sub.endpoint);
      }
      console.error("[Push] Send failed:", e.message);
    }
  }

  // Remove expired subscriptions
  if (failed.length > 0) {
    subscriptions = subscriptions.filter(s => !failed.includes(s.endpoint));
    saveSubscriptions(subscriptions);
  }
}

/**
 * Low-level Web Push sender using Node.js crypto + fetch.
 * No external library needed — implements RFC 8030 + RFC 8291 (encryption).
 */
async function sendWebPush(sub: PushSubscription, payload: string): Promise<void> {
  // Build VAPID JWT
  const vapidJwt = await buildVapidJwt(sub.endpoint);

  // Encrypt payload using Web Push encryption (RFC 8291)
  const encrypted = await encryptPayload(payload, sub.keys.p256dh, sub.keys.auth);

  const res = await fetch(sub.endpoint, {
    method: "POST",
    headers: {
      "Authorization":  `vapid t=${vapidJwt.token},k=${vapidKeys.publicKey}`,
      "Content-Type":   "application/octet-stream",
      "Content-Encoding": "aes128gcm",
      "TTL":            "86400",
    },
    body: encrypted,
  });

  if (!res.ok && res.status !== 201) {
    const err: any = new Error(`Push failed: ${res.status}`);
    err.statusCode = res.status;
    throw err;
  }
}

async function buildVapidJwt(endpoint: string): Promise<{ token: string }> {
  const origin  = new URL(endpoint).origin;
  const now     = Math.floor(Date.now() / 1000);
  const header  = { alg: "ES256", typ: "JWT" };
  const payload = { aud: origin, exp: now + 3600, sub: "mailto:janus@meshchat.local" };

  const b64url = (obj: object) =>
    Buffer.from(JSON.stringify(obj)).toString("base64url");

  const unsigned = `${b64url(header)}.${b64url(payload)}`;

  // Sign with ECDSA P-256
  const privKeyDer = Buffer.concat([
    Buffer.from("308141020100301306072a8648ce3d020106082a8648ce3d030107042730250201010420", "hex"),
    Buffer.from(vapidKeys.privateKey, "base64url"),
  ]);

  const sign = crypto.createSign("SHA256");
  sign.update(unsigned);
  const derSig = sign.sign({ key: privKeyDer, format: "der", type: "pkcs8" });

  // Convert DER signature to raw r||s format
  const r = derSig.subarray(4, 4 + derSig[3]);
  const s = derSig.subarray(4 + derSig[3] + 2);
  const rawSig = Buffer.concat([
    Buffer.alloc(32 - r.length), r,
    Buffer.alloc(32 - s.length), s,
  ]);

  const token = `${unsigned}.${rawSig.toString("base64url")}`;
  return { token };
}

async function encryptPayload(
  payload: string,
  p256dhB64: string,
  authB64: string
): Promise<Buffer> {
  // RFC 8291 Web Push Message Encryption
  const p256dh = Buffer.from(p256dhB64, "base64url");
  const auth   = Buffer.from(authB64,   "base64url");

  // Generate sender key pair
  const { privateKey: senderPriv, publicKey: senderPub } =
    crypto.generateKeyPairSync("ec", {
      namedCurve: "prime256v1",
      publicKeyEncoding:  { type: "spki",  format: "der" },
      privateKeyEncoding: { type: "pkcs8", format: "der" },
    });

  // ECDH shared secret
  const receiverPubKey = crypto.createPublicKey({
    key: Buffer.concat([
      Buffer.from("3059301306072a8648ce3d020106082a8648ce3d030107034200", "hex"),
      p256dh,
    ]),
    format: "der",
    type:   "spki",
  });

  const ecdh = crypto.createECDH("prime256v1");
  ecdh.setPrivateKey(senderPriv.subarray(36), "buffer" as any);
  const sharedSecret = ecdh.computeSecret(p256dh);

  // HKDF to derive content encryption key and nonce
  const senderPubRaw = senderPub.subarray(27); // uncompressed point
  const salt         = crypto.randomBytes(16);

  const prk = crypto.createHmac("sha256", auth)
    .update(Buffer.concat([sharedSecret, senderPubRaw, p256dh]))
    .digest();

  const cek = crypto.createHmac("sha256", prk)
    .update(Buffer.concat([Buffer.from("Content-Encoding: aes128gcm\0"), salt, Buffer.from([1])]))
    .digest().subarray(0, 16);

  const nonce = crypto.createHmac("sha256", prk)
    .update(Buffer.concat([Buffer.from("Content-Encoding: nonce\0"), salt, Buffer.from([1])]))
    .digest().subarray(0, 12);

  // Encrypt
  const cipher = crypto.createCipheriv("aes-128-gcm", cek, nonce);
  const padded  = Buffer.concat([Buffer.from(payload), Buffer.from([2])]); // padding delimiter
  const enc     = Buffer.concat([cipher.update(padded), cipher.final(), cipher.getAuthTag()]);

  // Build aes128gcm content-encoding header
  const header = Buffer.concat([
    salt,
    Buffer.from([0, 0, 16, 0]),  // rs=4096, idlen=0
    Buffer.from([senderPubRaw.length]),
    senderPubRaw,
  ]);

  return Buffer.concat([header, enc]);
}

// ── Express router ────────────────────────────────────────────────────────────

export const pushRouter = Router();

function isAuthorized(req: Request): boolean {
  const secret = process.env.RELAY_SECRET ?? "";
  const ip     = req.ip ?? req.socket.remoteAddress ?? "";
  const isLocal = ip === "127.0.0.1" || ip === "::1" || ip === "::ffff:127.0.0.1";
  if (isLocal) return true;
  const auth = req.headers.authorization ?? "";
  return auth.startsWith("Bearer ") && auth.slice(7) === secret;
}

// GET /api/push/vapid-key — phone fetches this to subscribe
pushRouter.get("/vapid-key", (_req, res) => {
  res.json({ publicKey: vapidKeys.publicKey });
});

// POST /api/push/subscribe — phone registers its push subscription
pushRouter.post("/subscribe", (req: Request, res: Response) => {
  if (!isAuthorized(req)) {
    res.status(401).json({ ok: false });
    return;
  }

  const sub = req.body as PushSubscription;
  if (!sub?.endpoint || !sub?.keys?.p256dh || !sub?.keys?.auth) {
    res.status(400).json({ ok: false, error: "Invalid subscription" });
    return;
  }

  // Upsert
  const exists = subscriptions.findIndex(s => s.endpoint === sub.endpoint);
  if (exists >= 0) {
    subscriptions[exists] = { ...sub, addedAt: new Date().toISOString() };
  } else {
    subscriptions.push({ ...sub, addedAt: new Date().toISOString() });
  }
  saveSubscriptions(subscriptions);

  console.log(`[Push] Subscription registered (total: ${subscriptions.length})`);
  res.json({ ok: true });
});

// POST /api/push/send — Janus Python calls this to send a push
pushRouter.post("/send", async (req: Request, res: Response) => {
  if (!isAuthorized(req)) {
    res.status(401).json({ ok: false });
    return;
  }

  const { title = "MeshChat", body, type = "text" } = req.body ?? {};
  if (!body) {
    res.status(400).json({ ok: false, error: "body required" });
    return;
  }

  await sendPushToAll({ title, body, type });
  res.json({ ok: true, sent: subscriptions.length });
});
