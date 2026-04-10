/**
 * janusNotify.ts
 * ==============
 * Notification service for the MeshISP dashboard.
 * Lets Janus text you while you're out of town — no API keys.
 *
 * How it works:
 *   Every US carrier has a free email-to-SMS gateway.
 *   Janus sends an email to <your_number>@<carrier_gateway>
 *   The carrier converts it to an SMS and delivers it to your phone.
 *
 * Carrier gateways (free, no signup):
 *   AT&T:      number@txt.att.net
 *   T-Mobile:  number@tmomail.net
 *   Verizon:   number@vtext.com
 *   Sprint:    number@messaging.sprintpcs.com
 *   US Cellular: number@email.uscc.net
 *   Cricket:   number@sms.cricketwireless.net
 *   Boost:     number@sms.myboostmobile.com
 *   Metro PCS: number@mymetropcs.com
 *
 * Setup:
 *   1. Add to your .env.local:
 *      NOTIFY_PHONE=10digitnumber
 *      NOTIFY_CARRIER=att   (or tmobile, verizon, etc.)
 *      SMTP_HOST=smtp.gmail.com
 *      SMTP_PORT=587
 *      SMTP_USER=your@gmail.com
 *      SMTP_PASS=your_app_password
 *
 *   2. Janus calls notifyOwner("message") from Python via the REST endpoint
 *      or directly from the tRPC router.
 */

import nodemailer from "nodemailer";
import { z } from "zod";
import { router, protectedProcedure, publicProcedure } from "./notification";

// ── Carrier gateway map ───────────────────────────────────────────────────────

const CARRIER_GATEWAYS: Record<string, string> = {
  att:       "txt.att.net",
  tmobile:   "tmomail.net",
  verizon:   "vtext.com",
  sprint:    "messaging.sprintpcs.com",
  uscellular:"email.uscc.net",
  cricket:   "sms.cricketwireless.net",
  boost:     "sms.myboostmobile.com",
  metro:     "mymetropcs.com",
  metropcs:  "mymetropcs.com",
  straighttalk: "vtext.com",
};

// ── Config from environment ───────────────────────────────────────────────────

function getNotifyConfig() {
  const phone   = process.env.NOTIFY_PHONE?.replace(/\D/g, "") ?? "";
  const carrier = (process.env.NOTIFY_CARRIER ?? "att").toLowerCase();
  const gateway = CARRIER_GATEWAYS[carrier] ?? "txt.att.net";
  const smsEmail = phone ? `${phone}@${gateway}` : null;

  return {
    smsEmail,
    phone,
    carrier,
    smtp: {
      host: process.env.SMTP_HOST ?? "smtp.gmail.com",
      port: parseInt(process.env.SMTP_PORT ?? "587"),
      user: process.env.SMTP_USER ?? "",
      pass: process.env.SMTP_PASS ?? "",
    },
  };
}

// ── Mailer ────────────────────────────────────────────────────────────────────

function createTransport() {
  const { smtp } = getNotifyConfig();
  return nodemailer.createTransport({
    host:   smtp.host,
    port:   smtp.port,
    secure: smtp.port === 465,
    auth: {
      user: smtp.user,
      pass: smtp.pass,
    },
  });
}

// ── Core send function ────────────────────────────────────────────────────────

export async function sendSMS(message: string): Promise<{ ok: boolean; error?: string }> {
  const config = getNotifyConfig();

  if (!config.smsEmail) {
    console.warn("[Notify] NOTIFY_PHONE not configured — SMS skipped");
    return { ok: false, error: "NOTIFY_PHONE not configured" };
  }

  if (!config.smtp.user || !config.smtp.pass) {
    console.warn("[Notify] SMTP credentials not configured — SMS skipped");
    return { ok: false, error: "SMTP credentials not configured" };
  }

  // SMS messages must be short — carriers truncate at 160 chars
  const truncated = message.slice(0, 155);

  try {
    const transport = createTransport();
    await transport.sendMail({
      from:    config.smtp.user,
      to:      config.smsEmail,
      subject: "",          // carriers ignore subject for SMS
      text:    truncated,
    });
    console.log(`[Notify] SMS sent to ${config.smsEmail}: ${truncated}`);
    return { ok: true };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[Notify] SMS failed: ${msg}`);
    return { ok: false, error: msg };
  }
}

export async function sendEmail(
  subject: string,
  body: string,
  to?: string
): Promise<{ ok: boolean; error?: string }> {
  const config = getNotifyConfig();
  const recipient = to ?? config.smtp.user;

  if (!config.smtp.user || !config.smtp.pass) {
    return { ok: false, error: "SMTP credentials not configured" };
  }

  try {
    const transport = createTransport();
    await transport.sendMail({
      from:    config.smtp.user,
      to:      recipient,
      subject,
      text:    body,
    });
    console.log(`[Notify] Email sent to ${recipient}: ${subject}`);
    return { ok: true };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[Notify] Email failed: ${msg}`);
    return { ok: false, error: msg };
  }
}

// ── Notification types ────────────────────────────────────────────────────────

export type NotifyLevel = "info" | "warning" | "alert";

export interface NotifyOptions {
  level?:    NotifyLevel;
  sms?:      boolean;   // also send SMS (default: true for alert, false for info)
  email?:    boolean;   // also send email (default: true)
}

/**
 * Main notification function — call this from anywhere in the dashboard.
 * Janus calls the /api/notify REST endpoint which calls this.
 */
export async function notifyOwner(
  message: string,
  opts: NotifyOptions = {}
): Promise<void> {
  const level   = opts.level   ?? "info";
  const doSMS   = opts.sms   ?? (level === "alert" || level === "warning");
  const doEmail = opts.email ?? true;

  const prefix = level === "alert" ? "🚨 JANUS ALERT" :
                 level === "warning" ? "⚠ Janus" : "Janus";
  const fullMsg = `${prefix}: ${message}`;

  const tasks: Promise<unknown>[] = [];

  if (doSMS) {
    tasks.push(sendSMS(fullMsg));
  }

  if (doEmail) {
    tasks.push(sendEmail(
      `[${prefix}] ${message.slice(0, 60)}`,
      `${fullMsg}\n\nSent at: ${new Date().toISOString()}`
    ));
  }

  await Promise.allSettled(tasks);
}

// ── tRPC router ───────────────────────────────────────────────────────────────

export const notifyRouter = router({
  /**
   * Send a notification (SMS + email).
   * Called by Janus Python bridge via HTTP POST.
   * Also callable from the dashboard UI.
   */
  send: publicProcedure
    .input(z.object({
      message: z.string().min(1).max(500),
      level:   z.enum(["info", "warning", "alert"]).default("info"),
      sms:     z.boolean().optional(),
      email:   z.boolean().optional(),
    }))
    .mutation(async ({ input }) => {
      await notifyOwner(input.message, {
        level: input.level,
        sms:   input.sms,
        email: input.email,
      });
      return { ok: true };
    }),

  /**
   * Test the notification setup.
   */
  test: protectedProcedure
    .mutation(async () => {
      const smsResult   = await sendSMS("Janus test message — SMS is working!");
      const emailResult = await sendEmail(
        "Janus Notification Test",
        "If you received this, email notifications are working."
      );
      return { sms: smsResult, email: emailResult };
    }),

  /**
   * Get current notification config (without secrets).
   */
  config: protectedProcedure
    .query(() => {
      const cfg = getNotifyConfig();
      return {
        smsConfigured:   !!cfg.smsEmail,
        emailConfigured: !!(cfg.smtp.user && cfg.smtp.pass),
        carrier:         cfg.carrier,
        gateway:         cfg.smsEmail ?? "not configured",
        smtpHost:        cfg.smtp.host,
      };
    }),
});
