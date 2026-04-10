/**
 * CallMeButton.tsx
 * ================
 * Dashboard button that triggers Janus to call your phone.
 * Appears in the ISP dashboard header/sidebar.
 *
 * Calls POST /api/voip/call with the message to speak.
 */

import { useState } from "react";
import { Button } from "./button";
import { Phone, PhoneCall, PhoneOff, Loader2 } from "lucide-react";

type CallStatus = "idle" | "calling" | "success" | "error";

interface CallMeButtonProps {
  /** Custom message for Janus to speak. Defaults to a status report. */
  message?: string;
  /** Show as icon-only (compact mode) */
  compact?: boolean;
}

export function CallMeButton({ message, compact = false }: CallMeButtonProps) {
  const [status, setStatus]   = useState<CallStatus>("idle");
  const [error,  setError]    = useState<string>("");
  const [custom, setCustom]   = useState("");
  const [showInput, setShowInput] = useState(false);

  const handleCall = async (msg?: string) => {
    setStatus("calling");
    setError("");

    const finalMessage = msg || message || custom ||
      "Janus status report. All systems are running. Check the dashboard for details.";

    try {
      const res = await fetch("/api/voip/call", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ message: finalMessage }),
      });

      const data = await res.json();

      if (data.ok) {
        setStatus("success");
        setTimeout(() => setStatus("idle"), 4000);
      } else {
        setStatus("error");
        setError(data.error ?? "Call failed");
        setTimeout(() => setStatus("idle"), 5000);
      }
    } catch (err) {
      setStatus("error");
      setError("Could not reach dashboard server");
      setTimeout(() => setStatus("idle"), 5000);
    }
  };

  const icon = status === "calling" ? <Loader2 className="h-4 w-4 animate-spin" /> :
               status === "success" ? <PhoneCall className="h-4 w-4" /> :
               status === "error"   ? <PhoneOff  className="h-4 w-4" /> :
                                      <Phone     className="h-4 w-4" />;

  const label = status === "calling" ? "Calling..." :
                status === "success" ? "Call initiated!" :
                status === "error"   ? "Call failed" :
                                       "Call me";

  const variant = status === "success" ? "default" :
                  status === "error"   ? "destructive" :
                                         "outline";

  if (compact) {
    return (
      <Button
        variant={variant}
        size="icon"
        onClick={() => handleCall()}
        disabled={status === "calling"}
        title={label}
      >
        {icon}
      </Button>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex gap-2">
        <Button
          variant={variant}
          onClick={() => handleCall()}
          disabled={status === "calling"}
          className="gap-2"
        >
          {icon}
          {label}
        </Button>

        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowInput(!showInput)}
          disabled={status === "calling"}
        >
          Custom message
        </Button>
      </div>

      {showInput && (
        <div className="flex gap-2">
          <input
            type="text"
            value={custom}
            onChange={e => setCustom(e.target.value)}
            placeholder="What should Janus say?"
            className="flex-1 rounded-md border border-input bg-background px-3 py-1 text-sm"
            onKeyDown={e => e.key === "Enter" && handleCall(custom)}
          />
          <Button
            size="sm"
            onClick={() => handleCall(custom)}
            disabled={!custom.trim() || status === "calling"}
          >
            Call
          </Button>
        </div>
      )}

      {status === "error" && error && (
        <p className="text-xs text-destructive">{error}</p>
      )}

      {status === "success" && (
        <p className="text-xs text-muted-foreground">
          Your phone should ring in a few seconds.
        </p>
      )}
    </div>
  );
}
