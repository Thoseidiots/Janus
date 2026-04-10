/**
 * meshchat-sw.js
 * ==============
 * MeshChat Service Worker
 * Handles push notifications and offline caching for iPhone/iOS.
 *
 * iOS 16.4+ supports Web Push when the PWA is added to home screen.
 * This service worker receives push events even when the app is closed.
 */

const CACHE_NAME = "meshchat-v1";
const OFFLINE_URLS = ["/meshchat", "/meshchat/", "/favicon.ico"];

// ── Install: cache shell ──────────────────────────────────────────────────────

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(OFFLINE_URLS))
  );
  self.skipWaiting();
});

// ── Activate: clean old caches ────────────────────────────────────────────────

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

// ── Fetch: serve from cache when offline ─────────────────────────────────────

self.addEventListener("fetch", (event) => {
  // Only cache GET requests for our app shell
  if (event.request.method !== "GET") return;
  if (!event.request.url.includes("/meshchat")) return;

  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});

// ── Push: show notification when app is closed ───────────────────────────────

self.addEventListener("push", (event) => {
  let data = { title: "MeshChat", body: "New message", type: "text" };

  try {
    data = event.data?.json() ?? data;
  } catch {
    data.body = event.data?.text() ?? "New message";
  }

  const isAlert = data.type === "alert";

  const options = {
    body:    data.body,
    icon:    "/meshchat-icon-192.png",
    badge:   "/meshchat-badge-72.png",
    tag:     isAlert ? "meshchat-alert" : "meshchat-message",
    renotify: true,
    vibrate: isAlert ? [200, 100, 200, 100, 200] : [100],
    data:    { url: "/meshchat", type: data.type },
    actions: [
      { action: "open",    title: "Open MeshChat" },
      { action: "dismiss", title: "Dismiss" },
    ],
    // iOS-specific: show immediately, don't batch
    requireInteraction: isAlert,
  };

  event.waitUntil(
    self.registration.showNotification(data.title ?? "MeshChat", options)
  );
});

// ── Notification click: open app ──────────────────────────────────────────────

self.addEventListener("notificationclick", (event) => {
  event.notification.close();

  if (event.action === "dismiss") return;

  event.waitUntil(
    clients
      .matchAll({ type: "window", includeUncontrolled: true })
      .then((clientList) => {
        // Focus existing window if open
        for (const client of clientList) {
          if (client.url.includes("/meshchat") && "focus" in client) {
            return client.focus();
          }
        }
        // Otherwise open new window
        return clients.openWindow("/meshchat");
      })
  );
});
