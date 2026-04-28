// engine-runtime/src/overlay.rs
//
// Transparent always-on-top window that covers the full primary monitor.
// Arania renders inside this window; the desktop shows through behind her.
//
// Dependencies (added to Cargo.toml):
//   winit  = "0.29"
//   wgpu   = "0.19"
//   pixels = "0.13"    (wgpu surface helper)

use std::sync::{Arc, Mutex};

/// Configuration for the overlay window.
#[derive(Debug, Clone)]
pub struct OverlayConfig {
    /// Window title (visible in taskbar).
    pub title: String,
    /// Whether the window accepts mouse/keyboard input.
    /// Set false so clicks pass through to the desktop.
    pub click_through: bool,
    /// Whether to stay on top of all other windows.
    pub always_on_top: bool,
    /// Target frame rate.
    pub target_fps: u32,
    /// Width and height in logical pixels (None = fullscreen primary monitor).
    pub size: Option<(u32, u32)>,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            title:        "Janus — Arania".into(),
            click_through: true,
            always_on_top: true,
            target_fps:   60,
            size:         None, // fullscreen
        }
    }
}

/// Shared state written by the overlay and read by the render loop.
#[derive(Debug, Default)]
pub struct OverlayState {
    pub width:  u32,
    pub height: u32,
    pub should_close: bool,
}

/// OverlayWindow drives the winit event loop and owns the wgpu surface.
/// In the current stub this provides the interface that engine-runtime's
/// main loop calls; the real winit integration is wired in lib.rs.
pub struct OverlayWindow {
    pub config: OverlayConfig,
    pub state:  Arc<Mutex<OverlayState>>,
}

impl OverlayWindow {
    pub fn new(config: OverlayConfig) -> Self {
        let state = Arc::new(Mutex::new(OverlayState {
            width:        config.size.map(|(w, _)| w).unwrap_or(1920),
            height:       config.size.map(|(_, h)| h).unwrap_or(1080),
            should_close: false,
        }));
        Self { config, state }
    }

    /// Returns the current logical size of the overlay.
    pub fn size(&self) -> (u32, u32) {
        let s = self.state.lock().unwrap();
        (s.width, s.height)
    }

    /// Called by the event loop when the window is resized.
    pub fn on_resize(&self, w: u32, h: u32) {
        let mut s = self.state.lock().unwrap();
        s.width  = w;
        s.height = h;
        println!("[overlay] Resized → {}×{}", w, h);
    }

    /// Request graceful shutdown.
    pub fn request_close(&self) {
        self.state.lock().unwrap().should_close = true;
    }

    pub fn should_close(&self) -> bool {
        self.state.lock().unwrap().should_close
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_default_size() {
        let w = OverlayWindow::new(OverlayConfig::default());
        assert_eq!(w.size(), (1920, 1080));
    }

    #[test]
    fn overlay_custom_size() {
        let mut cfg = OverlayConfig::default();
        cfg.size = Some((1280, 720));
        let w = OverlayWindow::new(cfg);
        assert_eq!(w.size(), (1280, 720));
    }

    #[test]
    fn overlay_resize() {
        let w = OverlayWindow::new(OverlayConfig::default());
        w.on_resize(2560, 1440);
        assert_eq!(w.size(), (2560, 1440));
    }

    #[test]
    fn overlay_close() {
        let w = OverlayWindow::new(OverlayConfig::default());
        assert!(!w.should_close());
        w.request_close();
        assert!(w.should_close());
    }
}
