use crate::types::{CaptureConfig, CaptureStreamId, CaptureStrategy, CAPTURE_CAP_ZERO_COPY, CAPTURE_CAP_STAGING_BUFFER};
use crate::backend::GfxBackend;

pub struct CaptureStream {
    pub id: CaptureStreamId,
    pub config: CaptureConfig,
}

pub struct UniversalCaptureEngine {
    pub streams: Vec<CaptureStream>,
}

impl UniversalCaptureEngine {
    pub fn new() -> Self {
        Self {
            streams: Vec::new(),
        }
    }

    pub fn add_stream(&mut self, backend: &mut dyn GfxBackend, mut config: CaptureConfig) -> CaptureStreamId {
        let caps = backend.get_capture_capabilities();
        
        // Negotiate strategy based on hardware capabilities
        if (caps & CAPTURE_CAP_ZERO_COPY) != 0 && (config.strategy == CaptureStrategy::ZeroCopy || config.strategy == CaptureStrategy::None) {
            config.strategy = CaptureStrategy::ZeroCopy;
        } else if (caps & CAPTURE_CAP_STAGING_BUFFER) != 0 {
            config.strategy = CaptureStrategy::StagingBuffer;
        } else {
            config.strategy = CaptureStrategy::None;
        }

        let id = backend.register_capture_stream(config);
        self.streams.push(CaptureStream { id, config });
        id
    }

    pub fn remove_stream(&mut self, backend: &mut dyn GfxBackend, id: CaptureStreamId) {
        backend.unregister_capture_stream(id);
        self.streams.retain(|s| s.id != id);
    }

    pub fn dispatch_captures(&self, backend: &mut dyn GfxBackend) {
        for stream in &self.streams {
            backend.dispatch_capture(stream.id);
        }
    }
}
