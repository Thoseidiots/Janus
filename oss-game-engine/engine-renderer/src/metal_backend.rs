#[cfg(target_os = "macos")]
pub use self::inner::MetalBackend;

#[cfg(target_os = "macos")]
mod inner {
    use crate::backend::GfxBackend;
    use crate::types::{DrawCall, ShaderError, ShaderId, ShaderSource};

    pub struct MetalBackend;

    impl MetalBackend {
        pub fn new() -> Self {
            MetalBackend
        }
    }

    impl GfxBackend for MetalBackend {
        fn begin_frame(&mut self) {}
        fn submit_draw_call(&mut self, _call: DrawCall) {}
        fn end_frame(&mut self) {}
        fn resize_viewport(&mut self, _width: u32, _height: u32) {}
        fn compile_shader(&mut self, source: &ShaderSource) -> Result<ShaderId, ShaderError> {
            let _ = source;
            Ok(ShaderId(0))
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }
}

// Provide a dummy type on non-macOS so the crate compiles everywhere.
#[cfg(not(target_os = "macos"))]
pub struct MetalBackend;
