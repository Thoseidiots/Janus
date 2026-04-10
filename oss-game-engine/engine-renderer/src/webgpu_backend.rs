#[cfg(target_arch = "wasm32")]
pub use self::inner::WebGpuBackend;

#[cfg(target_arch = "wasm32")]
mod inner {
    use crate::backend::GfxBackend;
    use crate::types::{DrawCall, ShaderError, ShaderId, ShaderSource};

    pub struct WebGpuBackend;

    impl WebGpuBackend {
        pub fn new() -> Self {
            WebGpuBackend
        }
    }

    impl GfxBackend for WebGpuBackend {
        fn begin_frame(&mut self) {}
        fn submit_draw_call(&mut self, _call: DrawCall) {}
        fn end_frame(&mut self) {}
        fn resize_viewport(&mut self, _width: u32, _height: u32) {}
        fn compile_shader(&mut self, source: &ShaderSource) -> Result<ShaderId, ShaderError> {
            let _ = source;
            Ok(ShaderId(0))
        }
    }
}

// Provide a dummy type on non-wasm32 so the crate compiles everywhere.
#[cfg(not(target_arch = "wasm32"))]
pub struct WebGpuBackend;
