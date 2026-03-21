#[cfg(target_os = "windows")]
pub use self::inner::Dx12Backend;

#[cfg(target_os = "windows")]
mod inner {
    use crate::backend::GfxBackend;
    use crate::types::{DrawCall, ShaderError, ShaderId, ShaderSource};

    pub struct Dx12Backend;

    impl Dx12Backend {
        pub fn new() -> Self {
            Dx12Backend
        }
    }

    impl GfxBackend for Dx12Backend {
        fn as_any(&self) -> &dyn std::any::Any { self }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
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

// Provide a dummy type on non-Windows so the crate compiles everywhere.
#[cfg(not(target_os = "windows"))]
pub struct Dx12Backend;
