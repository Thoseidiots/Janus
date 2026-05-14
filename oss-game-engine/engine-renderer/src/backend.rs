use crate::types::{DrawCall, ShaderSource, ShaderId, ShaderError, CaptureStreamId, CaptureConfig};

use std::any::Any;

pub trait GfxBackend: Any {
    fn begin_frame(&mut self);
    fn submit_draw_call(&mut self, call: DrawCall);
    fn end_frame(&mut self);
    fn resize_viewport(&mut self, width: u32, height: u32);
    fn compile_shader(&mut self, source: &ShaderSource) -> Result<ShaderId, ShaderError>;
    
    fn get_capture_capabilities(&self) -> u32;
    fn register_capture_stream(&mut self, config: CaptureConfig) -> CaptureStreamId;
    fn unregister_capture_stream(&mut self, stream: CaptureStreamId);
    fn dispatch_capture(&mut self, stream: CaptureStreamId);

    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
