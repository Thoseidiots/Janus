use std::collections::HashMap;
use crate::backend::GfxBackend;
use crate::types::{DrawCall, ShaderError, ShaderId, ShaderSource};

pub struct VulkanBackend {
    compiled_shaders: HashMap<ShaderId, String>,
    next_shader_id: u64,
    viewport: (u32, u32),
    frame_draw_calls: Vec<DrawCall>,
    pub fallback_shader_id: Option<ShaderId>,
}

impl VulkanBackend {
    pub fn new(width: u32, height: u32) -> Self {
        VulkanBackend {
            compiled_shaders: HashMap::new(),
            next_shader_id: 1,
            viewport: (width, height),
            frame_draw_calls: Vec::new(),
            fallback_shader_id: None,
        }
    }
}

impl GfxBackend for VulkanBackend {
    fn begin_frame(&mut self) {
        self.frame_draw_calls.clear();
    }

    fn submit_draw_call(&mut self, call: DrawCall) {
        self.frame_draw_calls.push(call);
    }

    fn end_frame(&mut self) {
        // Stub: no actual GPU submission
    }

    fn resize_viewport(&mut self, width: u32, height: u32) {
        self.viewport = (width, height);
    }

    fn compile_shader(&mut self, source: &ShaderSource) -> Result<ShaderId, ShaderError> {
        match &source.glsl {
            None => {
                let err = ShaderError::CompileError {
                    path: "unknown".to_string(),
                    line: 0,
                    message: "no GLSL source provided".to_string(),
                };
                eprintln!(
                    "[VulkanBackend] shader compile error for '{}': no GLSL source provided",
                    source.name
                );
                Err(err)
            }
            Some(_glsl) => {
                let id = ShaderId(self.next_shader_id);
                self.next_shader_id += 1;
                self.compiled_shaders.insert(id, source.name.clone());

                // Track the first successful compile of the fallback error shader
                if source.name == "fallback_error" && self.fallback_shader_id.is_none() {
                    self.fallback_shader_id = Some(id);
                }

                Ok(id)
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
