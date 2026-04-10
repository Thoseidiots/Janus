pub mod types;
pub mod backend;
pub mod renderer;
pub mod shaders;
pub mod vulkan_backend;
pub mod metal_backend;
pub mod dx12_backend;
pub mod webgpu_backend;

pub use types::{MeshId, MaterialId, ShaderId, ShaderSource, DrawCall, Mat4, ShaderError, RenderError};
pub use backend::GfxBackend;
pub use renderer::{Renderer, PipelineMode};
pub use vulkan_backend::VulkanBackend;
pub use shaders::{PBR_VERT_GLSL, PBR_FRAG_GLSL, PBR_WGSL, PBR_HLSL, PBR_MSL, FALLBACK_FRAG_GLSL};
