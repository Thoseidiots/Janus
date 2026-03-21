#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MaterialId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderId(pub u64);

pub struct ShaderSource {
    pub name: String,
    pub glsl: Option<String>,
    pub wgsl: Option<String>,
    pub hlsl: Option<String>,
    pub msl: Option<String>,
}

/// A 4x4 column-major matrix (no external math dep).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4(pub [[f32; 4]; 4]);

impl Mat4 {
    pub fn identity() -> Self {
        Mat4([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Standard perspective projection matrix (column-major, right-handed, depth [-1, 1]).
    pub fn perspective(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Self {
        let tan_half = (fov_y_rad / 2.0).tan();
        let mut m = [[0.0f32; 4]; 4];
        m[0][0] = 1.0 / (aspect * tan_half);
        m[1][1] = 1.0 / tan_half;
        m[2][2] = -(far + near) / (far - near);
        m[2][3] = -1.0;
        m[3][2] = -(2.0 * far * near) / (far - near);
        Mat4(m)
    }
}

#[derive(Debug, Clone)]
pub struct DrawCall {
    pub mesh_id: MeshId,
    pub material_id: MaterialId,
    pub transform: Mat4,
    pub sort_key: u64, // packed: material_id | depth
}

#[derive(Debug)]
pub enum ShaderError {
    CompileError { path: String, line: u32, message: String },
}

#[derive(Debug)]
pub enum RenderError {
    BackendLost,
}
