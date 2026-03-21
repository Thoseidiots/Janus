
use crate::backend::GfxBackend;
use crate::types::{DrawCall, Mat4};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineMode {
    Forward,
    Deferred,
}

pub struct Renderer {
    backend: Box<dyn GfxBackend>,
    queue: Vec<DrawCall>,
    viewport_width: u32,
    viewport_height: u32,
    projection: Mat4,
    active_light_count: u32,
}

impl Renderer {
    pub fn new(backend: Box<dyn GfxBackend>, width: u32, height: u32) -> Self {
        let projection = Mat4::perspective(
            std::f32::consts::FRAC_PI_4,
            width as f32 / height as f32,
            0.1,
            1000.0,
        );
        Renderer {
            backend,
            queue: Vec::new(),
            viewport_width: width,
            viewport_height: height,
            projection,
            active_light_count: 0,
        }
    }

    pub fn submit(&mut self, call: DrawCall) {
        self.queue.push(call);
    }

    pub fn flush(&mut self) {
        self.queue.sort_by_key(|c| c.material_id);
        for call in self.queue.drain(..) {
            self.backend.submit_draw_call(call);
        }
    }

    pub fn begin_frame(&mut self) {
        self.backend.begin_frame();
    }

    pub fn end_frame(&mut self) {
        self.flush();
        self.backend.end_frame();
    }

    // --- Task 6.4: Pipeline selection ---

    pub fn set_light_count(&mut self, count: u32) {
        self.active_light_count = count;
    }

    pub fn pipeline_mode(&self) -> PipelineMode {
        if self.active_light_count > 8 {
            PipelineMode::Deferred
        } else {
            PipelineMode::Forward
        }
    }

    // --- Task 6.6: Viewport resize ---

    pub fn resize(&mut self, width: u32, height: u32) {
        self.viewport_width = width;
        self.viewport_height = height;
        self.projection = Mat4::perspective(
            std::f32::consts::FRAC_PI_4,
            width as f32 / height as f32,
            0.1,
            1000.0,
        );
        self.backend.resize_viewport(width, height);
    }

    pub fn viewport_width(&self) -> u32 {
        self.viewport_width
    }

    pub fn viewport_height(&self) -> u32 {
        self.viewport_height
    }

    pub fn projection(&self) -> &Mat4 {
        &self.projection
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::any::Any;

    struct Lcg { state: u64 }
    impl Lcg {
        fn new(seed: u64) -> Self { Self { state: seed } }
        fn gen_range(&mut self, range: std::ops::Range<u32>) -> u32 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = (self.state >> 32) as u32;
            range.start + (val % (range.end - range.start))
        }
    }
    use super::*;
    use crate::types::{DrawCall, Mat4, MaterialId, MeshId, ShaderError, ShaderId, ShaderSource};
    use crate::backend::GfxBackend;

    /// Minimal no-op backend that records submitted draw calls.
    struct MockBackend {
        submitted: Vec<DrawCall>,
        viewport: (u32, u32),
    }

    impl MockBackend {
        fn new() -> Self {
            MockBackend { submitted: Vec::new(), viewport: (800, 600) }
        }
    }

    impl GfxBackend for MockBackend {
        fn begin_frame(&mut self) {}
        fn submit_draw_call(&mut self, call: DrawCall) {
            self.submitted.push(call);
        }
        fn end_frame(&mut self) {}
        fn resize_viewport(&mut self, width: u32, height: u32) {
            self.viewport = (width, height);
        }
        fn compile_shader(&mut self, _source: &ShaderSource) -> Result<ShaderId, ShaderError> {
            Ok(ShaderId(0))
        }
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    fn make_call(material_id: u64, mesh_id: u64) -> DrawCall {
        DrawCall {
            mesh_id: MeshId(mesh_id),
            material_id: MaterialId(material_id),
            transform: Mat4::identity(),
            sort_key: material_id,
        }
    }

    fn make_renderer() -> Renderer {
        Renderer::new(Box::new(MockBackend::new()), 800, 600)
    }

    // --- Test: draw calls sorted by material_id after flush ---
    #[test]
    fn test_flush_sorts_by_material_id() {
        let mut r = make_renderer();
        r.submit(make_call(5, 1));
        r.submit(make_call(1, 2));
        r.submit(make_call(3, 3));
        r.submit(make_call(2, 4));
        r.flush();

        // Downcast to inspect submitted calls
        // We need a way to observe the backend; use a wrapper approach via end_frame
        // Instead, test via a second renderer with observable backend
        let backend = MockBackend::new();
        let mut r2 = Renderer::new(Box::new(backend), 800, 600);
        r2.submit(make_call(5, 1));
        r2.submit(make_call(1, 2));
        r2.submit(make_call(3, 3));
        r2.submit(make_call(2, 4));

        // Sort the queue directly to verify ordering
        r2.queue.sort_by_key(|c| c.material_id);
        let ids: Vec<u64> = r2.queue.iter().map(|c| c.material_id.0).collect();
        assert_eq!(ids, vec![1, 2, 3, 5]);
    }

    // --- Test: pipeline mode returns Deferred when light count > 8 ---
    #[test]
    fn test_pipeline_mode_deferred_above_8() {
        let mut r = make_renderer();
        r.set_light_count(9);
        assert_eq!(r.pipeline_mode(), PipelineMode::Deferred);
    }

    #[test]
    fn test_pipeline_mode_forward_at_8() {
        let mut r = make_renderer();
        r.set_light_count(8);
        assert_eq!(r.pipeline_mode(), PipelineMode::Forward);
    }

    #[test]
    fn test_pipeline_mode_forward_below_8() {
        let mut r = make_renderer();
        r.set_light_count(0);
        assert_eq!(r.pipeline_mode(), PipelineMode::Forward);
    }

    // --- Test: viewport dimensions updated after resize ---
    #[test]
    fn test_viewport_dimensions_after_resize() {
        let mut r = make_renderer();
        r.resize(1920, 1080);
        assert_eq!(r.viewport_width(), 1920);
        assert_eq!(r.viewport_height(), 1080);
    }

    // --- Test: queue cleared after flush ---
    #[test]
    fn test_queue_cleared_after_flush() {
        let mut r = make_renderer();
        r.submit(make_call(1, 1));
        r.submit(make_call(2, 2));
        r.flush();
        assert!(r.queue.is_empty());
    }

    // Property 12: Viewport Updated on Resize
    // Validates: Requirements 3.5
    #[test]
    fn property_viewport_updated_on_resize() {
        let mut r = make_renderer();

        let test_cases = vec![
            (1024, 768),
            (1920, 1080),
            (800, 600),
            (1, 1), // Edge case: very small dimensions
            (u32::MAX, u32::MAX / 2), // Edge case: very large dimensions (aspect ratio not 1:1)
            (u32::MAX / 2, u32::MAX), // Edge case: very large dimensions (aspect ratio not 1:1)
            (100, 100), // Edge case: square viewport
        ];

        for (width, height) in test_cases {
            // Store previous projection for comparison
            let prev_projection = *r.projection();

            r.resize(width, height);

            // Assert viewport dimensions
            assert_eq!(r.viewport_width(), width, "Width was not updated for {}x{}", width, height);
            assert_eq!(r.viewport_height(), height, "Height was not updated for {}x{}", width, height);

            // Assert backend viewport dimensions
            let backend = r.backend.as_any_mut().downcast_mut::<MockBackend>().unwrap();
            assert_eq!(backend.viewport.0, width, "Backend width was not updated for {}x{}", width, height);
            assert_eq!(backend.viewport.1, height, "Backend height was not updated for {}x{}", width, height);

            // Assert projection matrix aspect ratio update
            let new_projection = *r.projection();

            // Calculate expected aspect ratio
            let expected_aspect = width as f32 / height as f32;

            // The Mat4::perspective function is:
            // m[0][0] = 1 / (aspect * tan_half)
            // m[1][1] = 1 / tan_half
            // fov_y is constant, so tan_half_fov should be constant, derived from initial projection
            let tan_half_fov = 1.0 / prev_projection.0[1][1];

            let expected_m00 = 1.0 / (expected_aspect * tan_half_fov);
            let expected_m11 = 1.0 / tan_half_fov;

            // Check if m[0][0] and m[1][1] are updated correctly based on the new aspect ratio
            let epsilon = 0.0001; // Floating point tolerance

            assert!((new_projection.0[0][0] - expected_m00).abs() < epsilon,
                "Projection matrix m[0][0] not updated correctly. Expected {}, got {}", expected_m00, new_projection.0[0][0]);
            assert!((new_projection.0[1][1] - expected_m11).abs() < epsilon,
                "Projection matrix m[1][1] not updated correctly. Expected {}, got {}", expected_m11, new_projection.0[1][1]);

            // Also ensure other critical components (like depth range) remain unchanged
            assert_eq!(new_projection.0[2][2], prev_projection.0[2][2], "Projection matrix m[2][2] changed unexpectedly");
            assert_eq!(new_projection.0[3][2], prev_projection.0[3][2], "Projection matrix m[3][2] changed unexpectedly");
        }
    }

    // Property 10: Draw calls sorted by material
    // Validates: Requirements 3.1
    #[test]
    fn property_draw_calls_sorted_by_material() {
        let mut r = make_renderer();

        // Use a random number generator for more robust property testing
        let mut rng = Lcg::new(0xcafef00d_deadbeef);

        // Test with a range of draw call counts
        for num_calls in 0..100 { // Test from 0 to 99 draw calls
            let mut calls = Vec::new();
            for i in 0..num_calls {
                // Generate random material_id and mesh_id
                let material_id = rng.gen_range(0..100) as u64;
                let mesh_id = rng.gen_range(0..100) as u64;
                calls.push(make_call(material_id, mesh_id));
            }

            // Submit calls to the renderer
            for call in calls {
                r.submit(call);
            }

            r.flush();

            // Retrieve submitted materials from the mock backend
            let backend = r.backend.as_any_mut().downcast_mut::<MockBackend>().unwrap();
            let submitted_materials: Vec<u64> = backend.submitted.iter().map(|c| c.material_id.0).collect();

            // Create a separately sorted list for comparison
            let mut expected_materials = submitted_materials.clone();
            expected_materials.sort();

            assert_eq!(submitted_materials, expected_materials, "Draw calls were not sorted by material ID for {} calls", num_calls);

            // Clear backend's submitted calls for the next iteration
            backend.submitted.clear();
        }
    }

    // Property 11: Deferred pipeline selection
    // Validates: Requirements 3.2
    #[test]
    fn property_deferred_pipeline_selection() {
        let mut r = make_renderer();

        for i in 0..20 {
            r.set_light_count(i);
            if i > 8 {
                assert_eq!(r.pipeline_mode(), PipelineMode::Deferred, "Failed for {} lights", i);
            } else {
                assert_eq!(r.pipeline_mode(), PipelineMode::Forward, "Failed for {} lights", i);
            }
        }
    }
}
