use crate::types::{Aabb, Quat, Vec3};

pub enum ColliderShape {
    Sphere { radius: f32 },
    Aabb { half_extents: Vec3 },
    Obb { half_extents: Vec3, orientation: Quat },
    Capsule { radius: f32, half_height: f32 },
    ConvexHull { vertices: Vec<Vec3> },
}

pub struct Collider {
    pub shape: ColliderShape,
    pub is_trigger: bool,
    pub body_index: usize,
}

impl Collider {
    /// Compute the world-space AABB for this collider given a body position and orientation.
    pub fn compute_aabb(&self, position: Vec3, orientation: Quat) -> Aabb {
        match &self.shape {
            ColliderShape::Sphere { radius } => {
                let r = Vec3::new(*radius, *radius, *radius);
                Aabb::new(position.sub(r), position.add(r))
            }
            ColliderShape::Aabb { half_extents } => {
                Aabb::new(position.sub(*half_extents), position.add(*half_extents))
            }
            ColliderShape::Obb { half_extents, orientation: local_orient } => {
                // Combine body orientation with local OBB orientation
                let combined = orientation.mul(*local_orient).normalize();
                obb_world_aabb(position, *half_extents, combined)
            }
            ColliderShape::Capsule { radius, half_height } => {
                // Axis-aligned capsule along Y by default
                let up = orientation.rotate_vec(Vec3::new(0.0, *half_height, 0.0));
                let top = position.add(up);
                let bot = position.sub(up);
                let r = Vec3::new(*radius, *radius, *radius);
                let min = Vec3::new(
                    top.x.min(bot.x) - radius,
                    top.y.min(bot.y) - radius,
                    top.z.min(bot.z) - radius,
                );
                let max = Vec3::new(
                    top.x.max(bot.x) + radius,
                    top.y.max(bot.y) + radius,
                    top.z.max(bot.z) + radius,
                );
                let _ = r; // suppress unused warning
                Aabb::new(min, max)
            }
            ColliderShape::ConvexHull { vertices } => {
                if vertices.is_empty() {
                    return Aabb::new(position, position);
                }
                let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
                let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
                for &v in vertices {
                    let world = position.add(orientation.rotate_vec(v));
                    min.x = min.x.min(world.x);
                    min.y = min.y.min(world.y);
                    min.z = min.z.min(world.z);
                    max.x = max.x.max(world.x);
                    max.y = max.y.max(world.y);
                    max.z = max.z.max(world.z);
                }
                Aabb::new(min, max)
            }
        }
    }
}

/// Compute the world AABB of an OBB defined by center, half-extents, and orientation.
fn obb_world_aabb(center: Vec3, half_extents: Vec3, orientation: Quat) -> Aabb {
    // Project each local axis onto world axes
    let ax = orientation.rotate_vec(Vec3::new(half_extents.x, 0.0, 0.0));
    let ay = orientation.rotate_vec(Vec3::new(0.0, half_extents.y, 0.0));
    let az = orientation.rotate_vec(Vec3::new(0.0, 0.0, half_extents.z));

    let hx = ax.x.abs() + ay.x.abs() + az.x.abs();
    let hy = ax.y.abs() + ay.y.abs() + az.y.abs();
    let hz = ax.z.abs() + ay.z.abs() + az.z.abs();

    let h = Vec3::new(hx, hy, hz);
    Aabb::new(center.sub(h), center.add(h))
}
