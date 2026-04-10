use crate::broadphase::AabbTree;
use crate::collider::Collider;
use crate::narrowphase::{ContactManifold, NarrowPhase};
use crate::rigid_body::RigidBody;
use crate::types::{Aabb, Vec3};

const FIXED_DT: f32 = 1.0 / 60.0;
const RESTITUTION: f32 = 0.3;
const BAUMGARTE: f32 = 1.0;

// ---------------------------------------------------------------------------

pub struct WorldBoundsExceededEvent {
    pub body_index: usize,
}

#[derive(Debug)]
pub struct RaycastHit {
    pub collider_index: usize,
    pub body_index: usize,
    pub point: Vec3,
    pub t: f32,
}

// ---------------------------------------------------------------------------

pub struct PhysicsWorld {
    pub bodies: Vec<RigidBody>,
    pub colliders: Vec<Collider>,
    broadphase: AabbTree,
    #[allow(dead_code)]
    narrowphase: NarrowPhase,
    pub world_bounds: Aabb,
    pub gravity: Vec3,
    accumulator: f32,
    pub events: Vec<WorldBoundsExceededEvent>,
    trigger_manifolds: Vec<ContactManifold>,
}

impl PhysicsWorld {
    pub fn new(world_bounds: Aabb) -> Self {
        Self {
            bodies: Vec::new(),
            colliders: Vec::new(),
            broadphase: AabbTree::new(),
            narrowphase: NarrowPhase,
            world_bounds,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            accumulator: 0.0,
            events: Vec::new(),
            trigger_manifolds: Vec::new(),
        }
    }

    pub fn add_body(&mut self, body: RigidBody) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(body);
        idx
    }

    pub fn add_collider(&mut self, collider: Collider) -> usize {
        let idx = self.colliders.len();
        self.colliders.push(collider);
        idx
    }

    /// Advance the simulation by `elapsed` seconds using a fixed timestep accumulator.
    pub fn update(&mut self, elapsed: f32) {
        self.events.clear();
        self.accumulator += elapsed;
        while self.accumulator >= FIXED_DT {
            self.step();
            self.accumulator -= FIXED_DT;
        }
    }

    /// One fixed-timestep physics tick.
    fn step(&mut self) {
        // 1. Integrate all non-frozen bodies
        for body in &mut self.bodies {
            body.integrate(FIXED_DT, self.gravity);
        }

        // 2. World bounds check
        let bounds = self.world_bounds;
        for (i, body) in self.bodies.iter_mut().enumerate() {
            if body.is_frozen {
                continue;
            }
            let p = body.position;
            if p.x < bounds.min.x
                || p.x > bounds.max.x
                || p.y < bounds.min.y
                || p.y > bounds.max.y
                || p.z < bounds.min.z
                || p.z > bounds.max.z
            {
                // Clamp position to bounds
                body.position = Vec3::new(
                    p.x.max(bounds.min.x).min(bounds.max.x),
                    p.y.max(bounds.min.y).min(bounds.max.y),
                    p.z.max(bounds.min.z).min(bounds.max.z),
                );
                body.is_frozen = true;
                self.events.push(WorldBoundsExceededEvent { body_index: i });
            }
        }

        // 3. Rebuild broadphase
        self.broadphase = AabbTree::new();
        for (ci, collider) in self.colliders.iter().enumerate() {
            let body = &self.bodies[collider.body_index];
            let aabb = collider.compute_aabb(body.position, body.orientation);
            self.broadphase.insert(ci, aabb);
        }

        // 4. Narrowphase
        let pairs = self.broadphase.query_pairs();
        let mut solid_manifolds: Vec<ContactManifold> = Vec::new();
        let mut trigger_manifolds: Vec<ContactManifold> = Vec::new();

        for (a, b) in pairs {
            if let Some(manifold) = NarrowPhase::test(&self.colliders, &self.bodies, a, b) {
                if self.colliders[a].is_trigger || self.colliders[b].is_trigger {
                    trigger_manifolds.push(manifold);
                } else {
                    solid_manifolds.push(manifold);
                }
            }
        }

        self.trigger_manifolds = trigger_manifolds;

        // 5. Resolve solid collisions
        self.resolve_collisions(&solid_manifolds);
    }

    /// Sequential impulse collision resolution.
    fn resolve_collisions(&mut self, manifolds: &[ContactManifold]) {
        for m in manifolds {
            let ca = &self.colliders[m.collider_a];
            let cb = &self.colliders[m.collider_b];

            // Skip triggers (already filtered, but be safe)
            if ca.is_trigger || cb.is_trigger {
                continue;
            }

            let ia = ca.body_index;
            let ib = cb.body_index;

            let inv_mass_a = self.bodies[ia].inv_mass;
            let inv_mass_b = self.bodies[ib].inv_mass;
            let total_inv_mass = inv_mass_a + inv_mass_b;

            if total_inv_mass == 0.0 {
                continue; // both static
            }

            // Relative velocity at contact point
            let vel_a = self.bodies[ia].linear_velocity;
            let vel_b = self.bodies[ib].linear_velocity;
            let rel_vel = vel_b.sub(vel_a);
            let v_rel_n = rel_vel.dot(m.normal);

            // If separating, skip
            if v_rel_n > 0.0 {
                continue;
            }

            // Impulse magnitude
            let j = -(1.0 + RESTITUTION) * v_rel_n / total_inv_mass;

            // Apply impulse
            let impulse = m.normal.scale(j);
            if !self.bodies[ia].is_frozen {
                let new_vel = self.bodies[ia].linear_velocity.sub(impulse.scale(inv_mass_a));
                self.bodies[ia].linear_velocity = new_vel;
            }
            if !self.bodies[ib].is_frozen {
                let new_vel = self.bodies[ib].linear_velocity.add(impulse.scale(inv_mass_b));
                self.bodies[ib].linear_velocity = new_vel;
            }

            // Positional correction (Baumgarte)
            let correction = m.normal.scale(BAUMGARTE * m.penetration / total_inv_mass);
            if !self.bodies[ia].is_frozen {
                self.bodies[ia].position =
                    self.bodies[ia].position.sub(correction.scale(inv_mass_a));
            }
            if !self.bodies[ib].is_frozen {
                self.bodies[ib].position =
                    self.bodies[ib].position.add(correction.scale(inv_mass_b));
            }
        }
    }

    /// Returns the trigger manifolds detected in the last step.
    pub fn trigger_overlaps(&self) -> &[ContactManifold] {
        &self.trigger_manifolds
    }

    /// Cast a ray from `origin` in `direction` (should be normalized).
    /// Returns the closest hit with positive t.
    pub fn raycast(&self, origin: Vec3, direction: Vec3) -> Option<RaycastHit> {
        let mut best: Option<RaycastHit> = None;

        for (ci, collider) in self.colliders.iter().enumerate() {
            let body = &self.bodies[collider.body_index];
            let pos = body.position;
            let ori = body.orientation;

            let t = match &collider.shape {
                crate::collider::ColliderShape::Sphere { radius } => {
                    ray_sphere(origin, direction, pos, *radius)
                }
                crate::collider::ColliderShape::Aabb { half_extents } => {
                    let aabb = Aabb::new(pos.sub(*half_extents), pos.add(*half_extents));
                    ray_aabb(origin, direction, aabb)
                }
                _ => {
                    // Fallback: use AABB approximation
                    let aabb = collider.compute_aabb(pos, ori);
                    ray_aabb(origin, direction, aabb)
                }
            };

            if let Some(t_val) = t {
                if t_val > 0.0 {
                    let is_better = best.as_ref().map_or(true, |h| t_val < h.t);
                    if is_better {
                        best = Some(RaycastHit {
                            collider_index: ci,
                            body_index: collider.body_index,
                            point: origin.add(direction.scale(t_val)),
                            t: t_val,
                        });
                    }
                }
            }
        }

        best
    }
}

// ---------------------------------------------------------------------------
// Ray intersection helpers

/// Ray vs sphere. Returns t of first positive intersection.
fn ray_sphere(origin: Vec3, dir: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    let oc = origin.sub(center);
    let a = dir.dot(dir);
    let b = 2.0 * oc.dot(dir);
    let c = oc.dot(oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }
    let sqrt_d = discriminant.sqrt();
    let t1 = (-b - sqrt_d) / (2.0 * a);
    let t2 = (-b + sqrt_d) / (2.0 * a);
    if t1 > 0.0 {
        Some(t1)
    } else if t2 > 0.0 {
        Some(t2)
    } else {
        None
    }
}

/// Ray vs AABB (slab method). Returns t of entry if positive.
fn ray_aabb(origin: Vec3, dir: Vec3, aabb: Aabb) -> Option<f32> {
    let inv_dir = Vec3::new(
        if dir.x.abs() > 1e-10 { 1.0 / dir.x } else { f32::INFINITY },
        if dir.y.abs() > 1e-10 { 1.0 / dir.y } else { f32::INFINITY },
        if dir.z.abs() > 1e-10 { 1.0 / dir.z } else { f32::INFINITY },
    );

    let t1 = (aabb.min.x - origin.x) * inv_dir.x;
    let t2 = (aabb.max.x - origin.x) * inv_dir.x;
    let t3 = (aabb.min.y - origin.y) * inv_dir.y;
    let t4 = (aabb.max.y - origin.y) * inv_dir.y;
    let t5 = (aabb.min.z - origin.z) * inv_dir.z;
    let t6 = (aabb.max.z - origin.z) * inv_dir.z;

    let t_min = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
    let t_max = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

    if t_max < 0.0 || t_min > t_max {
        return None;
    }

    if t_min > 0.0 { Some(t_min) } else { Some(t_max) }
}

// ---------------------------------------------------------------------------
// Unit tests

#[cfg(test)]
mod tests {
    use super::*;

    // Simple LCG PRNG — no external crates
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self { Lcg(seed) }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }
        fn gen_range_f32(&mut self, lo: f32, hi: f32) -> f32 {
            let t = (self.next_u64() as f32) / (u64::MAX as f32);
            lo + t * (hi - lo)
        }
        fn gen_bool(&mut self) -> bool { self.next_u64() % 2 == 0 }
    }
    use crate::collider::{Collider, ColliderShape};
    use crate::rigid_body::RigidBody;
    use crate::types::{Aabb, Vec3};

    fn default_world() -> PhysicsWorld {
        let bounds = Aabb::new(
            Vec3::new(-1000.0, -1000.0, -1000.0),
            Vec3::new(1000.0, 1000.0, 1000.0),
        );
        PhysicsWorld::new(bounds)
    }

    // -----------------------------------------------------------------------
    // Fixed timestep: update(0.1) should step exactly 6 times (6 * 1/60 ≈ 0.1s)

    #[test]
    fn test_fixed_timestep_step_count() {
        let mut world = default_world();
        world.gravity = Vec3::zero();

        // Add a body so we can observe integration
        let body = RigidBody::new(1.0, Vec3::new(0.0, 0.0, 0.0));
        let bi = world.add_body(body);
        // Give it a known velocity
        world.bodies[bi].linear_velocity = Vec3::new(1.0, 0.0, 0.0);

        // 6 steps * FIXED_DT = 6/60 = 0.1 s
        world.update(0.1);

        // After exactly 6 steps the position should be 6 * (1/60) * 1.0 = 0.1
        let expected = 6.0 * FIXED_DT * 1.0;
        let actual = world.bodies[bi].position.x;
        assert!(
            (actual - expected).abs() < 1e-4,
            "Expected x ≈ {expected}, got {actual}"
        );

        // Accumulator should be < FIXED_DT (no partial step)
        assert!(world.accumulator < FIXED_DT);
    }

    // -----------------------------------------------------------------------
    // Collision resolution: two overlapping spheres are separated after one step

    #[test]
    fn test_sphere_collision_resolution() {
        let mut world = default_world();
        world.gravity = Vec3::zero();

        // Two spheres of radius 1.0 placed 1.5 apart (overlapping by 0.5).
        // Give them approaching velocities so the impulse fires and separates them.
        let b0 = world.add_body(RigidBody::new(1.0, Vec3::new(-0.75, 0.0, 0.0)));
        let b1 = world.add_body(RigidBody::new(1.0, Vec3::new(0.75, 0.0, 0.0)));
        // Approaching each other
        world.bodies[b0].linear_velocity = Vec3::new(10.0, 0.0, 0.0);
        world.bodies[b1].linear_velocity = Vec3::new(-10.0, 0.0, 0.0);

        world.add_collider(Collider {
            shape: ColliderShape::Sphere { radius: 1.0 },
            is_trigger: false,
            body_index: b0,
        });
        world.add_collider(Collider {
            shape: ColliderShape::Sphere { radius: 1.0 },
            is_trigger: false,
            body_index: b1,
        });

        world.update(FIXED_DT);

        let p0 = world.bodies[b0].position;
        let p1 = world.bodies[b1].position;
        let dist = p1.sub(p0).length();
        assert!(
            dist >= 2.0 - 1e-3,
            "Spheres still overlapping after step: dist = {dist}"
        );
    }

    // -----------------------------------------------------------------------
    // World bounds: body outside bounds is frozen and event emitted

    #[test]
    fn test_world_bounds_event() {
        let bounds = Aabb::new(
            Vec3::new(-10.0, -10.0, -10.0),
            Vec3::new(10.0, 10.0, 10.0),
        );
        let mut world = PhysicsWorld::new(bounds);
        world.gravity = Vec3::zero();

        // Place body just inside, with velocity that will push it outside in one step
        let mut body = RigidBody::new(1.0, Vec3::new(9.9, 0.0, 0.0));
        body.linear_velocity = Vec3::new(100.0, 0.0, 0.0); // will exceed bounds
        let bi = world.add_body(body);

        world.update(FIXED_DT);

        assert!(world.bodies[bi].is_frozen, "Body should be frozen after exceeding bounds");
        assert!(
            !world.events.is_empty(),
            "WorldBoundsExceededEvent should have been emitted"
        );
        assert_eq!(world.events[0].body_index, bi);
    }

    // -----------------------------------------------------------------------
    // Trigger: trigger overlap detected but no impulse applied

    #[test]
    fn test_trigger_no_impulse() {
        let mut world = default_world();
        world.gravity = Vec3::zero();

        let b0 = world.add_body(RigidBody::new(1.0, Vec3::new(-0.5, 0.0, 0.0)));
        let b1 = world.add_body(RigidBody::new(1.0, Vec3::new(0.5, 0.0, 0.0)));

        // One of the colliders is a trigger
        world.add_collider(Collider {
            shape: ColliderShape::Sphere { radius: 1.0 },
            is_trigger: true,
            body_index: b0,
        });
        world.add_collider(Collider {
            shape: ColliderShape::Sphere { radius: 1.0 },
            is_trigger: false,
            body_index: b1,
        });

        // Record initial velocities (zero)
        world.update(FIXED_DT);

        // Trigger overlap should be detected
        assert!(
            !world.trigger_overlaps().is_empty(),
            "Trigger overlap should be detected"
        );

        // Velocities should remain zero (no impulse applied)
        let v0 = world.bodies[b0].linear_velocity;
        let v1 = world.bodies[b1].linear_velocity;
        assert!(
            v0.length() < 1e-5,
            "Trigger body should have no impulse applied: v0 = {:?}", v0
        );
        assert!(
            v1.length() < 1e-5,
            "Other body should have no impulse applied: v1 = {:?}", v1
        );
    }

    // Property 15: Raycast returns correct first intersection
    // Validates: Requirements 4.7
    #[test]
    fn property_raycast_sphere_aabb() {
        let mut rng = Lcg::new(0xdeadbeef_cafef00d);

        for _ in 0..1000 { // Run 1000 iterations with random inputs
            let mut world = default_world();

            // Generate a random sphere
            let sphere_pos = Vec3::new(rng.gen_range_f32(-10.0, 10.0), rng.gen_range_f32(-10.0, 10.0), rng.gen_range_f32(-10.0, 10.0));
            let sphere_radius = rng.gen_range_f32(0.1, 5.0);
            let sphere_bi = world.add_body(RigidBody::static_body(sphere_pos));
            world.add_collider(Collider {
                shape: ColliderShape::Sphere { radius: sphere_radius },
                is_trigger: false,
                body_index: sphere_bi,
            });

            // Generate a random AABB
            let aabb_pos = Vec3::new(rng.gen_range_f32(-10.0, 10.0), rng.gen_range_f32(-10.0, 10.0), rng.gen_range_f32(-10.0, 10.0));
            let half_extents = Vec3::new(rng.gen_range_f32(0.1, 5.0), rng.gen_range_f32(0.1, 5.0), rng.gen_range_f32(0.1, 5.0));
            let aabb_bi = world.add_body(RigidBody::static_body(aabb_pos));
            world.add_collider(Collider {
                shape: ColliderShape::Aabb { half_extents },
                is_trigger: false,
                body_index: aabb_bi,
            });

            // Generate a random ray
            let origin = Vec3::new(rng.gen_range_f32(-20.0, 20.0), rng.gen_range_f32(-20.0, 20.0), rng.gen_range_f32(-20.0, 20.0));
            let direction = Vec3::new(rng.gen_range_f32(-1.0, 1.0), rng.gen_range_f32(-1.0, 1.0), rng.gen_range_f32(-1.0, 1.0)).normalize();

            // Test raycast against both colliders
            let hit = world.raycast(origin, direction);

            // Manually calculate expected hits for sphere and AABB
            let sphere_t = ray_sphere(origin, direction, sphere_pos, sphere_radius);
            let aabb_t = ray_aabb(origin, direction, Aabb::new(aabb_pos.sub(half_extents), aabb_pos.add(half_extents)));

            let expected_t = match (sphere_t, aabb_t) {
                (Some(st), Some(at)) => Some(st.min(at)),
                (Some(st), None) => Some(st),
                (None, Some(at)) => Some(at),
                (None, None) => None,
            };

            match (hit, expected_t) {
                (Some(h), Some(et)) => {
                    // Ensure the hit 't' value is close to the expected 't' value
                    assert!((h.t - et).abs() < 1e-4,
                        "Raycast 't' value mismatch. Expected {}, got {}. Origin: {:?}, Dir: {:?}",
                        et, h.t, origin, direction);
                    // Ensure the hit point is on the ray at 't'
                    let expected_point = origin.add(direction.scale(et));
                    assert!((h.point.sub(expected_point)).length() < 1e-4,
                        "Raycast hit point mismatch. Expected {:?}, got {:?}. Origin: {:?}, Dir: {:?}",
                        expected_point, h.point, origin, direction);
                },
                (None, None) => {
                    // Both correctly reported no hit
                },
                (Some(h), None) => {
                    panic!("Raycast unexpectedly hit. Hit: {:?}. Origin: {:?}, Dir: {:?}", h, origin, direction);
                },
                (None, Some(et)) => {
                    panic!("Raycast unexpectedly missed. Expected hit at {}. Origin: {:?}, Dir: {:?}", et, origin, direction);
                },
            }
        }
    }

    // Property 14: Physics Fixed Timestep Invariant
    // Validates: Requirements 4.4
    #[test]
    fn property_physics_fixed_timestep_invariant() {
        let mut rng = Lcg::new(0xdeadbeef_cafef00d);

        for _ in 0..100 { // 100 iterations with random inputs
            let mut world = default_world();
            world.gravity = Vec3::zero();

            // Add a test body with random initial position and velocity
            let initial_pos = Vec3::new(rng.gen_range_f32(-10.0, 10.0), 0.0, 0.0);
            let velocity = Vec3::new(rng.gen_range_f32(1.0, 10.0), 0.0, 0.0);
            let bi = world.add_body(RigidBody::new(1.0, initial_pos));
            world.bodies[bi].linear_velocity = velocity;

            // Generate a random elapsed time between 0.0 and 1.0 seconds
            let elapsed = rng.gen_range_f32(0.0, 1.0);

            // Simulate
            world.update(elapsed);

            // Calculate expected number of full steps
            let num_steps = (elapsed / FIXED_DT).floor() as u32;
            let expected_advance = num_steps as f32 * FIXED_DT * velocity.x;
            let expected_pos = initial_pos.x + expected_advance;

            // Check position advanced correctly
            let actual_pos = world.bodies[bi].position.x;
            assert!((actual_pos - expected_pos).abs() < 1e-4,
                "Position mismatch after {} steps. Expected {}, got {}. Elapsed: {}",
                num_steps, expected_pos, actual_pos, elapsed);

            // Check accumulator holds the remainder (less than FIXED_DT)
            let expected_accum = elapsed - num_steps as f32 * FIXED_DT;
            assert!((world.accumulator - expected_accum).abs() < 1e-4,
                "Accumulator mismatch. Expected {}, got {}. Elapsed: {}",
                expected_accum, world.accumulator, elapsed);
            assert!(world.accumulator < FIXED_DT,
                "Accumulator should be < FIXED_DT, got {}", world.accumulator);

            // Edge case: zero elapsed should do nothing
            if elapsed == 0.0 {
                assert_eq!(world.accumulator, 0.0);
                assert_eq!(actual_pos, initial_pos.x);
            }
        }
    }

    // Property 13: Collision Resolution Separates Overlapping Bodies
    // Validates: Requirements 4.3
    #[test]
    fn property_collision_resolution_separates_overlapping_bodies() {
        let mut rng = Lcg::new(0xdeadbeef_cafef00d);

        for _ in 0..100 { // 100 iterations with random configurations
            let mut world = default_world();
            world.gravity = Vec3::zero();

            // Create two bodies with some overlap
            let radius_a = rng.gen_range_f32(0.5, 2.0);
            let radius_b = rng.gen_range_f32(0.5, 2.0);
            let overlap = rng.gen_range_f32(0.1, 0.5);

            let pos_a = Vec3::new(rng.gen_range_f32(-1.0, 1.0), 0.0, 0.0);
            let pos_b = pos_a.add(Vec3::new(radius_a + radius_b - overlap, 0.0, 0.0));

            let mass_a = rng.gen_range_f32(1.0, 5.0);
            let mass_b = rng.gen_range_f32(1.0, 5.0);

            let mut body_a = RigidBody::new(mass_a, pos_a);
            let mut body_b = RigidBody::new(mass_b, pos_b);

            // Give them approaching velocities to ensure collision
            body_a.linear_velocity = Vec3::new(rng.gen_range_f32(1.0, 5.0), 0.0, 0.0);
            body_b.linear_velocity = Vec3::new(rng.gen_range_f32(-5.0, -1.0), 0.0, 0.0);

            let bi_a = world.add_body(body_a);
            let bi_b = world.add_body(body_b);

            world.add_collider(Collider {
                shape: ColliderShape::Sphere { radius: radius_a },
                is_trigger: false,
                body_index: bi_a,
            });
            world.add_collider(Collider {
                shape: ColliderShape::Sphere { radius: radius_b },
                is_trigger: false,
                body_index: bi_b,
            });

            // Simulate one step
            world.step();

            // Check that bodies are no longer overlapping (or are moving apart)
            let p_a = world.bodies[bi_a].position;
            let p_b = world.bodies[bi_b].position;
            let dist = p_b.sub(p_a).length();

            let v_a = world.bodies[bi_a].linear_velocity;
            let v_b = world.bodies[bi_b].linear_velocity;
            let rel_vel = v_b.sub(v_a);
            let normal = p_b.sub(p_a).normalize();
            let separating_vel = rel_vel.dot(normal);

            assert!(dist >= radius_a + radius_b - 1e-3 || separating_vel > 0.0,
                "Collision resolution failed. Dist: {}, Vel: {}, Overlap: {}", dist, separating_vel, overlap);
        }
    }
}
