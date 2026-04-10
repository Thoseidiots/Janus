use crate::collider::{Collider, ColliderShape};
use crate::rigid_body::RigidBody;
use crate::types::{Aabb, Vec3};

pub struct ContactManifold {
    pub collider_a: usize,
    pub collider_b: usize,
    /// Collision normal pointing from A toward B.
    pub normal: Vec3,
    /// Positive value = penetration depth.
    pub penetration: f32,
    pub contact_point: Vec3,
}

pub struct NarrowPhase;

impl NarrowPhase {
    /// Test a pair of colliders. Returns `Some(manifold)` if they overlap.
    pub fn test(
        colliders: &[Collider],
        bodies: &[RigidBody],
        a: usize,
        b: usize,
    ) -> Option<ContactManifold> {
        let ca = &colliders[a];
        let cb = &colliders[b];
        let ba = &bodies[ca.body_index];
        let bb = &bodies[cb.body_index];

        let pos_a = ba.position;
        let pos_b = bb.position;
        let ori_a = ba.orientation;
        let ori_b = bb.orientation;

        match (&ca.shape, &cb.shape) {
            (ColliderShape::Sphere { radius: ra }, ColliderShape::Sphere { radius: rb }) => {
                sphere_sphere(a, b, pos_a, *ra, pos_b, *rb)
            }
            (ColliderShape::Aabb { half_extents: ha }, ColliderShape::Aabb { half_extents: hb }) => {
                aabb_aabb(a, b, pos_a, *ha, pos_b, *hb)
            }
            (ColliderShape::Sphere { radius: rs }, ColliderShape::Aabb { half_extents: hb }) => {
                sphere_aabb(a, b, pos_a, *rs, pos_b, *hb)
            }
            (ColliderShape::Aabb { half_extents: ha }, ColliderShape::Sphere { radius: rs }) => {
                // Swap and flip normal
                sphere_aabb(b, a, pos_b, *rs, pos_a, *ha).map(|mut m| {
                    m.collider_a = a;
                    m.collider_b = b;
                    m.normal = m.normal.neg();
                    m
                })
            }
            _ => {
                // Conservative fallback: AABB overlap test
                let aabb_a = ca.compute_aabb(pos_a, ori_a);
                let aabb_b = cb.compute_aabb(pos_b, ori_b);
                aabb_conservative(a, b, aabb_a, aabb_b)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sphere vs Sphere

fn sphere_sphere(
    a: usize,
    b: usize,
    pos_a: Vec3,
    ra: f32,
    pos_b: Vec3,
    rb: f32,
) -> Option<ContactManifold> {
    let delta = pos_b.sub(pos_a);
    let dist_sq = delta.dot(delta);
    let sum_r = ra + rb;
    if dist_sq >= sum_r * sum_r {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = if dist < 1e-10 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        delta.scale(1.0 / dist)
    };
    let penetration = sum_r - dist;
    let contact_point = pos_a.add(normal.scale(ra - penetration * 0.5));
    Some(ContactManifold { collider_a: a, collider_b: b, normal, penetration, contact_point })
}

// ---------------------------------------------------------------------------
// AABB vs AABB (SAT on 3 axes)

fn aabb_aabb(
    a: usize,
    b: usize,
    pos_a: Vec3,
    ha: Vec3,
    pos_b: Vec3,
    hb: Vec3,
) -> Option<ContactManifold> {
    let delta = pos_b.sub(pos_a);
    let overlap_x = (ha.x + hb.x) - delta.x.abs();
    let overlap_y = (ha.y + hb.y) - delta.y.abs();
    let overlap_z = (ha.z + hb.z) - delta.z.abs();

    if overlap_x <= 0.0 || overlap_y <= 0.0 || overlap_z <= 0.0 {
        return None;
    }

    // Choose the axis of minimum penetration
    let (penetration, normal) = if overlap_x <= overlap_y && overlap_x <= overlap_z {
        let sign = if delta.x >= 0.0 { 1.0 } else { -1.0 };
        (overlap_x, Vec3::new(sign, 0.0, 0.0))
    } else if overlap_y <= overlap_z {
        let sign = if delta.y >= 0.0 { 1.0 } else { -1.0 };
        (overlap_y, Vec3::new(0.0, sign, 0.0))
    } else {
        let sign = if delta.z >= 0.0 { 1.0 } else { -1.0 };
        (overlap_z, Vec3::new(0.0, 0.0, sign))
    };

    let contact_point = pos_a.add(pos_b).scale(0.5);
    Some(ContactManifold { collider_a: a, collider_b: b, normal, penetration, contact_point })
}

// ---------------------------------------------------------------------------
// Sphere vs AABB (closest-point test)

fn sphere_aabb(
    sphere_idx: usize,
    aabb_idx: usize,
    sphere_pos: Vec3,
    radius: f32,
    aabb_pos: Vec3,
    half_extents: Vec3,
) -> Option<ContactManifold> {
    // Closest point on AABB to sphere center
    let aabb_min = aabb_pos.sub(half_extents);
    let aabb_max = aabb_pos.add(half_extents);

    let closest = Vec3::new(
        sphere_pos.x.max(aabb_min.x).min(aabb_max.x),
        sphere_pos.y.max(aabb_min.y).min(aabb_max.y),
        sphere_pos.z.max(aabb_min.z).min(aabb_max.z),
    );

    let delta = sphere_pos.sub(closest);
    let dist_sq = delta.dot(delta);

    if dist_sq >= radius * radius {
        return None;
    }

    let dist = dist_sq.sqrt();
    let normal = if dist < 1e-10 {
        // Sphere center is inside AABB — push out along shortest axis
        let d = sphere_pos.sub(aabb_pos);
        let ox = half_extents.x - d.x.abs();
        let oy = half_extents.y - d.y.abs();
        let oz = half_extents.z - d.z.abs();
        if ox <= oy && ox <= oz {
            Vec3::new(if d.x >= 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0)
        } else if oy <= oz {
            Vec3::new(0.0, if d.y >= 0.0 { 1.0 } else { -1.0 }, 0.0)
        } else {
            Vec3::new(0.0, 0.0, if d.z >= 0.0 { 1.0 } else { -1.0 })
        }
    } else {
        delta.scale(1.0 / dist)
    };

    let penetration = radius - dist;
    Some(ContactManifold {
        collider_a: sphere_idx,
        collider_b: aabb_idx,
        normal,
        penetration,
        contact_point: closest,
    })
}

// ---------------------------------------------------------------------------
// Conservative fallback: AABB overlap → separation axis manifold

fn aabb_conservative(
    a: usize,
    b: usize,
    aabb_a: Aabb,
    aabb_b: Aabb,
) -> Option<ContactManifold> {
    if !aabb_a.intersects(aabb_b) {
        return None;
    }

    let delta = aabb_b.center().sub(aabb_a.center());
    let ha = aabb_a.half_extents();
    let hb = aabb_b.half_extents();

    let overlap_x = (ha.x + hb.x) - delta.x.abs();
    let overlap_y = (ha.y + hb.y) - delta.y.abs();
    let overlap_z = (ha.z + hb.z) - delta.z.abs();

    let (penetration, normal) = if overlap_x <= overlap_y && overlap_x <= overlap_z {
        let sign = if delta.x >= 0.0 { 1.0 } else { -1.0 };
        (overlap_x, Vec3::new(sign, 0.0, 0.0))
    } else if overlap_y <= overlap_z {
        let sign = if delta.y >= 0.0 { 1.0 } else { -1.0 };
        (overlap_y, Vec3::new(0.0, sign, 0.0))
    } else {
        let sign = if delta.z >= 0.0 { 1.0 } else { -1.0 };
        (overlap_z, Vec3::new(0.0, 0.0, sign))
    };

    let contact_point = aabb_a.center().add(aabb_b.center()).scale(0.5);
    Some(ContactManifold { collider_a: a, collider_b: b, normal, penetration, contact_point })
}
