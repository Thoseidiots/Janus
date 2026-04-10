use crate::types::{Mat3, Quat, Vec3};

pub struct RigidBody {
    pub position: Vec3,
    pub orientation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub mass: f32,
    pub inv_mass: f32,
    pub inertia_tensor: Mat3,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub is_frozen: bool,
}

impl RigidBody {
    /// Create a dynamic body with the given mass at the given position.
    pub fn new(mass: f32, position: Vec3) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            position,
            orientation: Quat::identity(),
            linear_velocity: Vec3::zero(),
            angular_velocity: Vec3::zero(),
            mass,
            inv_mass,
            inertia_tensor: Mat3::identity(),
            linear_damping: 0.0,
            angular_damping: 0.0,
            is_frozen: false,
        }
    }

    /// Create a static (immovable) body.
    pub fn static_body(position: Vec3) -> Self {
        Self {
            position,
            orientation: Quat::identity(),
            linear_velocity: Vec3::zero(),
            angular_velocity: Vec3::zero(),
            mass: 0.0,
            inv_mass: 0.0,
            inertia_tensor: Mat3::identity(),
            linear_damping: 0.0,
            angular_damping: 0.0,
            is_frozen: true,
        }
    }

    /// Apply an impulse at a contact point (world space).
    pub fn apply_impulse(&mut self, impulse: Vec3, contact_point: Vec3) {
        if self.is_frozen || self.inv_mass == 0.0 {
            return;
        }
        // Linear impulse
        self.linear_velocity = self.linear_velocity.add(impulse.scale(self.inv_mass));
        // Angular impulse: τ = r × J, then ω += I^-1 * τ
        let r = contact_point.sub(self.position);
        let torque = r.cross(impulse);
        let inv_inertia = self.inertia_tensor.inverse();
        let delta_omega = inv_inertia.mul_vec(torque);
        self.angular_velocity = self.angular_velocity.add(delta_omega);
    }

    /// Semi-implicit Euler integration for one timestep.
    pub fn integrate(&mut self, dt: f32, gravity: Vec3) {
        if self.is_frozen || self.inv_mass == 0.0 {
            return;
        }

        // Apply gravity to velocity first (semi-implicit)
        self.linear_velocity = self.linear_velocity.add(gravity.scale(dt));

        // Apply damping
        let lin_damp = (1.0 - self.linear_damping * dt).max(0.0);
        let ang_damp = (1.0 - self.angular_damping * dt).max(0.0);
        self.linear_velocity = self.linear_velocity.scale(lin_damp);
        self.angular_velocity = self.angular_velocity.scale(ang_damp);

        // Integrate position
        self.position = self.position.add(self.linear_velocity.scale(dt));

        // Integrate orientation using angular velocity
        let half_dt = 0.5 * dt;
        let omega = self.angular_velocity;
        let dq = Quat {
            x: omega.x * half_dt,
            y: omega.y * half_dt,
            z: omega.z * half_dt,
            w: 0.0,
        };
        // q' = q + dq * q  (first-order integration)
        let q = self.orientation;
        self.orientation = Quat {
            x: q.x + dq.x * q.w + dq.y * q.z - dq.z * q.y,
            y: q.y - dq.x * q.z + dq.y * q.w + dq.z * q.x,
            z: q.z + dq.x * q.y - dq.y * q.x + dq.z * q.w,
            w: q.w - dq.x * q.x - dq.y * q.y - dq.z * q.z,
        }
        .normalize();
    }
}
