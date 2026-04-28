// engine-runtime/src/arania_controller.rs
//
// Character AI controller for Arania's roaming behaviour.
//
// Responsibilities:
//   - Waypoint navigation along screen edges (walking + idle at corners)
//   - Expression / animation state machine (idle, walk, talk, think)
//   - Receives commands from the Janus Python brain over stdin (JSON lines)
//
// Command protocol (JSON lines on stdin):
//   {"cmd":"expr",   "value":"smile"}           — set facial expression
//   {"cmd":"status", "value":"Training..."}     — update status text
//   {"cmd":"talk",   "phonemes":["A","B","M"]}  — lip-sync phoneme sequence
//   {"cmd":"stop"}                              — halt movement, stand idle
//   {"cmd":"walk"}                              — resume roaming

use std::collections::VecDeque;
use std::time::Duration;

// ── Waypoints ────────────────────────────────────────────────────────────────

/// A point in world space that Arania can navigate to.
#[derive(Debug, Clone, Copy)]
pub struct Waypoint {
    pub x: f32,
    pub z: f32,
}

impl Waypoint {
    pub fn new(x: f32, z: f32) -> Self {
        Self { x, z }
    }

    pub fn distance_to(&self, other: &Waypoint) -> f32 {
        let dx = self.x - other.x;
        let dz = self.z - other.z;
        (dx * dx + dz * dz).sqrt()
    }
}

/// Default screen-edge waypoints for a 16:9 world space (−8..8, −4.5..4.5).
pub fn default_screen_waypoints() -> Vec<Waypoint> {
    vec![
        Waypoint::new(-7.0,  0.0),
        Waypoint::new(-7.0,  3.8),
        Waypoint::new( 0.0,  3.8),
        Waypoint::new( 7.0,  3.8),
        Waypoint::new( 7.0,  0.0),
        Waypoint::new( 7.0, -3.8),
        Waypoint::new( 0.0, -3.8),
        Waypoint::new(-7.0, -3.8),
    ]
}

// ── Animation state ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum AnimState {
    Idle,
    Walk,
    Talk,
    Think,
    React,
}

impl std::fmt::Display for AnimState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle  => write!(f, "idle"),
            Self::Walk  => write!(f, "walk"),
            Self::Talk  => write!(f, "talk"),
            Self::Think => write!(f, "think"),
            Self::React => write!(f, "react"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Neutral,
    Smile,
    Thinking,
    Surprised,
}

impl Expression {
    pub fn from_str(s: &str) -> Self {
        match s {
            "smile"     => Self::Smile,
            "thinking"  => Self::Thinking,
            "surprised" => Self::Surprised,
            _           => Self::Neutral,
        }
    }
    /// Morph target weight [0, 1] for the smile blend shape.
    pub fn smile_weight(&self) -> f32 {
        match self {
            Self::Smile     => 1.0,
            Self::Surprised => 0.3,
            _               => 0.0,
        }
    }
    /// Head tilt X for thinking pose.
    pub fn head_tilt_x(&self) -> f32 {
        match self {
            Self::Thinking => -0.14,
            _              =>  0.0,
        }
    }
}

// ── Command protocol ─────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum AraniaCommand {
    SetExpression(Expression),
    SetStatus(String),
    Talk(Vec<String>),
    Stop,
    Walk,
}

// ── Controller ───────────────────────────────────────────────────────────────

pub struct AraniaController {
    // Navigation
    pub waypoints:    Vec<Waypoint>,
    current_wp:       usize,
    pub position:     Waypoint,
    pub facing_angle: f32,    // radians, Y-up
    walk_speed:       f32,    // world units / second
    is_roaming:       bool,

    // Animation & expression
    pub anim_state:   AnimState,
    pub expression:   Expression,
    pub status_text:  String,

    // Phoneme queue for lip sync
    phoneme_queue:    VecDeque<String>,
    phoneme_timer:    f32,

    // Idle timers
    idle_timer:       f32,
    blink_timer:      f32,
    pub blink_weight: f32,   // 0=open 1=closed
    blink_phase:      u8,    // 0=open 1=closing 2=opening

    // Idle sway
    time_acc:         f32,
    pub head_sway_x:  f32,
    pub head_sway_z:  f32,
    pub breathe_y:    f32,
}

impl AraniaController {
    pub fn new(waypoints: Vec<Waypoint>) -> Self {
        let start = waypoints.first().copied().unwrap_or(Waypoint::new(0.0, 0.0));
        Self {
            waypoints,
            current_wp:    0,
            position:      start,
            facing_angle:  0.0,
            walk_speed:    1.4,
            is_roaming:    true,
            anim_state:    AnimState::Idle,
            expression:    Expression::Neutral,
            status_text:   "Janus is ready.".into(),
            phoneme_queue: VecDeque::new(),
            phoneme_timer: 0.0,
            idle_timer:    0.0,
            blink_timer:   4.0,
            blink_weight:  0.0,
            blink_phase:   0,
            time_acc:      0.0,
            head_sway_x:   0.0,
            head_sway_z:   0.0,
            breathe_y:     0.0,
        }
    }

    /// Process an incoming command from the Python brain.
    pub fn apply_command(&mut self, cmd: AraniaCommand) {
        match cmd {
            AraniaCommand::SetExpression(e) => {
                self.expression  = e;
            }
            AraniaCommand::SetStatus(s) => {
                self.status_text = s;
            }
            AraniaCommand::Talk(phonemes) => {
                self.phoneme_queue.extend(phonemes);
                self.anim_state = AnimState::Talk;
            }
            AraniaCommand::Stop => {
                self.is_roaming  = false;
                self.anim_state  = AnimState::Idle;
            }
            AraniaCommand::Walk => {
                self.is_roaming  = true;
                self.anim_state  = AnimState::Walk;
            }
        }
    }

    /// Tick the controller by `dt` seconds. Call once per frame.
    pub fn update(&mut self, dt: f32) {
        self.time_acc += dt;

        // Roaming navigation
        if self.is_roaming {
            self.update_navigation(dt);
        }

        // Idle animations (always active)
        self.update_idle(dt);

        // Blink
        self.update_blink(dt);

        // Phoneme/lip sync
        self.update_lip_sync(dt);
    }

    fn update_navigation(&mut self, dt: f32) {
        if self.waypoints.is_empty() {
            return;
        }
        let target = self.waypoints[self.current_wp % self.waypoints.len()];
        let dx = target.x - self.position.x;
        let dz = target.z - self.position.z;
        let dist = (dx * dx + dz * dz).sqrt();

        if dist < 0.05 {
            // Reached waypoint — pause briefly then advance
            self.idle_timer += dt;
            self.anim_state = AnimState::Idle;
            if self.idle_timer > 1.5 {
                self.idle_timer  = 0.0;
                self.current_wp  = (self.current_wp + 1) % self.waypoints.len();
                self.anim_state  = AnimState::Walk;
            }
        } else {
            self.anim_state = AnimState::Walk;
            let step = self.walk_speed * dt;
            let nx = dx / dist * step;
            let nz = dz / dist * step;
            self.position.x += nx;
            self.position.z += nz;
            self.facing_angle = dz.atan2(dx);
        }
    }

    fn update_idle(&mut self, _dt: f32) {
        let t = self.time_acc;
        self.head_sway_x = f32::sin(t * 0.41) * 0.012 + f32::sin(t * 0.73) * 0.006;
        self.head_sway_z = f32::sin(t * 0.33) * 0.010;
        self.breathe_y   = f32::sin(t * 1.10) * 0.004;
    }

    fn update_blink(&mut self, dt: f32) {
        self.blink_timer -= dt;
        if self.blink_timer <= 0.0 && self.blink_phase == 0 {
            self.blink_phase = 1;
            self.blink_timer = 3.5 + pseudo_rand(self.time_acc) * 2.5;
        }
        let speed = 8.0_f32;
        match self.blink_phase {
            1 => {
                self.blink_weight = (self.blink_weight + dt * speed).min(1.0);
                if self.blink_weight >= 1.0 { self.blink_phase = 2; }
            }
            2 => {
                self.blink_weight = (self.blink_weight - dt * speed * 0.7).max(0.0);
                if self.blink_weight <= 0.0 { self.blink_phase = 0; }
            }
            _ => {}
        }
    }

    fn update_lip_sync(&mut self, dt: f32) {
        if self.phoneme_queue.is_empty() {
            if self.anim_state == AnimState::Talk {
                self.anim_state = AnimState::Idle;
            }
            return;
        }
        self.phoneme_timer -= dt;
        if self.phoneme_timer <= 0.0 {
            self.phoneme_queue.pop_front();
            self.phoneme_timer = 0.08; // ~12 phonemes/sec
        }
    }

    /// Current phoneme being spoken (None if silent).
    pub fn current_phoneme(&self) -> Option<&str> {
        self.phoneme_queue.front().map(String::as_str)
    }

    /// Lip open amount [0,1] driven by phoneme.
    pub fn lip_open(&self) -> f32 {
        match self.current_phoneme() {
            Some("A") | Some("E") | Some("O") => 0.85,
            Some("I") | Some("U")             => 0.45,
            Some("M") | Some("B") | Some("P") => 0.02,
            Some(_)                            => 0.3,
            None => {
                // Idle mouth: very subtle breathing movement
                (self.time_acc * 1.1).sin().abs() * 0.04
            }
        }
    }

    /// Combined head rotation X (sway + expression tilt).
    pub fn head_rot_x(&self) -> f32 {
        self.head_sway_x + self.expression.head_tilt_x()
    }
}

/// Deterministic pseudo-random [0,1) from a float seed.
fn pseudo_rand(seed: f32) -> f32 {
    let x = seed.sin() * 43758.5453;
    x - x.floor()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ctrl() -> AraniaController {
        AraniaController::new(default_screen_waypoints())
    }

    #[test]
    fn initial_state() {
        let c = ctrl();
        assert_eq!(c.anim_state, AnimState::Idle);
        assert_eq!(c.expression, Expression::Neutral);
        assert!(!c.waypoints.is_empty());
    }

    #[test]
    fn walk_updates_position() {
        let mut c = ctrl();
        c.is_roaming = true;
        let start_x = c.position.x;
        for _ in 0..60 {
            c.update(1.0 / 60.0);
        }
        // After 1 second of walking she should have moved
        let moved = (c.position.x - start_x).abs() > 0.01
                 || c.position.z.abs() > 0.01;
        assert!(moved, "Character did not move after 1s");
    }

    #[test]
    fn set_expression_smile() {
        let mut c = ctrl();
        c.apply_command(AraniaCommand::SetExpression(Expression::Smile));
        assert_eq!(c.expression.smile_weight(), 1.0);
    }

    #[test]
    fn stop_command_halts_roaming() {
        let mut c = ctrl();
        c.apply_command(AraniaCommand::Stop);
        assert!(!c.is_roaming);
        assert_eq!(c.anim_state, AnimState::Idle);
    }

    #[test]
    fn phoneme_queues_talk() {
        let mut c = ctrl();
        c.apply_command(AraniaCommand::Talk(vec!["A".into(), "M".into(), "O".into()]));
        assert_eq!(c.anim_state, AnimState::Talk);
        assert!(c.lip_open() > 0.0);
    }

    #[test]
    fn blink_cycles() {
        let mut c = ctrl();
        // Force a blink
        c.blink_timer = -1.0;
        c.update(0.016); // one frame
        assert!(c.blink_phase == 1 || c.blink_phase == 0);
    }

    #[test]
    fn waypoint_distance() {
        let a = Waypoint::new(0.0, 0.0);
        let b = Waypoint::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 0.001);
    }

    #[test]
    fn expression_from_str() {
        assert_eq!(Expression::from_str("smile"),    Expression::Smile);
        assert_eq!(Expression::from_str("thinking"), Expression::Thinking);
        assert_eq!(Expression::from_str("unknown"),  Expression::Neutral);
    }
}
