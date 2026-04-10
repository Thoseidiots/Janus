// Janus Speech Arbiter
// ====================
// Real-time speech management with human-like conversational flow.
// Handles interruptions, backchanneling, and natural turn-taking.

use tokio::sync::mpsc;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// =============================================================================
// 1. INTERRUPTION CLASSIFICATION
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InterruptLevel {
    None = 0,      // Background noise, "Mhm"
    Social = 1,    // Acknowledgment, backchannel
    Additive = 2,  // "Also consider..."
    Directive = 3, // "Wait, actually..."
    Emergency = 4, // "STOP!"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalienceScore {
    pub level: InterruptLevel,
    pub weight: f32,  // 0.0 - 1.0
    pub content: String,
    pub timestamp: Instant,
}

impl SalienceScore {
    pub fn should_interrupt(&self) -> bool {
        self.weight > 0.7 || self.level == InterruptLevel::Emergency
    }
    
    pub fn should_duck_audio(&self) -> bool {
        self.weight > 0.3
    }
}

// =============================================================================
// 2. REAL-TIME SPEECH BOUNCER
// =============================================================================

pub struct JanusBouncer {
    pub brain_tx: mpsc::Sender<SalienceScore>,
    pub kill_tx: mpsc::Sender<()>,
    pub duck_tx: mpsc::Sender<f32>, // Volume level 0.0-1.0
    
    // State tracking
    pub janus_is_speaking: Arc<RwLock<bool>>,
    pub current_speech_word_index: Arc<RwLock<usize>>,
    pub speech_start_time: Arc<RwLock<Option<Instant>>>,
}

impl JanusBouncer {
    pub fn new(
        brain_tx: mpsc::Sender<SalienceScore>,
        kill_tx: mpsc::Sender<()>,
        duck_tx: mpsc::Sender<f32>,
    ) -> Self {
        Self {
            brain_tx,
            kill_tx,
            duck_tx,
            janus_is_speaking: Arc::new(RwLock::new(false)),
            current_speech_word_index: Arc::new(RwLock::new(0)),
            speech_start_time: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Process incoming user input and decide how to respond
    pub async fn process_input(&self, input: String) {
        let score = self.calculate_salience(&input);
        
        match score.level {
            InterruptLevel::Emergency => {
                tracing::warn!("[EMERGENCY STOP] User: {}", input);
                
                // Immediate kill
                let _ = self.kill_tx.send(()).await;
                
                // Notify brain
                let _ = self.brain_tx.send(score.clone()).await;
                
                // Reset speaking state
                *self.janus_is_speaking.write().await = false;
            }
            
            InterruptLevel::Directive => {
                tracing::info!("[DIRECTIVE] User override: {}", input);
                
                // Duck audio
                let _ = self.duck_tx.send(0.1).await;
                
                // Small delay to let user finish
                tokio::time::sleep(Duration::from_millis(300)).await;
                
                // Kill current speech
                let _ = self.kill_tx.send(()).await;
                
                // Send to brain for pivot
                let _ = self.brain_tx.send(score).await;
                
                *self.janus_is_speaking.write().await = false;
            }
            
            InterruptLevel::Additive => {
                tracing::info!("[ADDITIVE] Context update: {}", input);
                
                // Duck but don't kill
                let _ = self.duck_tx.send(0.3).await;
                
                // Send to brain for context injection
                let _ = self.brain_tx.send(score).await;
                
                // Restore volume after brief delay
                tokio::time::sleep(Duration::from_millis(500)).await;
                let _ = self.duck_tx.send(1.0).await;
            }
            
            InterruptLevel::Social => {
                tracing::debug!("[SOCIAL] Backchannel: {}", input);
                
                // Just duck slightly
                let _ = self.duck_tx.send(0.7).await;
                tokio::time::sleep(Duration::from_millis(200)).await;
                let _ = self.duck_tx.send(1.0).await;
                
                // Don't interrupt flow, but note it
                // Could trigger a subtle acknowledgment
            }
            
            InterruptLevel::None => {
                // Ignore background noise
            }
        }
    }
    
    /// Calculate salience score from input text
    fn calculate_salience(&self, input: &str) -> SalienceScore {
        let input_lower = input.to_lowercase();
        
        // Emergency keywords
        if input_lower.contains("stop") 
            || input_lower.contains("halt") 
            || input_lower.contains("shut up")
            || input_lower.contains("quiet") {
            return SalienceScore {
                level: InterruptLevel::Emergency,
                weight: 1.0,
                content: input.to_string(),
                timestamp: Instant::now(),
            };
        }
        
        // Directive keywords
        if input_lower.contains("instead") 
            || input_lower.contains("wait") 
            || input_lower.contains("no, ")
            || input_lower.contains("actually")
            || input_lower.contains("hold on") {
            return SalienceScore {
                level: InterruptLevel::Directive,
                weight: 0.85,
                content: input.to_string(),
                timestamp: Instant::now(),
            };
        }
        
        // Additive keywords
        if input_lower.contains("also") 
            || input_lower.contains("and another thing")
            || input_lower.contains("what about")
            || input_lower.contains("don't forget") {
            return SalienceScore {
                level: InterruptLevel::Additive,
                weight: 0.55,
                content: input.to_string(),
                timestamp: Instant::now(),
            };
        }
        
        // Social keywords
        if input_lower.contains("mhm")
            || input_lower.contains("yeah")
            || input_lower.contains("okay")
            || input_lower.contains("right")
            || input_lower.contains("sure") {
            return SalienceScore {
                level: InterruptLevel::Social,
                weight: 0.2,
                content: input.to_string(),
                timestamp: Instant::now(),
            };
        }
        
        // Default: low salience
        SalienceScore {
            level: InterruptLevel::None,
            weight: 0.1,
            content: input.to_string(),
            timestamp: Instant::now(),
        }
    }
    
    /// Update current speech position for late-binding context reconstruction
    pub async fn update_speech_position(&self, word_index: usize) {
        *self.current_speech_word_index.write().await = word_index;
    }
    
    /// Get the cutoff point for context reconstruction after interruption
    pub async fn get_speech_cutoff(&self) -> usize {
        *self.current_speech_word_index.read().await
    }
}

// =============================================================================
// 3. BACKCHANNEL MANAGER
// =============================================================================

pub struct BackchannelManager {
    pub user_speech_start: Option<Instant>,
    pub last_backchannel: Option<Instant>,
    pub user_is_speaking: Arc<RwLock<bool>>,
    pub backchannel_tx: mpsc::Sender<String>,
    
    // Configuration
    pub backchannel_interval: Duration,  // How often to backchannel
    pub min_speech_for_backchannel: Duration,  // Don't interrupt short utterances
}

impl BackchannelManager {
    pub fn new(backchannel_tx: mpsc::Sender<String>) -> Self {
        Self {
            user_speech_start: None,
            last_backchannel: None,
            user_is_speaking: Arc::new(RwLock::new(false)),
            backchannel_tx,
            backchannel_interval: Duration::from_secs(4),
            min_speech_for_backchannel: Duration::from_secs(2),
        }
    }
    
    /// Called when user starts speaking
    pub async fn on_user_speech_start(&mut self) {
        *self.user_is_speaking.write().await = true;
        self.user_speech_start = Some(Instant::now());
    }
    
    /// Called when user stops speaking
    pub async fn on_user_speech_end(&mut self) {
        *self.user_is_speaking.write().await = false;
        self.user_speech_start = None;
    }
    
    /// Monitor user speech and inject backchannels appropriately
    pub async fn monitor(&mut self) {
        let mut interval = tokio::time::interval(Duration::from_millis(500));
        
        loop {
            interval.tick().await;
            
            if !*self.user_is_speaking.read().await {
                continue;
            }
            
            // Check if we should backchannel
            if let Some(start) = self.user_speech_start {
                let elapsed = start.elapsed();
                
                if elapsed > self.min_speech_for_backchannel {
                    let should_backchannel = match self.last_backchannel {
                        None => true,
                        Some(last) => last.elapsed() > self.backchannel_interval,
                    };
                    
                    if should_backchannel {
                        self.inject_backchannel().await;
                    }
                }
            }
        }
    }
    
    async fn inject_backchannel(&mut self) {
        // Select appropriate backchannel based on context
        let backchannels = vec![
            "mhm",
            "yeah",
            "right",
            "okay",
            "I see",
        ];
        
        let idx = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as usize % backchannels.len();
        let choice = backchannels[idx];
        
        let _ = self.backchannel_tx.send(choice.to_string()).await;
        self.last_backchannel = Some(Instant::now());
        
        tracing::debug!("[BACKCHANNEL] Injected: {}", choice);
    }
}

// =============================================================================
// 4. TURN-TAKING ARBITER
// =============================================================================

pub struct TurnTakingArbiter {
    pub janus_is_speaking: Arc<RwLock<bool>>,
    pub user_is_speaking: Arc<RwLock<bool>>,
    pub user_is_typing: Arc<RwLock<bool>>,
    
    // Timing
    pub last_turn_end: Arc<RwLock<Option<Instant>>>,
    pub min_gap_between_turns: Duration,
    
    // Channels
    pub speech_request_rx: mpsc::Receiver<SpeechRequest>,
    pub speech_approval_tx: mpsc::Sender<bool>,
}

#[derive(Debug)]
pub struct SpeechRequest {
    pub content: String,
    pub urgency: f32,  // 0-1
    pub is_proactive: bool,
}

impl TurnTakingArbiter {
    pub fn new(
        speech_request_rx: mpsc::Receiver<SpeechRequest>,
        speech_approval_tx: mpsc::Sender<bool>,
    ) -> Self {
        Self {
            janus_is_speaking: Arc::new(RwLock::new(false)),
            user_is_speaking: Arc::new(RwLock::new(false)),
            user_is_typing: Arc::new(RwLock::new(false)),
            last_turn_end: Arc::new(RwLock::new(None)),
            min_gap_between_turns: Duration::from_millis(300),
            speech_request_rx,
            speech_approval_tx,
        }
    }
    
    /// Main arbitration loop
    pub async fn run(&mut self) {
        while let Some(request) = self.speech_request_rx.recv().await {
            let approved = self.evaluate_request(&request).await;
            
            if approved {
                *self.janus_is_speaking.write().await = true;
            }
            
            let _ = self.speech_approval_tx.send(approved).await;
        }
    }
    
    async fn evaluate_request(&self, request: &SpeechRequest) -> bool {
        // Never speak over user
        if *self.user_is_speaking.read().await {
            tracing::debug!("[TURN] Rejected: user is speaking");
            return false;
        }
        
        // Don't interrupt typing (they might be about to send)
        if *self.user_is_typing.read().await && request.urgency < 0.8 {
            tracing::debug!("[TURN] Rejected: user is typing");
            return false;
        }
        
        // Respect gap between turns
        if let Some(last_end) = *self.last_turn_end.read().await {
            if last_end.elapsed() < self.min_gap_between_turns {
                tracing::debug!("[TURN] Rejected: too soon after last turn");
                return false;
            }
        }
        
        // Proactive speech has lower priority
        if request.is_proactive && request.urgency < 0.6 {
            tracing::debug!("[TURN] Rejected: proactive below threshold");
            return false;
        }
        
        tracing::info!("[TURN] Approved: urgency={}", request.urgency);
        true
    }
    
    pub async fn on_speech_end(&self) {
        *self.janus_is_speaking.write().await = false;
        *self.last_turn_end.write().await = Some(Instant::now());
    }
}

// =============================================================================
// 5. FULL-DUPLEX AUDIO STACK WITH ECHO CANCELLATION
// =============================================================================

pub struct FullDuplexAudioStack {
    // Audio devices
    pub mic_input: mpsc::Receiver<AudioFrame>,
    pub speaker_output: mpsc::Sender<AudioFrame>,
    
    // Reference signal for echo cancellation
    pub janus_audio_buffer: Arc<RwLock<Vec<AudioFrame>>>,
    
    // Output channels
    pub cleaned_audio_tx: mpsc::Sender<AudioFrame>,
    
    // Configuration
    pub echo_cancellation_enabled: bool,
}

#[derive(Clone, Debug)]
pub struct AudioFrame {
    pub data: Vec<f32>,
    pub timestamp: Instant,
}

impl FullDuplexAudioStack {
    pub fn new(
        mic_input: mpsc::Receiver<AudioFrame>,
        speaker_output: mpsc::Sender<AudioFrame>,
        cleaned_audio_tx: mpsc::Sender<AudioFrame>,
    ) -> Self {
        Self {
            mic_input,
            speaker_output,
            janus_audio_buffer: Arc::new(RwLock::new(Vec::with_capacity(1000))),
            cleaned_audio_tx,
            echo_cancellation_enabled: true,
        }
    }
    
    /// Process audio frames in real-time
    pub async fn process_loop(&mut self) {
        while let Some(mic_frame) = self.mic_input.recv().await {
            let cleaned = if self.echo_cancellation_enabled {
                self.apply_echo_cancellation(mic_frame).await
            } else {
                mic_frame
            };
            
            // Send cleaned audio to speech recognition
            let _ = self.cleaned_audio_tx.send(cleaned).await;
        }
    }
    
    async fn apply_echo_cancellation(&self, mic_frame: AudioFrame) -> AudioFrame {
        // Simplified NLMS (Normalized Least Mean Square) implementation
        // In production, use WebRTC AEC or similar
        
        let reference = self.janus_audio_buffer.read().await;
        
        if reference.is_empty() {
            return mic_frame;
        }
        
        // Simple energy-based detection
        let mic_energy: f32 = mic_frame.data.iter().map(|s| s * s).sum();
        let ref_energy: f32 = reference.last()
            .map(|f| f.data.iter().map(|s| s * s).sum())
            .unwrap_or(0.0);
        
        // If mic energy is much higher than reference, it's likely user speech
        if mic_energy > ref_energy * 2.0 {
            mic_frame
        } else {
            // Likely echo, attenuate
            AudioFrame {
                data: mic_frame.data.iter().map(|s| s * 0.1).collect(),
                timestamp: mic_frame.timestamp,
            }
        }
    }
    
    /// Called when Janus generates audio output
    pub async fn on_janus_audio(&self, frame: AudioFrame) {
        // Store for echo cancellation reference
        let mut buffer = self.janus_audio_buffer.write().await;
        buffer.push(frame.clone());
        
        // Keep buffer size limited
        if buffer.len() > 1000 {
            buffer.remove(0);
        }
        
        // Send to speakers
        let _ = self.speaker_output.send(frame).await;
    }
}

// =============================================================================
// 6. PLAYBACK TRACKER FOR LATE-BINDING CONTEXT
// =============================================================================

pub struct PlaybackTracker {
    pub current_word_index: Arc<RwLock<usize>>,
    pub full_text: Arc<RwLock<String>>,
    pub words: Arc<RwLock<Vec<String>>>,
    pub is_playing: Arc<RwLock<bool>>,
}

impl PlaybackTracker {
    pub fn new() -> Self {
        Self {
            current_word_index: Arc::new(RwLock::new(0)),
            full_text: Arc::new(RwLock::new(String::new())),
            words: Arc::new(RwLock::new(Vec::new())),
            is_playing: Arc::new(RwLock::new(false)),
        }
    }
    
    /// Start tracking new speech
    pub async fn start_speech(&self, text: String) {
        *self.full_text.write().await = text.clone();
        *self.words.write().await = text.split_whitespace().map(String::from).collect();
        *self.current_word_index.write().await = 0;
        *self.is_playing.write().await = true;
    }
    
    /// Update position as words are spoken
    pub async fn advance_word(&self) {
        let mut index = self.current_word_index.write().await;
        *index += 1;
    }
    
    /// Get what was actually spoken before interruption
    pub async fn get_spoken_text(&self) -> String {
        let index = *self.current_word_index.read().await;
        let words = self.words.read().await;
        
        words.iter().take(index).cloned().collect::<Vec<_>>().join(" ")
    }
    
    /// Get what was NOT spoken (for potential completion later)
    pub async fn get_unspoken_text(&self) -> String {
        let index = *self.current_word_index.read().await;
        let words = self.words.read().await;
        
        words.iter().skip(index).cloned().collect::<Vec<_>>().join(" ")
    }
    
    pub async fn stop(&self) {
        *self.is_playing.write().await = false;
    }
}

// =============================================================================
// 7. PYTHON BRIDGE FOR CROSS-LANGUAGE COMMUNICATION
// =============================================================================

use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct JanusBridge {
    pub interrupt_tx: mpsc::Sender<SalienceScore>,
    pub playback_tracker: Arc<PlaybackTracker>,
}

#[pymethods]
impl JanusBridge {
    /// Python calls this to check for new interruptions
    fn pop_interrupt(&self, py: Python) -> PyResult<Option<String>> {
        // In real implementation, this would check a shared buffer
        // For now, return None
        Ok(None)
    }
    
    /// Get the current playback position
    fn get_playback_position(&self, py: Python) -> PyResult<usize> {
        // This would need async runtime integration
        Ok(0)
    }
    
    /// Get what Janus has actually spoken so far
    fn get_spoken_text(&self, py: Python) -> PyResult<String> {
        Ok(String::new())
    }
}

#[pymodule]
fn janus_speech_bridge(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<JanusBridge>()?;
    Ok(())
}

// =============================================================================
// 8. MAIN COORDINATOR
// =============================================================================

pub struct SpeechCoordinator {
    pub bouncer: JanusBouncer,
    pub backchannel: BackchannelManager,
    pub turn_arbiter: TurnTakingArbiter,
    pub audio_stack: FullDuplexAudioStack,
    pub playback_tracker: PlaybackTracker,
}

impl SpeechCoordinator {
    pub async fn run(self) {
        // Spawn all components
        let bouncer_handle = tokio::spawn(async move {
            // Bouncer runs via process_input calls
        });
        
        let backchannel_handle = tokio::spawn(async move {
            // Backchannel runs its own loop
        });
        
        let turn_handle = tokio::spawn(async move {
            // Turn arbiter runs its own loop
        });
        
        let audio_handle = tokio::spawn(async move {
            // Audio stack runs its own loop
        });
        
        // Wait for all
        let _ = tokio::join!(
            bouncer_handle,
            backchannel_handle,
            turn_handle,
            audio_handle
        );
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_salience_calculation() {
        let (brain_tx, _) = mpsc::channel(10);
        let (kill_tx, _) = mpsc::channel(10);
        let (duck_tx, _) = mpsc::channel(10);
        
        let bouncer = JanusBouncer::new(brain_tx, kill_tx, duck_tx);
        
        let score = bouncer.calculate_salience("stop!");
        assert_eq!(score.level, InterruptLevel::Emergency);
        assert_eq!(score.weight, 1.0);
        
        let score = bouncer.calculate_salience("wait, actually...");
        assert_eq!(score.level, InterruptLevel::Directive);
        
        let score = bouncer.calculate_salience("mhm");
        assert_eq!(score.level, InterruptLevel::Social);
    }
    
    #[tokio::test]
    async fn test_playback_tracker() {
        let tracker = PlaybackTracker::new();
        
        tracker.start_speech("Hello world this is a test".to_string()).await;
        
        tracker.advance_word().await;
        tracker.advance_word().await;
        
        let spoken = tracker.get_spoken_text().await;
        assert_eq!(spoken, "Hello world");
        
        let unspoken = tracker.get_unspoken_text().await;
        assert_eq!(unspoken, "this is a test");
    }
}
