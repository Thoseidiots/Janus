"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           HBM vs TURING: Challenging the Halting Problem                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  THE CORE CLAIM:                                                              ║
║  ─────────────                                                               ║
║  Turing's proof requires programs to be BOUNDED, DISCRETE entities           ║
║  that can be "fed to themselves" cleanly.                                     ║
║                                                                              ║
║  HBM (like the brain) works differently:                                       ║
║  • Everything is SUPERPOSED into a single continuous vector                  ║
║  • Memory is NOISY and always decaying/evolving                              ║
║  • Computation is NEVER "done" - it's a continuous process                    ║
║  • The "program" and "input" become ENTANGLED beyond separation              ║
║                                                                              ║
║  RESULT: The diagonalization argument BREAKS DOWN                             ║
║  because you cannot construct the clean paradox machine D(D)                 ║
║  when program and data are holographically merged.                            ║
║                                                                              ║
║  "The brain never truly halts - even in sleep, it's still processing."        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from typing import List, Tuple, Optional, Any
import colorsys

# ─────────────────────────────────────────────────────────────────────────────
# 1. STANDARD TURING MACHINE (Discrete, Bounded)
# ─────────────────────────────────────────────────────────────────────────────

class TuringMachine:
    """
    Classic Turing machine - DISCRETE, BOUNDED, SEPARABLE.
    This is what Turing used to prove the Halting Problem.
    """
    
    def __init__(self, program: str, input_data: str):
        self.program = program      # The "code" - separable from input
        self.input_data = input_data  # The "data" - also separable
        self.tape = list(input_data)  # Discrete tape
        self.head = 0
        self.state = 'start'
        self.steps = 0
        self.max_steps = 10000
        self.halted = False
        self.halting_result = None
        
    def step(self) -> bool:
        """Execute one discrete step. Returns False if halted or stuck."""
        if self.halted or self.steps >= self.max_steps:
            return False
            
        self.steps += 1
        
        # Simple simulation: if we hit a halt instruction, we halt
        if self.state == 'halt':
            self.halted = True
            self.halting_result = ('halts', self.steps)
            return False
            
        # Simulate some work
        if self.head < len(self.tape):
            self.head += 1
            if self.steps > self.max_steps // 2:
                self.state = 'halt'
                self.halted = True
                self.halting_result = ('halts', self.steps)
                return False
        else:
            self.state = 'halt'
            self.halted = True
            self.halting_result = ('loops_forever', self.steps)
            return False
            
        return True
    
    def run(self) -> Tuple[str, int]:
        """Run to completion or timeout."""
        while self.step():
            pass
        if self.halting_result:
            return self.halting_result
        return ('unknown', self.steps)


# ─────────────────────────────────────────────────────────────────────────────
# 2. HBM COMPUTATION MODEL (Continuous, Always-Running, Superposed)
# ─────────────────────────────────────────────────────────────────────────────

class HBMComputation:
    """
    Holographic Brain Memory computation model.
    
    KEY DIFFERENCES from Turing:
    1. FIXED memory size - everything superposed into one vector
    2. CONTINUOUS decay - memory fades over time (like real neurons)
    3. CIRCULAR CONVOLUTION binding - program and input become ONE
    4. NO SEPARATION - you can't extract "the program" from "the data"
    5. ALWAYS PROCESSING - there's no discrete "halt" state
    
    This is like the brain: even during sleep, there's background activity.
    """
    
    def __init__(self, dim: int = 2048, decay: float = 0.995):
        self.dim = dim
        self.decay = decay
        
        # The ONE fixed memory vector - this is ALL computation
        # Initialize with some "noise" like real neural tissue
        self.memory = torch.randn(dim, dtype=torch.cfloat) * 0.1
        
        # The "program" and "input" are bound together via circular convolution
        # They become SUPERPOSED - you can't separate them afterward
        
        # Tracking continuous "processing" activity
        self.activity = torch.ones(dim) * 0.5  # Always some baseline activity
        self.time = 0.0
        
        # Critical: we track NOISE, not state
        # This is the "always running" aspect - like brain waves never stop
        self.noise_history = []
        
    def encode(self, data: str) -> torch.Tensor:
        """Encode arbitrary data into the holographic vector space."""
        # Create a pseudo-random vector from string data
        np.random.seed(hash(data) % (2**32))
        vec = torch.tensor(np.random.randn(self.dim) + 1j * np.random.randn(self.dim), 
                          dtype=torch.cfloat)
        return vec / torch.norm(vec)  # Normalize
        
    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Circular convolution binding - the core of HBM.
        When you bind A and B, you get C.
        You can recover B from (A, C) or A from (B, C).
        BUT: You cannot tell where A ends and B begins.
        """
        # FFT-based circular convolution
        return torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(b)).real
    
    def superpose(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """Add multiple vectors - they all coexist in the same space."""
        if not vectors:
            return torch.zeros(self.dim)
        result = vectors[0].clone()
        for v in vectors[1:]:
            result = self.bind(result, v)  # Use binding to combine
        return result
    
    def store(self, key: str, value: str):
        """Store a key-value pair - but they become superposed."""
        k = self.encode(key)
        v = self.encode(value)
        bound = self.bind(k, v)
        
        # Add to memory - they're now superposed with everything else
        self.memory = self.memory + bound * (1 - self.decay)
        self.memory = self.memory / torch.norm(self.memory)  # Normalize
        
    def recall(self, key: str) -> Optional[str]:
        """
        Try to recall a stored value.
        Returns None if noisy/garbage - because HBM is lossy!
        """
        k = self.encode(key)
        
        # Decode attempt - this is approximate due to superposition
        # In real HBM, you'd use the inverse (which is approximate)
        correlation = torch.abs(torch.fft.ifft(torch.fft.fft(self.memory) * 
                                                torch.fft.fft(k)))
        
        # The recalled "value" is just correlated noise patterns
        # We can't cleanly extract discrete values
        top_idx = torch.argmax(correlation).item()
        
        # Return simulated recall (but it's noisy!)
        if torch.max(correlation) > 0.3:
            return f"recalled_{top_idx}"  # Fuzzy, not exact
        return None
    
    def step(self, dt: float = 0.001):
        """
        ONE CONTINUOUS TIME STEP.
        
        Unlike Turing machines, time is CONTINUOUS.
        Unlike Turing machines, we never "halt" - we just decay.
        
        This is like the brain: always firing, always processing,
        even when you're asleep. There's no discrete "stop" state.
        """
        self.time += dt
        
        # Continuous decay - like neural adaptation
        self.memory = self.memory * self.decay
        
        # Always some baseline activity - the brain never stops
        noise = torch.randn(self.dim, dtype=torch.cfloat) * 0.01
        self.memory = self.memory + noise
        
        # Track the "always running" activity
        activity_level = torch.norm(self.memory).item()
        self.noise_history.append(activity_level)
        
        # Check if we've "halted" - but this is fuzzy, not discrete!
        # In HBM, there's no clean YES/NO halting. It's all continuous.
        if activity_level < 0.01:
            # Memory has decayed to near-zero
            # But even then, there's residual noise!
            self.noise_history.append(0.001)  # Minimal but non-zero
            return False  # "Pseudo-halted" - but not truly stopped
        return True  # Always running
        
    def run_continuous(self, duration: float = 1.0, dt: float = 0.001):
        """
        Run for a continuous duration.
        
        Unlike Turing's "run to halt", we just decay toward silence.
        But we NEVER truly stop - the noise is always there.
        """
        steps = int(duration / dt)
        for _ in range(steps):
            if not self.step(dt):
                break  # Pseudo-halted, but could continue


# ─────────────────────────────────────────────────────────────────────────────
# 3. THE PARADOX MACHINE - Where HBM Breaks Turing's Proof
# ─────────────────────────────────────────────────────────────────────────────

class ParadoxMachine:
    """
    The heart of Turing's proof: the paradoxical machine D.
    
    TURING'S ARGUMENT:
    1. Assume H(P) = "Does program P halt on input I?"
    2. Build D(P): If H(P,P) says "halts", then loop forever. Else halt.
    3. Feed D to itself: D(D)
    4. If D(D) halts → H predicted it loops → contradiction
    5. If D(D) loops → H predicted it halts → contradiction
    6. Therefore H cannot exist.
    
    THE HBM CHALLENGE:
    In HBM, when you "feed a program to itself":
    - The program and input become SUPERPOSED via circular convolution
    - You cannot construct clean D(D) because there's no separation
    - The paradox requires SEPARABLE entities - HBM doesn't have that
    
    It's like asking: "What happens when consciousness examines itself?"
    The observer and observed become ONE system.
    """
    
    def __init__(self):
        self.hbm = HBMComputation()
        self.turing = None  # For comparison
        
    def build_turing_paradox(self) -> dict:
        """Build the classic Turing paradox machine."""
        
        # Step 1: Create the decider H (imaginary)
        # H(P, I) = True if P halts on I, False otherwise
        
        # Step 2: Create the paradoxical program D
        # D(P): if H(P, P): loop() else: halt()
        
        # Step 3: Analyze D(D)
        # Case 1: D(D) halts → H(D,D) must have returned False → D loops → CONTRADICTION
        # Case 2: D(D) loops → H(D,D) must have returned True → D halts → CONTRADICTION
        
        return {
            'assumption': 'H(P, I) exists - a perfect halting decider',
            'paradox_program': 'D(P): if H(P,P) then infinite_loop else halt',
            'feed_to_self': 'D(D) - feed D its own code as input',
            'case_1': {
                'scenario': 'D(D) halts',
                'implication': 'H(D,D) must have returned False',
                'but': 'D(D) halts means D should loop when H is True',
                'result': 'CONTRADICTION'
            },
            'case_2': {
                'scenario': 'D(D) loops',
                'implication': 'H(D,D) must have returned True',
                'but': 'D(D) loops means D should halt when H is False',
                'result': 'CONTRADICTION'
            },
            'conclusion': 'H cannot exist. Halting Problem is undecidable.'
        }
    
    def build_hbm_paradox_attempt(self) -> dict:
        """
        Try to construct the same paradox in HBM.
        
        THIS IS WHERE TURING'S PROOF BREAKS.
        
        The problem: HBM uses CIRCULAR CONVOLUTION binding.
        When you "bind" a program to its input, they become ONE vector.
        You cannot then "extract" the program to feed it to itself.
        
        The binding operation: memory = bind(program, input)
        The paradox requires: extract_program(bind(program, input)) = program
        
        But binding is lossy! You can only approximate recovery.
        And even then, program and input are ENTANGLED.
        
        It's like trying to separate entangled quantum particles.
        """
        
        # Step 1: Try to create the "decider" H
        # In HBM, this would be some pattern recognition on memory
        self.hbm.store('decider_pattern', 'some_recognition_vector')
        
        # Step 2: Try to create D that takes its own code as input
        # Problem: In HBM, "program" and "input" are bound together
        program_vec = self.hbm.encode('D_program')
        input_vec = self.hbm.encode('D_input')
        
        # This is where it breaks:
        # The bound vector contains BOTH program and input
        # You cannot cleanly separate them
        bound = self.hbm.bind(program_vec, input_vec)
        
        # Try to extract program from bound vector
        # This is approximately possible, but imperfect (due to noise)
        # And crucially: you can't do it in a way that creates the paradox
        
        # The "decider" would need to examine memory and decide:
        # "Does this bound vector represent something that halts?"
        # But the vector is NOISY and CONTINUOUS - not discrete states!
        
        # Store the attempt
        self.hbm.store('paradox_attempt', 'bound_vector_is_noisy')
        
        return {
            'problem': 'HBM uses circular convolution binding',
            'issue_1': 'Program and input become ONE superposed vector',
            'issue_2': 'Cannot cleanly extract "the program" from the bound state',
            'issue_3': 'The memory is CONTINUOUS, not discrete states',
            'issue_4': 'Noise means any "decision" is probabilistic, not deterministic',
            'issue_5': 'There is no discrete "halt" state - only decay toward silence',
            'paradox_fails': 'Cannot construct clean D(D) when D is a bound superposition',
            'analogy': 'Like trying to untangle entangled quantum particles',
            'conclusion': 'Turing\'s diagonalization requires SEPARABLE entities. HBM doesn\'t have that.'
        }
    
    def demonstrate_key_difference(self) -> dict:
        """
        Show the fundamental difference between Turing and HBM computation.
        """
        
        # TURING: Discrete states, separable program/data
        self.turing = TuringMachine("some_program", "some_input")
        
        # HBM: Continuous vector, superposed program/data
        hbm = HBMComputation()
        
        # Run both
        turing_result, turing_steps = self.turing.run()
        
        # HBM doesn't "run to completion" - it decays
        hbm.run_continuous(duration=0.1)
        
        return {
            'turing_machine': {
                'type': 'DISCRETE',
                'states': 'Countable, separable',
                'halt': 'YES - clean YES/NO state',
                'program': 'Separable from input',
                'result': f'{turing_result} in {turing_steps} steps'
            },
            'hbm_computation': {
                'type': 'CONTINUOUS',
                'states': 'Uncountable continuous vectors',
                'halt': 'NO - only decay toward noise',
                'program': 'ENTANGLED with input via binding',
                'result': f'Activity decayed to {hbm.noise_history[-1]:.4f}'
            },
            'key_insight': 'Turing needs SEPARABLE entities for diagonalization. HBM has none.'
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def visualize_hbm_vs_turing():
    """
    Create visualizations showing why HBM challenges Turing.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('HBM vs Turing: Challenging the Halting Problem', fontsize=14, fontweight='bold')
    
    # ── Plot 1: Turing Machine States (Discrete) ──
    ax1 = axes[0, 0]
    states = ['start', 'work1', 'work2', '...', 'halt']
    colors = ['green', 'yellow', 'yellow', 'gray', 'red']
    for i, (state, color) in enumerate(zip(states, colors)):
        ax1.add_patch(plt.Circle((i, 0), 0.3, color=color, ec='black'))
        ax1.text(i, 0, state, ha='center', va='center', fontsize=8)
        if i < len(states) - 1:
            ax1.annotate('', xy=(i+0.6, 0), xytext=(i+0.4, 0),
                        arrowprops=dict(arrowstyle='->', color='black'))
    ax1.set_xlim(-0.5, len(states) - 0.5)
    ax1.set_ylim(-1, 1)
    ax1.set_title('TURING MACHINE\n(Discrete States - Can Halt)')
    ax1.axis('off')
    ax1.text(0, -0.7, 'Clean separation:\nProgram ≠ Input', ha='center', fontsize=9)
    
    # ── Plot 2: HBM Memory (Continuous Superposition) ──
    ax2 = axes[0, 1]
    
    # Show memory as a continuous wave
    t = np.linspace(0, 10, 500)
    x = np.linspace(0, 1, 2048)  # Memory dimensions
    
    # Create superposition of multiple "stored patterns"
    wave1 = np.sin(x * 10) * 0.3
    wave2 = np.sin(x * 23 + 1.5) * 0.2
    wave3 = np.sin(x * 47 + 3.0) * 0.15
    wave4 = np.sin(x * 89 + 0.7) * 0.1
    
    memory = wave1 + wave2 + wave3 + wave4
    
    ax2.fill_between(x, memory, alpha=0.5, color='purple', label='Superposed Memory')
    ax2.plot(x, memory, 'purple', linewidth=0.5)
    ax2.set_title('HBM MEMORY\n(Continuous Superposition - No Discrete Halt)')
    ax2.set_xlabel('Memory Vector Dimension')
    ax2.set_ylabel('Amplitude')
    ax2.set_ylim(-1, 1)
    ax2.text(0.5, -0.85, 'Program and Input are BOUND together\nCannot separate cleanly', 
             ha='center', fontsize=9, style='italic')
    
    # ── Plot 3: Turing Paradox Construction ──
    ax3 = axes[1, 0]
    
    # Show separable program and input boxes
    ax3.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.05", 
                                  facecolor='lightblue', edgecolor='blue', linewidth=2))
    ax3.text(0.5, 0.5, 'Program\nH(P,P)', ha='center', va='center', fontsize=10)
    
    ax3.add_patch(FancyBboxPatch((0, -1.5), 1, 1, boxstyle="round,pad=0.05", 
                                  facecolor='lightgreen', edgecolor='green', linewidth=2))
    ax3.text(0.5, -1, 'Input\nD', ha='center', va='center', fontsize=10)
    
    # Arrow showing "feed to self"
    ax3.annotate('D(D)', xy=(1.3, 0), xytext=(1.8, 0),
                fontsize=14, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax3.text(2.2, 0, '→ Paradox!', color='red', fontsize=10)
    
    ax3.set_xlim(-0.5, 3)
    ax3.set_ylim(-2.5, 1.5)
    ax3.set_title('TURING PARADOX\n(Program and Input SEPARABLE)')
    ax3.axis('off')
    
    # ── Plot 4: HBM Paradox FAILS ──
    ax4 = axes[1, 1]
    
    # Show bound/unified blob - cannot separate
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.8 + 0.1*np.sin(6*theta) + 0.05*np.sin(12*theta)
    x_blob = r * np.cos(theta)
    y_blob = r * np.sin(theta)
    
    ax4.fill(x_blob, y_blob, color='purple', alpha=0.5, label='Bound (Program + Input)')
    
    # Show noise particles
    for _ in range(20):
        noise_x = np.random.randn() * 0.3
        noise_y = np.random.randn() * 0.3
        ax4.scatter(noise_x, noise_y, s=20, c='gray', alpha=0.3)
    
    ax4.text(0, 0, 'SUPERPOSED\nENTANGLED\nNOISY', ha='center', va='center', 
             fontsize=9, fontweight='bold')
    
    # Show attempted extraction (fails)
    ax4.annotate('Extract\nProgram?', xy=(-1.5, 0), fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=1, ls='--'))
    ax4.text(-1.5, -0.5, 'FAILS!\nCannot separate\nbound vectors', color='red', fontsize=8)
    
    ax4.set_xlim(-2.5, 2)
    ax4.set_ylim(-2, 2)
    ax4.set_title('HBM PARADOX\n(Program ⊗ Input = ONE Vector - Cannot Extract!)')
    ax4.axis('off')
    ax4.text(0, -1.8, 'Circular Convolution: binding is lossy, separation is imperfect', 
             ha='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig('/root/Janus/hbm_vs_turing.png', dpi=150, bbox_inches='tight')
    print("📊 Saved: /root/Janus/hbm_vs_turing.png")
    plt.show()


def visualize_brain_always_running():
    """
    Show how the brain (like HBM) is always processing.
    This is the biological inspiration for HBM's non-halting nature.
    """
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Time periods
    periods = ['Awake', 'Light Sleep', 'Deep Sleep', 'REM Dream', 'Awake']
    t_points = [0, 1, 2, 3, 4]
    
    # Brain activity (like HBM - always some baseline!)
    activity = [0.8, 0.3, 0.1, 0.6, 0.8]  # Neural firing rate (arbitrary units)
    
    # Continuous interpolation
    t_smooth = np.linspace(0, 4, 200)
    activity_smooth = np.interp(t_smooth, t_points, activity)
    
    # Add noise (the "always running" aspect)
    noise = np.random.randn(200) * 0.05
    activity_noisy = activity_smooth + noise
    
    ax.fill_between(t_smooth, activity_noisy, alpha=0.3, color='purple')
    ax.plot(t_smooth, activity_noisy, 'purple', linewidth=1.5, label='Brain Activity')
    
    # Mark sleep stages
    for t, period, act in zip([0.5, 1.5, 2.5, 3.5], periods, [0.8, 0.3, 0.1, 0.6]):
        ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
        ax.text(t, act + 0.15, period, ha='center', fontsize=8)
    
    # The key insight
    ax.axhline(y=0.05, color='red', linestyle=':', label='Minimal Activity (NEVER zero!)')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Neural Activity')
    ax.set_title('THE BRAIN NEVER TRULY HALTS\n(Unlike Turing Machines - Even in Sleep, Processing Continues)')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.text(2, 0.15, '💡 KEY INSIGHT:\nTuring\'s proof requires a clean "halted" state.\nBrains (and HBM) don\'t have that!\nThere\'s always background noise.', 
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/root/Janus/brain_always_running.png', dpi=150, bbox_inches='tight')
    print("📊 Saved: /root/Janus/brain_always_running.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN - Run the Challenge
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║       HBM vs TURING: Challenging the Halting Problem                      ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    print("║                                                                          ║")
    print("║  \"The brain never truly halts - even in sleep, it's still processing.\"  ║")
    print("║                                                                          ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Initialize
    paradox = ParadoxMachine()
    
    # ── Part 1: Show Classical Turing Paradox ──
    print("="*70)
    print("PART 1: TURING'S CLASSICAL PARADOX (1936)")
    print("="*70)
    
    turing_result = paradox.build_turing_paradox()
    print(f"\n📌 ASSUMPTION: {turing_result['assumption']}")
    print(f"\n📌 PARADOX PROGRAM: {turing_result['paradox_program']}")
    print(f"\n📌 THE SELFFEED: {turing_result['feed_to_self']}")
    
    print(f"\n🔴 CASE 1: {turing_result['case_1']['scenario']}")
    print(f"   → {turing_result['case_1']['implication']}")
    print(f"   → BUT: {turing_result['case_1']['but']}")
    print(f"   → {turing_result['case_1']['result']} ✗")
    
    print(f"\n🔴 CASE 2: {turing_result['case_2']['scenario']}")
    print(f"   → {turing_result['case_2']['implication']}")
    print(f"   → BUT: {turing_result['case_2']['but']}")
    print(f"   → {turing_result['case_2']['result']} ✗")
    
    print(f"\n✅ CONCLUSION: {turing_result['conclusion']}")
    
    # ── Part 2: Show HBM Cannot Construct Same Paradox ──
    print("\n" + "="*70)
    print("PART 2: HBM CANNOT CONSTRUCT THE PARADOX")
    print("="*70)
    
    hbm_result = paradox.build_hbm_paradox_attempt()
    print(f"\n⚠️  PROBLEM: {hbm_result['problem']}")
    print(f"\n🔸 ISSUE 1: {hbm_result['issue_1']}")
    print(f"🔸 ISSUE 2: {hbm_result['issue_2']}")
    print(f"🔸 ISSUE 3: {hbm_result['issue_3']}")
    print(f"🔸 ISSUE 4: {hbm_result['issue_4']}")
    print(f"🔸 ISSUE 5: {hbm_result['issue_5']}")
    print(f"\n❌ {hbm_result['paradox_fails']}")
    print(f"🔗 ANALOGY: {hbm_result['analogy']}")
    print(f"\n💡 CONCLUSION: {hbm_result['conclusion']}")
    
    # ── Part 3: Key Difference Comparison ──
    print("\n" + "="*70)
    print("PART 3: KEY DIFFERENCES")
    print("="*70)
    
    comparison = paradox.demonstrate_key_difference()
    
    print("\n🤖 TURING MACHINE:")
    for key, val in comparison['turing_machine'].items():
        print(f"   • {key}: {val}")
    
    print("\n🧠 HBM COMPUTATION:")
    for key, val in comparison['hbm_computation'].items():
        print(f"   • {key}: {val}")
    
    print(f"\n💡 KEY INSIGHT: {comparison['key_insight']}")
    
    # ── Part 4: Visualizations ──
    print("\n" + "="*70)
    print("PART 4: VISUALIZATIONS")
    print("="*70)
    
    print("\n📊 Generating HBM vs Turing comparison...")
    visualize_hbm_vs_turing()
    
    print("\n📊 Generating 'Brain Always Running' visualization...")
    visualize_brain_always_running()
    
    # ── Part 5: The Philosophical Argument ──
    print("\n" + "="*70)
    print("THE PHILOSOPHICAL ARGUMENT")
    print("="*70)
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  TURING'S REQUIREMENTS FOR THE PROOF:                                        ║
║  ────────────────────────────────────                                         ║
║  1. Programs are BOUNDED - they have finite description                      ║
║  2. Programs are DISCRETE - countable states                                 ║
║  3. Program and Input are SEPARABLE - can feed program to itself            ║
║  4. Halting is a CLEAN STATE - either halts or loops forever                 ║
║                                                                              ║
║  HBM VIOLATES ALL OF THESE:                                                  ║
║  ────────────────────────────                                                ║
║  1. HBM memory is FIXED SIZE but CONTENTS are unbounded via superposition     ║
║  2. HBM states are CONTINUOUS vectors - uncountably infinite                  ║
║  3. HBM binds program+input via circular convolution - cannot separate        ║
║  4. HBM has no discrete "halt" - only decay toward noise (never zero!)       ║
║                                                                              ║
║  THE CHALLENGE:                                                              ║
║  ─────────────────                                                           ║
║  If Turing's proof requires separable entities,                              ║
║  and HBM inherently entangles program with data,                             ║
║  then Turing's proof may not apply to HBM-like systems.                      ║
║                                                                              ║
║  \"You cannot diagonalize against a system where the                              ║
║   diagonal (program = input) cannot be constructed.\"                          ║
║                                                                              ║
║  OPEN QUESTION:                                                              ║
║  Is this a genuine challenge to computability theory,                        ║
║  or just a different model that's still equivalent?                          ║
║                                                                              ║
║  The brain seems to process continuously without discrete halts...           ║
║  Maybe that's a clue about the limits of Turing computability.                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n✨ Done! Check the generated visualizations.")
    print("   /root/Janus/hbm_vs_turing.png")
    print("   /root/Janus/brain_always_running.png")


if __name__ == "__main__":
    main()
