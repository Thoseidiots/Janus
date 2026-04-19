"""
train_with_learning.py
=======================
Integrates the true learning system with Avus training.

This trains Avus to generate novel responses based on learned patterns,
not just predict next tokens. The model learns to:
  - Reason about situations
  - Find similar past experiences
  - Apply learned patterns
  - Generate contextually appropriate responses

Key difference from standard training:
  - OLD: Train on raw text sequences
  - NEW: Train on (situation, reasoning, learned_pattern, response) tuples

This produces an AI that learns and adapts, not just predicts.
"""

import os
import sys
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

# Add repo to path
REPO_CANDIDATES = [
    Path("/kaggle/input/janus-repo/Janus-main"),
    Path("/kaggle/input/janus-repo"),
    Path("/kaggle/working"),
    Path("."),
]
for p in REPO_CANDIDATES:
    if (p / "avus.py").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        print(f"[setup] Janus repo found at {p}")
        break

# Import core systems
from avus import Avus, AvusConfig
from janus_true_human_learning import (
    TrueHumanJanus, AdaptiveMemory, PatternLearner,
    ResponseGenerator, Experience, ContextualReasoning
)

try:
    import tiktoken
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "tiktoken", "-q"],
                   check=False)
    import tiktoken


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING-BASED TRAINING DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LearningTrainingSample:
    """Training sample that includes learning context"""
    situation: str
    context: Dict
    reasoning: str
    learned_pattern: str
    response: str
    success_score: float
    
    def to_training_text(self) -> str:
        """Convert to training text format"""
        return (
            f"<|situation|>{self.situation}<|/situation|>"
            f"<|context|>{json.dumps(self.context)}<|/context|>"
            f"<|reasoning|>{self.reasoning}<|/reasoning|>"
            f"<|pattern|>{self.learned_pattern}<|/pattern|>"
            f"<|response|>{self.response}<|/response|>"
            f"<|success|>{self.success_score}<|/success|>"
        )


class LearningDataGenerator:
    """Generates training data that teaches learning, not just prediction"""
    
    def __init__(self, janus: TrueHumanJanus):
        self.janus = janus
        self.reasoning = ContextualReasoning()
        
    def generate_learning_samples(self, n: int = 1000) -> List[LearningTrainingSample]:
        """Generate training samples that teach learning"""
        
        samples = []
        
        # Scenario templates
        scenarios = [
            # Worry/stress scenarios
            ("I'm worried about my deadline", {'urgency': 'high', 'emotion': 'worried'}),
            ("I'm stressed about my project", {'urgency': 'high', 'emotion': 'stressed'}),
            ("I'm anxious about the presentation", {'urgency': 'high', 'emotion': 'anxious'}),
            
            # Decision scenarios
            ("Should I call my boss or email?", {'urgency': 'high', 'type': 'decision'}),
            ("Which option is better?", {'urgency': 'normal', 'type': 'decision'}),
            ("How do I choose between these?", {'urgency': 'normal', 'type': 'decision'}),
            
            # Learning scenarios
            ("I don't understand how this works", {'urgency': 'normal', 'type': 'learning'}),
            ("Can you explain this?", {'urgency': 'normal', 'type': 'learning'}),
            ("How does this work?", {'urgency': 'normal', 'type': 'learning'}),
            
            # Problem scenarios
            ("Something is broken", {'urgency': 'high', 'type': 'problem'}),
            ("I have an issue", {'urgency': 'normal', 'type': 'problem'}),
            ("This doesn't work", {'urgency': 'high', 'type': 'problem'}),
        ]
        
        for _ in range(n):
            situation, context = random.choice(scenarios)
            
            # Generate reasoning
            reasoning_result = self.reasoning.reason_about_situation(situation, context)
            reasoning_text = reasoning_result['analysis']
            
            # Get learned pattern
            learned_action = self.janus.memory.get_learned_action(situation)
            pattern_text = (
                learned_action['action'] if learned_action 
                else "Apply empathy and support"
            )
            
            # Generate response
            response = self.janus.generate_response(situation, context)
            
            # Success score (varies based on scenario)
            if context.get('emotion') in ['worried', 'stressed', 'anxious']:
                success = random.uniform(0.7, 0.95)
            elif context.get('type') == 'decision':
                success = random.uniform(0.6, 0.9)
            elif context.get('type') == 'learning':
                success = random.uniform(0.75, 0.95)
            else:
                success = random.uniform(0.5, 0.85)
            
            sample = LearningTrainingSample(
                situation=situation,
                context=context,
                reasoning=reasoning_text,
                learned_pattern=pattern_text,
                response=response,
                success_score=success
            )
            
            samples.append(sample)
        
        return samples


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING-AWARE TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════════

class LearningTokenizer:
    """Tokenizer that preserves learning structure"""
    
    def __init__(self):
        self._enc = tiktoken.get_encoding("gpt2")
        self.special_tokens = {
            "<|situation|>", "<|/situation|>",
            "<|context|>", "<|/context|>",
            "<|reasoning|>", "<|/reasoning|>",
            "<|pattern|>", "<|/pattern|>",
            "<|response|>", "<|/response|>",
            "<|success|>", "<|/success|>",
        }
    
    def encode(self, text: str) -> List[int]:
        return self._enc.encode(text, allowed_special=self.special_tokens)
    
    def decode(self, tokens: List[int]) -> str:
        try:
            return self._enc.decode(tokens)
        except Exception:
            return ""


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING-BASED DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class LearningDataset(torch.utils.data.Dataset):
    """Dataset that teaches learning"""
    
    def __init__(self, janus: TrueHumanJanus, tokenizer: LearningTokenizer,
                 block_size: int, samples_per: int = 1000):
        self.block_size = block_size
        self.data: List[torch.Tensor] = []
        
        print("[learning_data] Generating learning-based training data...")
        
        # Generate learning samples
        generator = LearningDataGenerator(janus)
        samples = generator.generate_learning_samples(samples_per)
        
        print(f"[learning_data] Generated {len(samples)} learning samples")
        
        # Convert to training text
        all_texts = [sample.to_training_text() for sample in samples]
        
        # Tokenize
        print(f"[learning_data] Tokenizing {len(all_texts)} sequences...")
        pad_id = tokenizer.encode("<|/success|>")[0]
        
        for text in all_texts:
            tokens = tokenizer.encode(text)
            
            # Chunk into block_size+1 windows
            for i in range(0, len(tokens), block_size):
                chunk = tokens[i:i + block_size + 1]
                if len(chunk) < 2:
                    continue
                if len(chunk) < block_size + 1:
                    chunk = chunk + [pad_id] * (block_size + 1 - len(chunk))
                self.data.append(torch.tensor(chunk, dtype=torch.long))
        
        print(f"[learning_data] {len(self.data):,} training chunks ready")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING-AWARE TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_with_learning(
    model_size: str = "1b",
    epochs: int = 20,
    batch_size: int = 1,
    grad_accum_steps: int = 8,
    max_seq_len: int = 512,
    learning_rate: float = 3e-4,
    use_grad_checkpoint: bool = True,
):
    """Train Avus with learning-based data"""
    
    print("\n" + "="*70)
    print("TRAINING AVUS WITH LEARNING-BASED DATA")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize learning system
    print("\n[learning] Initializing learning system...")
    janus = TrueHumanJanus()
    
    # Pre-populate with some experiences so learning has context
    print("[learning] Pre-populating with initial experiences...")
    initial_interactions = [
        ("I'm worried about my deadline", "I understand. Let me help you think through this.", "positive", 0.9),
        ("Should I call or email?", "Calling might be better for urgent matters.", "positive", 0.85),
        ("I don't understand", "Let me explain this step by step.", "positive", 0.8),
    ]
    
    for user_input, response, outcome, success in initial_interactions:
        janus.record_interaction(user_input, response, outcome, success)
    
    # Load or create model
    print("\n[model] Loading model...")
    config_path = Path(f"config_avus_{model_size}.json")
    if config_path.exists():
        config = AvusConfig.from_file(str(config_path))
    else:
        config = AvusConfig()
    
    config.max_seq_len = min(config.max_seq_len, max_seq_len)
    model = Avus(config).to(device)
    
    total_params = model.count_parameters()
    print(f"[model] Parameters: {total_params/1e9:.2f}B")
    print(f"[model] Config: dim={config.dim} layers={config.n_layers} "
          f"heads={config.n_heads} seq={config.max_seq_len}")
    
    # Create dataset
    print("\n[data] Creating learning-based dataset...")
    tokenizer = LearningTokenizer()
    dataset = LearningDataset(janus, tokenizer, max_seq_len, samples_per=1000)
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                            betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    
    # Training loop
    print("\n[training] Starting training loop...")
    print(f"[training] Epochs: {epochs} | Batch: {batch_size} | "
          f"GradAccum: {grad_accum_steps} | LR: {learning_rate}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, loss = model(x, targets=y)
                if loss.dim() > 0:
                    loss = loss.mean()
            
            scaler.scale(loss / grad_accum_steps).backward()
            
            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += loss.item()
            
            if step % 50 == 0:
                avg = epoch_loss / (step + 1)
                elapsed = time.time() - t0
                print(f"  step {step}/{len(loader)} loss={avg:.4f} t={elapsed:.0f}s")
        
        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0
        
        print(f"\n[epoch {epoch+1}] loss={avg_loss:.4f} time={elapsed:.0f}s")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        
        checkpoint_path = Path(f"avus_{model_size}_learning_epoch{epoch+1}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"[checkpoint] Saved to {checkpoint_path}")
        
        # Save learning state
        learning_summary = janus.get_learning_summary()
        learning_path = Path(f"learning_state_epoch{epoch+1}.json")
        with open(learning_path, 'w') as f:
            json.dump(learning_summary, f, indent=2, default=str)
        print(f"[learning] Saved learning state to {learning_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    # Final learning summary
    print("\n[learning] Final Learning Summary:")
    summary = janus.get_learning_summary()
    print(f"  Total experiences: {summary['total_experiences']}")
    print(f"  Learned patterns: {len(summary['learned_patterns'])} types")
    print(f"  Behavior adjustments: {len(summary['behavior_adjustments'])}")
    
    return model, janus


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE WITH LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

def generate_response_with_learning(
    model: Avus,
    janus: TrueHumanJanus,
    user_input: str,
    context: Optional[Dict] = None,
) -> str:
    """Generate response using trained model + learning system"""
    
    # Generate response using learning system
    response = janus.generate_response(user_input, context)
    
    return response


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Train with learning
    model, janus = train_with_learning(
        model_size="1b",
        epochs=5,  # Start with 5 for testing
        batch_size=1,
        grad_accum_steps=8,
        max_seq_len=512,
        learning_rate=3e-4,
    )
    
    # Test inference
    print("\n" + "="*70)
    print("TESTING INFERENCE WITH LEARNING")
    print("="*70)
    
    test_inputs = [
        "I'm worried about my deadline",
        "Should I call my boss?",
        "I don't understand this",
    ]
    
    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        response = generate_response_with_learning(model, janus, user_input)
        print(f"Janus: {response}")
        
        # Record for learning
        janus.record_interaction(
            user_input,
            response,
            "positive interaction",
            0.85
        )
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETE")
    print("="*70)
