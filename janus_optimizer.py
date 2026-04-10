import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import functools

class JanusOptimizer:
    """
    A high-performance optimization wrapper for the Janus AI model.
    Implements:
    - Activation Checkpointing (Memory optimization)
    - Gradient Accumulation (Virtual batch size)
    - Automatic Mixed Precision (Speed & Memory)
    - Kernel Fusion via torch.compile (Speed)
    """
    
    def __init__(
        self, 
        model, 
        optimizer, 
        scaler=None, 
        accumulation_steps=4, 
        use_checkpointing=True,
        use_compile=True
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler or torch.cuda.amp.GradScaler()
        self.accumulation_steps = accumulation_steps
        self.use_checkpointing = use_checkpointing
        
        # Apply torch.compile for kernel fusion if supported (PyTorch 2.0+)
        if use_compile and hasattr(torch, 'compile'):
            print("[optimizer] Applying torch.compile for kernel fusion...")
            self.model = torch.compile(self.model)
            
        if self.use_checkpointing:
            print("[optimizer] Activation checkpointing enabled.")
            self._apply_checkpointing()

    def _apply_checkpointing(self):
        """
        Wraps the AvusBlocks in the model with activation checkpointing.
        This significantly reduces VRAM usage by recomputing activations during backward pass.
        """
        # We need to find the blocks. In Avus, they are in model.layers
        # Note: If model is compiled, we need to access the original model
        base_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        
        if hasattr(base_model, 'layers'):
            for i, layer in enumerate(base_model.layers):
                # Wrap the forward method of each block
                layer.forward = functools.partial(checkpoint, layer.forward, use_reentrant=False)
            print(f"[optimizer] Checkpointed {len(base_model.layers)} layers.")

    def training_step(self, x, y, step_idx):
        """
        Performs a single optimized training step.
        Handles mixed precision and gradient accumulation.
        """
        device = x.device
        
        # 1. Mixed Precision Forward Pass
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits, loss = self.model(x, targets=y)
            # Scale loss for accumulation
            loss = loss / self.accumulation_steps

        # 2. Scaled Backward Pass
        self.scaler.scale(loss).backward()

        # 3. Optimizer Step (only after accumulation_steps)
        if (step_idx + 1) % self.accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
            return loss.item() * self.accumulation_steps, True
        
        return loss.item() * self.accumulation_steps, False

def get_optimized_dataloader(dataset, batch_size, device):
    """
    Returns a DataLoader optimized for GPU throughput.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4, # Increase for faster data loading
        pin_memory=(device.type == "cuda"),
        persistent_workers=True if device.type == "cuda" else False
    )
