"""
Visualization utilities for memory traces and training metrics
"""

import torch
import numpy as np
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class MemoryTraceVisualizer:
    """Utilities for visualizing holographic memory traces and training metrics."""
    
    @staticmethod
    def plot_memory_magnitude(memory_magnitudes: List[float], save_path: Optional[str] = None):
        """
        Plot memory magnitude over time.
        
        Args:
            memory_magnitudes: List of memory magnitudes at each step
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(10, 6))
        plt.plot(memory_magnitudes, linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Memory Magnitude')
        plt.title('Holographic Memory Magnitude Over Time')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_training_metrics(metrics: Dict[str, List[float]], save_path: Optional[str] = None):
        """
        Plot multiple training metrics.
        
        Args:
            metrics: Dictionary of metric names to lists of values
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, (name, values) in zip(axes, metrics.items()):
            ax.plot(values, linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel(name)
            ax.set_title(f'{name} Over Time')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_memory_phase_distribution(memory: torch.Tensor, save_path: Optional[str] = None):
        """
        Plot the phase distribution of memory vector.
        
        Args:
            memory: Complex memory vector
            save_path: Optional path to save the figure
        """
        phases = torch.angle(memory).cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        plt.hist(phases, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Phase (radians)')
        plt.ylabel('Frequency')
        plt.title('Memory Phase Distribution')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_memory_magnitude_distribution(memory: torch.Tensor, save_path: Optional[str] = None):
        """
        Plot the magnitude distribution of memory vector.
        
        Args:
            memory: Complex memory vector
            save_path: Optional path to save the figure
        """
        magnitudes = torch.abs(memory).cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        plt.hist(magnitudes, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Magnitude')
        plt.ylabel('Frequency')
        plt.title('Memory Magnitude Distribution')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def create_memory_animation(memory_history: List[torch.Tensor], 
                               save_path: Optional[str] = None,
                               fps: int = 10):
        """
        Create an animation of memory evolution.
        
        Args:
            memory_history: List of memory vectors at each step
            save_path: Optional path to save the animation
            fps: Frames per second for animation
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            
            memory = memory_history[frame]
            magnitudes = torch.abs(memory).cpu().numpy()
            phases = torch.angle(memory).cpu().numpy()
            
            ax1.bar(range(len(magnitudes)), magnitudes)
            ax1.set_title(f'Memory Magnitude (Step {frame})')
            ax1.set_ylabel('Magnitude')
            
            ax2.scatter(range(len(phases)), phases, alpha=0.6)
            ax2.set_title(f'Memory Phase (Step {frame})')
            ax2.set_ylabel('Phase (radians)')
            
            return ax1, ax2
        
        anim = FuncAnimation(fig, update, frames=len(memory_history), 
                           interval=1000/fps, blit=False)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=fps)
        
        plt.close()
        return anim
