"""
Reward Visualizer
Step 5: Visualizes training reward distribution for monitoring ML convergence.
Saves histogram to logs/reward_hist.png (overwrites per cycle).
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import matplotlib (optional dependency)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


class RewardVisualizer:
    """
    Visualizes reward distribution during training.
    Helps diagnose convergence issues and reward function behavior.
    """
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.reward_history: List[float] = []
        self.epoch_rewards: List[float] = []
        self.cycle_number = 0
    
    def add_rewards(self, rewards: List[float], epoch: int) -> None:
        """Add batch of rewards from an epoch."""
        self.reward_history.extend(rewards)
        if rewards:
            avg_reward = sum(rewards) / len(rewards)
            self.epoch_rewards.append(avg_reward)
    
    def add_epoch_reward(self, reward: float) -> None:
        """Add single epoch total reward."""
        self.epoch_rewards.append(reward)
    
    def set_cycle(self, cycle: int) -> None:
        """Set current cycle number for labeling."""
        self.cycle_number = cycle
    
    def clear(self) -> None:
        """Clear accumulated data for new training session."""
        self.reward_history.clear()
        self.epoch_rewards.clear()
    
    def save_histogram(
        self,
        filename: str = "reward_hist.png",
        title: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate and save reward histogram.
        
        Args:
            filename: Output filename (saved to logs_dir)
            title: Custom title for the plot
            
        Returns:
            Path to saved file, or None if matplotlib unavailable
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if not self.reward_history and not self.epoch_rewards:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        
        # Left: Reward distribution histogram
        ax1 = axes[0]
        if self.reward_history:
            rewards = np.array(self.reward_history)
            
            # Filter out extreme outliers for visualization
            p5, p95 = np.percentile(rewards, [5, 95])
            filtered = rewards[(rewards >= p5) & (rewards <= p95)]
            
            ax1.hist(filtered, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
            ax1.axvline(x=np.mean(rewards), color='green', linestyle='-', linewidth=2, 
                       label=f'Mean: {np.mean(rewards):.4f}')
            ax1.axvline(x=np.median(rewards), color='orange', linestyle='-', linewidth=2,
                       label=f'Median: {np.median(rewards):.4f}')
            
            ax1.set_xlabel('Reward Value')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'Reward Distribution (n={len(rewards):,})')
            ax1.legend(loc='upper right')
            
            # Add statistics text box
            stats_text = (
                f"Mean: {np.mean(rewards):.4f}\n"
                f"Std: {np.std(rewards):.4f}\n"
                f"Min: {np.min(rewards):.4f}\n"
                f"Max: {np.max(rewards):.4f}\n"
                f"% Positive: {100 * np.sum(rewards > 0) / len(rewards):.1f}%"
            )
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax1.text(0.5, 0.5, 'No reward data', ha='center', va='center')
        
        # Right: Epoch reward progression
        ax2 = axes[1]
        if self.epoch_rewards:
            epochs = list(range(1, len(self.epoch_rewards) + 1))
            rewards = self.epoch_rewards
            
            ax2.plot(epochs, rewards, 'b-', linewidth=2, marker='o', markersize=4)
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
            
            # Add trend line
            if len(epochs) > 2:
                z = np.polyfit(epochs, rewards, 1)
                p = np.poly1d(z)
                ax2.plot(epochs, p(epochs), 'g--', linewidth=1, alpha=0.7,
                        label=f'Trend: {z[0]:.4f}/epoch')
                ax2.legend(loc='upper left')
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Total Reward')
            ax2.set_title('Reward Progression by Epoch')
            ax2.grid(True, alpha=0.3)
            
            # Mark best epoch
            best_idx = np.argmax(rewards)
            ax2.scatter([epochs[best_idx]], [rewards[best_idx]], 
                       color='gold', s=100, zorder=5, 
                       label=f'Best: {rewards[best_idx]:.4f}')
        else:
            ax2.text(0.5, 0.5, 'No epoch data', ha='center', va='center')
        
        # Overall title
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        else:
            fig.suptitle(
                f'AML HFT Reward Analysis - Cycle {self.cycle_number} | {timestamp} UTC',
                fontsize=14, fontweight='bold'
            )
        
        plt.tight_layout()
        
        # Save to file
        filepath = self.logs_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return str(filepath)
    
    def save_json(self, filename: str = "reward_data.json") -> str:
        """Save raw reward data to JSON for further analysis."""
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": self.cycle_number,
            "epoch_rewards": self.epoch_rewards,
            "reward_stats": {
                "count": len(self.reward_history),
                "mean": float(np.mean(self.reward_history)) if self.reward_history else 0,
                "std": float(np.std(self.reward_history)) if self.reward_history else 0,
                "min": float(np.min(self.reward_history)) if self.reward_history else 0,
                "max": float(np.max(self.reward_history)) if self.reward_history else 0,
                "pct_positive": float(100 * np.sum(np.array(self.reward_history) > 0) / len(self.reward_history)) if self.reward_history else 0
            }
        }
        
        filepath = self.logs_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)

    def save_return_history(
        self,
        returns: List[float],
        dates: Optional[List[str]] = None,
        filename: str = "return_hist.png"
    ) -> Optional[str]:
        """
        Step 5: Generate and save return history plot for monitoring.
        
        Args:
            returns: List of cumulative returns
            dates: Optional list of date labels
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if not HAS_MATPLOTLIB or not returns:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_vals = range(len(returns))
        ax.plot(x_vals, returns, color='steelblue', linewidth=2, label='Cumulative Return')
        ax.fill_between(x_vals, 0, returns, alpha=0.3, color='steelblue')
        
        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        
        # Add 5% target line
        if len(returns) > 0:
            ax.axhline(y=5.0, color='green', linestyle='--', linewidth=2, label='5% Target')
        
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.set_title('AML Return History - Step 5 Monitoring', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        filepath = self.logs_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        return str(filepath)


# Singleton instance for easy access
_visualizer: Optional[RewardVisualizer] = None


def get_visualizer() -> RewardVisualizer:
    """Get or create singleton visualizer instance."""
    global _visualizer
    if _visualizer is None:
        _visualizer = RewardVisualizer()
    return _visualizer


if __name__ == "__main__":
    # Demo/test mode
    import random
    
    viz = RewardVisualizer()
    viz.set_cycle(1)
    
    # Generate sample data
    print("Generating sample reward data...")
    for epoch in range(20):
        # Simulate improving rewards
        base_reward = -0.5 + (epoch * 0.05)
        rewards = [base_reward + random.gauss(0, 0.1) for _ in range(100)]
        viz.add_rewards(rewards, epoch)
        viz.add_epoch_reward(sum(rewards))
    
    # Save outputs
    hist_path = viz.save_histogram()
    json_path = viz.save_json()
    
    print(f"Saved histogram to: {hist_path}")
    print(f"Saved JSON data to: {json_path}")
