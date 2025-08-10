import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2
from typing import Dict, Any, List, Optional
import time
import threading
from collections import deque
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TrainingVisualizer:
    """
    Real-time visualization of Sonic AI training.
    
    Features:
    - Live game screen display
    - Training metrics plots
    - Reward tracking
    - Performance statistics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fig = None
        self.axes = None
        self.animation = None
        
        # Data storage
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.episode_scores = deque(maxlen=1000)
        self.training_rewards = deque(maxlen=1000)
        self.training_losses = deque(maxlen=1000)
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Threading for real-time updates
        self.running = False
        self.update_thread = None
        
    def start(self):
        """Start the visualization."""
        self.running = True
        self._setup_plots()
        self._start_animation()
        
    def stop(self):
        """Stop the visualization."""
        self.running = False
        if self.animation:
            self.animation.event_source.stop()
        if self.update_thread:
            self.update_thread.join()
        plt.close('all')
    
    def _setup_plots(self):
        """Set up the matplotlib figure and subplots."""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('Sonic AI Training Visualization', fontsize=16, fontweight='bold')
        
        # Configure subplots
        self.axes[0, 0].set_title('Live Game Screen')
        self.axes[0, 0].set_xticks([])
        self.axes[0, 0].set_yticks([])
        
        self.axes[0, 1].set_title('Episode Rewards')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].set_ylabel('Reward')
        
        self.axes[0, 2].set_title('Training Progress')
        self.axes[0, 2].set_xlabel('Step')
        self.axes[0, 2].set_ylabel('Reward')
        
        self.axes[1, 0].set_title('Episode Lengths')
        self.axes[1, 0].set_xlabel('Episode')
        self.axes[1, 0].set_ylabel('Steps')
        
        self.axes[1, 1].set_title('Scores')
        self.axes[1, 1].set_xlabel('Episode')
        self.axes[1, 1].set_ylabel('Score')
        
        self.axes[1, 2].set_title('Performance Stats')
        self.axes[1, 2].set_xlabel('Metric')
        self.axes[1, 2].set_ylabel('Value')
        
        plt.tight_layout()
    
    def _start_animation(self):
        """Start the matplotlib animation."""
        self.animation = animation.FuncAnimation(
            self.fig, self._update_plots, interval=100, blit=False
        )
        plt.show(block=False)
    
    def _update_plots(self, frame):
        """Update all plots with current data."""
        if not self.running:
            return
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Update game screen (placeholder)
        self._update_game_screen()
        
        # Update reward plots
        self._update_reward_plots()
        
        # Update performance plots
        self._update_performance_plots()
        
        # Update stats
        self._update_stats()
        
        # Redraw
        self.fig.canvas.draw()
    
    def _update_game_screen(self):
        """Update the live game screen display."""
        ax = self.axes[0, 0]
        ax.set_title('Live Game Screen')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Placeholder for game screen
        # In practice, this would display the current game frame
        placeholder = np.random.randint(0, 255, (224, 256, 3), dtype=np.uint8)
        ax.imshow(placeholder)
    
    def _update_reward_plots(self):
        """Update reward-related plots."""
        # Episode rewards
        ax = self.axes[0, 1]
        if self.episode_rewards:
            episodes = range(len(self.episode_rewards))
            ax.plot(episodes, list(self.episode_rewards), 'b-', alpha=0.7)
            ax.set_title('Episode Rewards')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
        
        # Training rewards
        ax = self.axes[0, 2]
        if self.training_rewards:
            steps = range(len(self.training_rewards))
            ax.plot(steps, list(self.training_rewards), 'g-', alpha=0.7)
            ax.set_title('Training Progress')
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
    
    def _update_performance_plots(self):
        """Update performance-related plots."""
        # Episode lengths
        ax = self.axes[1, 0]
        if self.episode_lengths:
            episodes = range(len(self.episode_lengths))
            ax.plot(episodes, list(self.episode_lengths), 'r-', alpha=0.7)
            ax.set_title('Episode Lengths')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Steps')
            ax.grid(True, alpha=0.3)
        
        # Scores
        ax = self.axes[1, 1]
        if self.episode_scores:
            episodes = range(len(self.episode_scores))
            ax.plot(episodes, list(self.episode_scores), 'm-', alpha=0.7)
            ax.set_title('Scores')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
    
    def _update_stats(self):
        """Update performance statistics."""
        ax = self.axes[1, 2]
        ax.set_title('Performance Stats')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        
        # Calculate statistics
        stats = []
        labels = []
        
        if self.episode_rewards:
            stats.extend([
                np.mean(list(self.episode_rewards)),
                np.max(list(self.episode_rewards)),
                np.min(list(self.episode_rewards))
            ])
            labels.extend(['Avg Reward', 'Max Reward', 'Min Reward'])
        
        if self.episode_lengths:
            stats.extend([
                np.mean(list(self.episode_lengths)),
                np.max(list(self.episode_lengths))
            ])
            labels.extend(['Avg Length', 'Max Length'])
        
        if self.episode_scores:
            stats.extend([
                np.mean(list(self.episode_scores)),
                np.max(list(self.episode_scores))
            ])
            labels.extend(['Avg Score', 'Max Score'])
        
        # Add FPS
        stats.append(self.current_fps)
        labels.append('FPS')
        
        if stats:
            bars = ax.bar(labels, stats, color=['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray'])
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, stat in zip(bars, stats):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{stat:.1f}', ha='center', va='bottom')
    
    def update_game_screen(self, screen: np.ndarray):
        """Update the live game screen."""
        # This would be called from the training loop
        # to display the current game state
        pass
    
    def add_episode_data(self, reward: float, length: int, score: int):
        """Add episode data for visualization."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_scores.append(score)
    
    def add_training_data(self, reward: float, loss: Optional[float] = None):
        """Add training step data."""
        self.training_rewards.append(reward)
        if loss is not None:
            self.training_losses.append(loss)
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time


class SonicGameVisualizer:
    """
    Specialized visualizer for Sonic game-specific features.
    
    Features:
    - Sonic character tracking
    - Ring collection visualization
    - Enemy detection display
    - Level progress tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fig = None
        self.axes = None
        
        # Sonic-specific data
        self.sonic_positions = deque(maxlen=1000)
        self.ring_counts = deque(maxlen=1000)
        self.enemy_positions = deque(maxlen=1000)
        self.level_progress = deque(maxlen=1000)
        
    def setup_sonic_plots(self):
        """Set up Sonic-specific visualization plots."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Sonic AI - Game Analysis', fontsize=16, fontweight='bold')
        
        # Sonic movement tracking
        self.axes[0, 0].set_title('Sonic Movement Path')
        self.axes[0, 0].set_xlabel('X Position')
        self.axes[0, 0].set_ylabel('Y Position')
        
        # Ring collection over time
        self.axes[0, 1].set_title('Ring Collection')
        self.axes[0, 1].set_xlabel('Time')
        self.axes[0, 1].set_ylabel('Rings')
        
        # Level progress
        self.axes[1, 0].set_title('Level Progress')
        self.axes[1, 0].set_xlabel('Time')
        self.axes[1, 0].set_ylabel('Progress %')
        
        # Enemy encounters
        self.axes[1, 1].set_title('Enemy Encounters')
        self.axes[1, 1].set_xlabel('Time')
        self.axes[1, 1].set_ylabel('Enemies Defeated')
        
        plt.tight_layout()
    
    def update_sonic_position(self, x: float, y: float):
        """Update Sonic's position tracking."""
        self.sonic_positions.append((x, y))
    
    def update_ring_count(self, rings: int):
        """Update ring count tracking."""
        self.ring_counts.append(rings)
    
    def update_enemy_position(self, x: float, y: float):
        """Update enemy position tracking."""
        self.enemy_positions.append((x, y))
    
    def update_level_progress(self, progress: float):
        """Update level progress tracking."""
        self.level_progress.append(progress)
    
    def plot_sonic_analysis(self):
        """Create comprehensive Sonic game analysis plots."""
        if not self.fig:
            self.setup_sonic_plots()
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot Sonic movement path
        if self.sonic_positions:
            positions = list(self.sonic_positions)
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            self.axes[0, 0].plot(x_coords, y_coords, 'b-', alpha=0.7, linewidth=2)
            self.axes[0, 0].scatter(x_coords[-1], y_coords[-1], c='red', s=100, zorder=5)
            self.axes[0, 0].set_title('Sonic Movement Path')
            self.axes[0, 0].set_xlabel('X Position')
            self.axes[0, 0].set_ylabel('Y Position')
            self.axes[0, 0].grid(True, alpha=0.3)
        
        # Plot ring collection
        if self.ring_counts:
            times = range(len(self.ring_counts))
            self.axes[0, 1].plot(times, list(self.ring_counts), 'g-', linewidth=2)
            self.axes[0, 1].set_title('Ring Collection')
            self.axes[0, 1].set_xlabel('Time')
            self.axes[0, 1].set_ylabel('Rings')
            self.axes[0, 1].grid(True, alpha=0.3)
        
        # Plot level progress
        if self.level_progress:
            times = range(len(self.level_progress))
            self.axes[1, 0].plot(times, list(self.level_progress), 'r-', linewidth=2)
            self.axes[1, 0].set_title('Level Progress')
            self.axes[1, 0].set_xlabel('Time')
            self.axes[1, 0].set_ylabel('Progress %')
            self.axes[1, 0].grid(True, alpha=0.3)
        
        # Plot enemy encounters
        if self.enemy_positions:
            times = range(len(self.enemy_positions))
            enemy_counts = [i for i in range(len(self.enemy_positions))]
            self.axes[1, 1].plot(times, enemy_counts, 'm-', linewidth=2)
            self.axes[1, 1].set_title('Enemy Encounters')
            self.axes[1, 1].set_xlabel('Time')
            self.axes[1, 1].set_ylabel('Enemies Defeated')
            self.axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)


class TrainingMetricsLogger:
    """
    Log training metrics for later analysis.
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_scores': [],
            'training_rewards': [],
            'training_losses': [],
            'sonic_positions': [],
            'ring_counts': [],
            'enemy_positions': [],
            'level_progress': []
        }
    
    def log_episode(self, reward: float, length: int, score: int):
        """Log episode data."""
        self.metrics['episode_rewards'].append(reward)
        self.metrics['episode_lengths'].append(length)
        self.metrics['episode_scores'].append(score)
    
    def log_training_step(self, reward: float, loss: Optional[float] = None):
        """Log training step data."""
        self.metrics['training_rewards'].append(reward)
        if loss is not None:
            self.metrics['training_losses'].append(loss)
    
    def log_sonic_data(self, position: tuple, rings: int, enemies: list, progress: float):
        """Log Sonic-specific game data."""
        self.metrics['sonic_positions'].append(position)
        self.metrics['ring_counts'].append(rings)
        self.metrics['enemy_positions'].append(enemies)
        self.metrics['level_progress'].append(progress)
    
    def save_metrics(self, filename: str = 'training_metrics.json'):
        """Save metrics to file."""
        import json
        import numpy as np
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                serializable_metrics[key] = [v.tolist() for v in value]
            else:
                serializable_metrics[key] = value
        
        with open(os.path.join(self.log_dir, filename), 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def load_metrics(self, filename: str = 'training_metrics.json'):
        """Load metrics from file."""
        import json
        
        filepath = os.path.join(self.log_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.metrics = json.load(f) 