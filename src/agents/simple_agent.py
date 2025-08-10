import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional
import random


class SimpleCNN(nn.Module):
    """
    Simple CNN for processing Sonic game observations.
    """
    
    def __init__(self, input_channels: int = 1, num_actions: int = 9):
        super(SimpleCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class SimpleAgent:
    """
    Simple reinforcement learning agent for Sonic.
    
    This is a basic implementation that can be used for testing
    and as a baseline for more sophisticated agents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network parameters
        self.input_channels = config.get('frame_stack', 1)
        self.num_actions = 9  # Basic Sonic actions
        self.learning_rate = config['agent']['learning_rate']
        
        # Create network
        self.network = SimpleCNN(self.input_channels, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # Training parameters
        self.gamma = config['agent']['gamma']
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Experience replay
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 32
        
        # Training state
        self.training_step = 0
        
    def preprocess_observation(self, obs: np.ndarray) -> torch.Tensor:
        """Preprocess observation for the network."""
        # Convert to tensor
        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, axis=0)  # Add channel dimension
        
        # Normalize
        obs = obs.astype(np.float32) / 255.0
        
        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        return obs_tensor
    
    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        """Select an action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            # Random action
            return random.randint(0, self.num_actions - 1)
        
        # Network action
        obs_tensor = self.preprocess_observation(obs)
        with torch.no_grad():
            q_values = self.network(obs_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def store_experience(self, obs: np.ndarray, action: int, reward: float, 
                        next_obs: np.ndarray, done: bool):
        """Store experience in replay memory."""
        experience = (obs.copy(), action, reward, next_obs.copy(), done)
        self.memory.append(experience)
        
        # Remove old experiences if memory is full
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def train(self):
        """Train the agent on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
        
        # Convert to tensors
        obs_batch = np.array(obs_batch)
        next_obs_batch = np.array(next_obs_batch)
        # Ensure shape is (batch, channels, height, width)
        if len(obs_batch.shape) == 4 and obs_batch.shape[-1] == 1:
            # (batch, height, width, 1) -> (batch, 1, height, width)
            obs_batch = np.transpose(obs_batch, (0, 3, 1, 2))
            next_obs_batch = np.transpose(next_obs_batch, (0, 3, 1, 2))
        elif len(obs_batch.shape) == 3:
            # (batch, height, width) -> (batch, 1, height, width)
            obs_batch = obs_batch[:, None, :, :]
            next_obs_batch = next_obs_batch[:, None, :, :]
        
        obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
        action_tensor = torch.LongTensor(action_batch).to(self.device)
        reward_tensor = torch.FloatTensor(reward_batch).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs_batch).to(self.device)
        done_tensor = torch.BoolTensor(done_batch).to(self.device)
        
        # Normalize observations
        obs_tensor = obs_tensor / 255.0
        next_obs_tensor = next_obs_tensor / 255.0
        
        # Current Q values
        current_q_values = self.network(obs_tensor)
        current_q = current_q_values.gather(1, action_tensor.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.network(next_obs_tensor)
            next_q = next_q_values.max(1)[0]
            target_q = reward_tensor + (self.gamma * next_q * ~done_tensor)
        
        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_step += 1
        
        return loss.item()
    
    def save(self, path: str):
        """Save the agent model."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load the agent model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']


class SonicHeuristicAgent:
    """
    Heuristic agent that uses simple rules to play Sonic.
    
    This agent implements basic Sonic gameplay strategies:
    - Move right to progress
    - Jump over obstacles
    - Collect rings
    - Avoid enemies
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.last_action = 0
        self.jump_cooldown = 0
        self.stuck_counter = 0
        self.last_position = (0, 0)
        
    def select_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        """Select action based on heuristic rules."""
        # Get current position
        current_pos = info.get('position', (0, 0))
        
        # Check if stuck
        if current_pos == self.last_position:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        self.last_position = current_pos
        
        # Decrease jump cooldown
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
        
        # Basic strategy: move right and jump when needed
        action = 2  # RIGHT
        
        # Jump if there's an obstacle ahead or if stuck
        if self._should_jump(obs, info):
            action = 5  # A (Jump)
            self.jump_cooldown = 10
        
        # Spin dash if moving slowly and on ground
        if self._should_spin_dash(info):
            action = 6  # B (Spin dash)
        
        # Avoid enemies
        if self._should_avoid_enemy(obs, info):
            action = 5  # A (Jump)
        
        self.last_action = action
        return action
    
    def _should_jump(self, obs: np.ndarray, info: Dict[str, Any]) -> bool:
        """Determine if Sonic should jump."""
        # Jump if stuck for too long
        if self.stuck_counter > 20:
            return True
        
        # Jump if there's an obstacle ahead (simplified detection)
        # This would require more sophisticated obstacle detection
        return False
    
    def _should_spin_dash(self, info: Dict[str, Any]) -> bool:
        """Determine if Sonic should spin dash."""
        velocity = info.get('velocity', (0, 0))
        position = info.get('position', (0, 0))
        
        # Spin dash if moving slowly and on ground
        if abs(velocity[0]) < 2 and velocity[1] == 0:
            return True
        
        return False
    
    def _should_avoid_enemy(self, obs: np.ndarray, info: Dict[str, Any]) -> bool:
        """Determine if Sonic should avoid an enemy."""
        # This would require enemy detection from the observation
        # For now, use a simple heuristic
        return False


class SonicRandomAgent:
    """
    Random agent for baseline comparison.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_actions = 9
    
    def select_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        """Select a random action."""
        return random.randint(0, self.num_actions - 1)
    
    def train(self):
        """No training for random agent."""
        pass
    
    def save(self, path: str):
        """No saving for random agent."""
        pass
    
    def load(self, path: str):
        """No loading for random agent."""
        pass 