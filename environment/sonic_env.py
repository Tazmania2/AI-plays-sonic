import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple
import time
import os
from pathlib import Path
import random

from emulator.sonic_emulator import SonicEmulator
from utils.reward_calculator import RewardCalculator
from utils.observation_processor import ObservationProcessor


class SonicEnvironment(gym.Env):
    """
    Sonic the Hedgehog environment for reinforcement learning.
    
    This environment wraps a Sonic emulator and provides a gym-like interface
    for training RL agents to play Sonic games.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.game_config = config['game']
        self.env_config = config['environment']
        self.obs_config = config['observation']
        self.action_config = config['actions']
        
        # Initialize emulator
        emulator_config = config.get('emulator', {})
        core_path = emulator_config.get('core', 'genesis_plus_gx_libretro.dll')
        
        self.emulator = SonicEmulator(
            rom_path=self.game_config['rom_path'],
            screen_width=self.env_config['screen_width'],
            screen_height=self.env_config['screen_height'],
            core_path=core_path
        )
        
        # Initialize processors
        self.obs_processor = ObservationProcessor(self.obs_config)
        self.reward_calculator = RewardCalculator(config['rewards'])
        
        # Environment state
        self.current_step = 0
        self.max_steps = self.game_config['max_steps']
        self.frame_skip = self.game_config['frame_skip']
        self.frame_stack = self.obs_config['frame_stack']
        
        # Frame buffer for temporal information
        self.frame_buffer = []
        
        # Define action space
        self.action_space = self._create_action_space()
        
        # Define observation space
        self.observation_space = self._create_observation_space()
        
        # Game state tracking
        self.previous_state = None
        self.current_state = None
        
    def _create_action_space(self) -> spaces.Discrete:
        """Create the action space based on available actions."""
        # Basic actions: NOOP, LEFT, RIGHT, UP, DOWN, A, B, START, SELECT
        # Plus combinations: LEFT+A, RIGHT+A, DOWN+B, LEFT+DOWN+B, RIGHT+DOWN+B
        basic_actions = self.action_config.get('basic', [])
        combinations = self.action_config.get('combinations', [])
        num_actions = len(basic_actions) + len(combinations)
        return spaces.Discrete(num_actions)
    
    def _create_observation_space(self) -> spaces.Box:
        """Create the observation space."""
        height, width = self.obs_config['resize']
        channels = 1 if self.env_config['grayscale'] else 3
        if self.frame_stack > 1:
            channels *= self.frame_stack
            
        return spaces.Box(
            low=0,
            high=255,
            shape=(channels, height, width),  # Stable Baselines 3 expects (C, H, W)
            dtype=np.uint8
        )
    
    def _get_action_mapping(self, action: int) -> list:
        """Convert discrete action to emulator action mapping."""
        basic_actions = self.action_config.get('basic', [])
        combinations = self.action_config.get('combinations', [])
        
        if action < len(basic_actions):
            return [basic_actions[action]]
        else:
            combo_idx = action - len(basic_actions)
            return combinations[combo_idx]
    
    def _process_observation(self, screen: np.ndarray) -> np.ndarray:
        """Process raw screen observation."""
        return self.obs_processor.process(screen)
    
    def _update_frame_buffer(self, obs: np.ndarray):
        """Update the frame buffer for temporal information."""
        self.frame_buffer.append(obs)
        if len(self.frame_buffer) > self.frame_stack:
            self.frame_buffer.pop(0)
    
    def _get_stacked_observation(self) -> np.ndarray:
        """Get observation with stacked frames."""
        if self.frame_stack == 1:
            # Convert from (H, W, C) to (C, H, W)
            obs = self.frame_buffer[-1]
            return np.transpose(obs, (2, 0, 1))
        
        # Pad with zeros if not enough frames
        while len(self.frame_buffer) < self.frame_stack:
            self.frame_buffer.insert(0, np.zeros_like(self.frame_buffer[0]))
        
        # Stack frames along channel dimension
        stacked = np.concatenate(self.frame_buffer, axis=-1)
        # Convert from (H, W, C) to (C, H, W)
        return np.transpose(stacked, (2, 0, 1))
    
    def _get_game_state(self) -> Dict[str, Any]:
        """Extract game state information from emulator memory."""
        return self.emulator.get_game_state()
    
    def _calculate_reward(self, prev_state: Dict[str, Any], 
                         curr_state: Dict[str, Any]) -> float:
        """Calculate reward based on state changes."""
        return self.reward_calculator.calculate_reward(prev_state, curr_state)
    
    def _is_done(self, state: Dict[str, Any]) -> bool:
        """Check if episode is done."""
        # Game over conditions
        if state.get('lives', 0) <= 0:
            return True
        
        # Level completed
        if state.get('level_completed', False):
            return True
        
        # Time limit
        if self.current_step >= self.max_steps:
            return True
        
        # Stuck detection (optional)
        if self._is_stuck(state):
            return True
        
        return False
    
    def _is_stuck(self, state: Dict[str, Any]) -> bool:
        """Detect if agent is stuck (optional feature)."""
        # Simple stuck detection based on position
        if hasattr(self, 'last_positions'):
            if len(self.last_positions) >= 50:  # Check last 50 steps
                # If position hasn't changed significantly
                recent_positions = self.last_positions[-50:]
                if len(set(recent_positions)) < 5:  # Less than 5 unique positions
                    return True
        return False
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset emulator
        self.emulator.reset()
        
        # Randomize number of START presses between 3 and 10
        n_start_presses = random.randint(3, 10)
        for _ in range(n_start_presses):
            self.emulator.step(['START'])
            time.sleep(0.15)
            self.emulator.step(['NOOP'])
            time.sleep(0.05)
        
        # Reset environment state
        self.current_step = 0
        self.frame_buffer = []
        self.last_positions = []
        
        # Get initial observation
        screen = self.emulator.get_screen()
        obs = self._process_observation(screen)
        self._update_frame_buffer(obs)
        
        # Get initial game state
        self.current_state = self._get_game_state()
        self.previous_state = self.current_state.copy()
        
        # Wait for game to stabilize
        for _ in range(30):  # Wait 30 frames
            self.emulator.step(['NOOP'])
            time.sleep(0.016)  # ~60 FPS
        
        return self._get_stacked_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Convert action to emulator actions
        emulator_actions = self._get_action_mapping(action)
        
        # Execute actions with frame skipping
        for _ in range(self.frame_skip):
            self.emulator.step(emulator_actions)
            time.sleep(0.016)  # ~60 FPS
        
        # Update step counter
        self.current_step += 1
        
        # Get new observation
        screen = self.emulator.get_screen()
        obs = self._process_observation(screen)
        self._update_frame_buffer(obs)
        
        # Update game state
        self.previous_state = self.current_state
        self.current_state = self._get_game_state()
        
        # Calculate reward
        reward = self._calculate_reward(self.previous_state, self.current_state)
        
        # Check if done
        done = self._is_done(self.current_state)
        
        # Update position tracking for stuck detection
        if 'position' in self.current_state:
            self.last_positions.append(self.current_state['position'])
            if len(self.last_positions) > 100:
                self.last_positions.pop(0)
        
        # Prepare info
        info = {
            'step': self.current_step,
            'score': self.current_state.get('score', 0),
            'rings': self.current_state.get('rings', 0),
            'lives': self.current_state.get('lives', 0),
            'level': self.current_state.get('level', 0),
            'position': self.current_state.get('position', (0, 0)),
            'game_state': self.current_state
        }
        
        return self._get_stacked_observation(), reward, done, False, info
    
    def render(self, mode: str = 'human'):
        """Render the current game state."""
        if mode == 'human':
            screen = self.emulator.get_screen()
            cv2.imshow('Sonic AI', screen)
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return self.emulator.get_screen()
    
    def close(self):
        """Close the environment."""
        self.emulator.close()
        cv2.destroyAllWindows()
    
    def get_action_meanings(self) -> list:
        """Get the meaning of each action."""
        basic_actions = self.action_config.get('basic', [])
        combinations = self.action_config.get('combinations', [])
        return basic_actions + [f"COMBO_{i}" for i in range(len(combinations))]
    
    def save_state(self, path: str):
        """Save the current game state."""
        self.emulator.save_state(path)
    
    def load_state(self, path: str):
        """Load a saved game state."""
        self.emulator.load_state(path) 