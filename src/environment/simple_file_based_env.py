#!/usr/bin/env python3
"""
Simple File-Based Sonic Environment

This module provides a simplified file-based environment that uses the fixed Lua bridge
for direct input injection. It's more reliable than the complex input sequence approach.
"""

import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List
import time
import os
import json
import subprocess
from pathlib import Path
import threading
import random

class SimpleFileBasedSonicEnvironment(gym.Env):
    """
    Simple file-based Sonic environment for reinforcement learning.
    
    This environment uses the fixed Lua bridge for direct input injection
    and file-based communication for game state reading.
    """
    
    def __init__(self, config: Dict[str, Any], instance_id: int = 0):
        super().__init__()
        
        self.config = config
        self.instance_id = instance_id
        self.game_config = config['game']
        self.env_config = config['environment']
        self.obs_config = config['observation']
        self.action_config = config['actions']
        
        # File-based communication setup
        self.comm_dir = Path(f"bizhawk_comm_{instance_id}")
        self.comm_dir.mkdir(exist_ok=True)
        
        self.request_file = self.comm_dir / "request.txt"
        self.response_file = self.comm_dir / "response.txt"
        self.status_file = self.comm_dir / "status.txt"
        
        # Emulator process
        self.emulator_process = None
        self.bizhawk_dir = config.get('bizhawk_dir', r"C:\Program Files (x86)\BizHawk-2.10-win-x64")
        self.lua_script = config.get('lua_script_path', 'emulator/bizhawk_bridge_fixed.lua')
        self.rom_path = self.game_config['rom_path']
        
        # Environment state
        self.current_step = 0
        self.max_steps = self.game_config['max_steps']
        self.frame_skip = self.game_config['frame_skip']
        self.frame_stack = self.obs_config['frame_stack']
        
        # Frame buffer for temporal information
        self.frame_buffer = []
        
        # Game state tracking
        self.previous_state = None
        self.current_state = None
        
        # Define action space
        self.action_space = self._create_action_space()
        
        # Define observation space
        self.observation_space = self._create_observation_space()
        
        # Action mapping
        self.action_mapping = self._get_action_mapping()
        
        # Initialize emulator
        self._launch_emulator()
        
        # Wait for emulator to be ready
        self._wait_for_ready()
        
    def _create_action_space(self) -> spaces.Discrete:
        """Create the action space."""
        # Define available actions
        actions = self.action_config.get('available_actions', [
            'NOOP', 'LEFT', 'RIGHT', 'UP', 'DOWN', 
            'A', 'B', 'C', 'START', 'SELECT'
        ])
        
        return spaces.Discrete(len(actions))
    
    def _create_observation_space(self) -> spaces.Box:
        """Create the observation space."""
        # Simple observation space based on game state
        # We'll use a combination of position, score, rings, etc.
        obs_size = 10  # x, y, rings, lives, score, zone, act, timer, invincibility, status
        
        if self.frame_stack > 1:
            obs_size *= self.frame_stack
            
        return spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
    
    def _get_action_mapping(self) -> Dict[int, str]:
        """Get the mapping from action indices to action names."""
        actions = self.action_config.get('available_actions', [
            'NOOP', 'LEFT', 'RIGHT', 'UP', 'DOWN', 
            'A', 'B', 'C', 'START', 'SELECT'
        ])
        
        return {i: action for i, action in enumerate(actions)}
    
    def _launch_emulator(self):
        """Launch the BizHawk emulator."""
        if self.emulator_process:
            self._close_emulator()
        
        # Set environment variables
        env = os.environ.copy()
        env['BIZHAWK_INSTANCE_ID'] = str(self.instance_id)
        env['BIZHAWK_COMM_BASE'] = os.getcwd()
        
        # Launch command
        cmd = [
            os.path.join(self.bizhawk_dir, "EmuHawk.exe"),
            f"--lua={self.lua_script}",
            str(self.rom_path)
        ]
        
        print(f"[SimpleFileBasedEnv-{self.instance_id}] Launching BizHawk...")
        self.emulator_process = subprocess.Popen(cmd, env=env)
        
        # Wait a bit for startup
        time.sleep(5)
    
    def _wait_for_ready(self, timeout: int = 30):
        """Wait for the emulator to be ready."""
        print(f"[SimpleFileBasedEnv-{self.instance_id}] Waiting for emulator to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.status_file.exists():
                try:
                    with open(self.status_file, 'r') as f:
                        status = f.read().strip()
                    if status == "READY":
                        print(f"[SimpleFileBasedEnv-{self.instance_id}] Emulator ready!")
                        return True
                except:
                    pass
            time.sleep(1)
        
        print(f"[SimpleFileBasedEnv-{self.instance_id}] Emulator not ready after {timeout} seconds")
        return False
    
    def _send_command(self, command: str) -> Optional[str]:
        """Send a command to the Lua bridge and get response."""
        try:
            # Write request
            with open(self.request_file, 'w') as f:
                f.write(command + "\n")
            
            # Wait for response
            timeout = 2.0
            start_time = time.time()
            
            while not self.response_file.exists():
                if time.time() - start_time > timeout:
                    print(f"[SimpleFileBasedEnv-{self.instance_id}] Timeout waiting for response")
                    return None
                time.sleep(0.01)
            
            # Read response
            with open(self.response_file, 'r') as f:
                response = f.read().strip()
            
            # Clean up response file
            try:
                self.response_file.unlink()
            except:
                pass
            
            return response
            
        except Exception as e:
            print(f"[SimpleFileBasedEnv-{self.instance_id}] Command send error: {e}")
            return None
    
    def _get_game_state(self) -> Dict[str, Any]:
        """Get the current game state from the emulator."""
        response = self._send_command("ACTION:GET_STATE")
        
        if response and "SUCCESS:true" in response:
            # Parse the state data
            data_part = response.split("DATA:")[1] if "DATA:" in response else ""
            state = {}
            
            # Parse state string like "x:123,y:456,rings:5,lives:3,level:1"
            for part in data_part.split(","):
                if ":" in part:
                    key, value = part.split(":", 1)
                    try:
                        state[key] = int(value)
                    except:
                        state[key] = value
            
            return state
        else:
            # Return default state
            return {
                'x': 0, 'y': 0, 'rings': 0, 'lives': 0, 'level': 0
            }
    
    def _send_inputs(self, inputs: List[str]):
        """Send inputs to the emulator."""
        if not inputs:
            return
        
        # Convert inputs to the format expected by the Lua bridge
        input_parts = []
        for input_name in inputs:
            if input_name != 'NOOP':
                input_parts.append(f"{input_name}:true")
        
        if input_parts:
            input_str = "|".join(input_parts)
            command = f"ACTION:SET_INPUTS|INPUTS:{input_str}"
            self._send_command(command)
    
    def _reset_inputs(self):
        """Reset all inputs."""
        self._send_command("ACTION:RESET_INPUTS")
    
    def _process_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """Process the game state into an observation."""
        # Create a simple observation vector
        obs = np.array([
            state.get('x', 0),
            state.get('y', 0),
            state.get('rings', 0),
            state.get('lives', 0),
            state.get('level', 0),
            self.current_step,
            0, 0, 0, 0  # Placeholder for additional features
        ], dtype=np.float32)
        
        return obs
    
    def _update_frame_buffer(self, obs: np.ndarray):
        """Update the frame buffer with a new observation."""
        self.frame_buffer.append(obs)
        if len(self.frame_buffer) > self.frame_stack:
            self.frame_buffer.pop(0)
    
    def _get_stacked_observation(self) -> np.ndarray:
        """Get the stacked observation."""
        if len(self.frame_buffer) < self.frame_stack:
            # Pad with zeros if not enough frames
            padding = [np.zeros_like(self.frame_buffer[0]) for _ in range(self.frame_stack - len(self.frame_buffer))]
            stacked = padding + self.frame_buffer
        else:
            stacked = self.frame_buffer
        
        return np.concatenate(stacked)
    
    def _calculate_reward(self, prev_state: Dict[str, Any], curr_state: Dict[str, Any]) -> float:
        """Calculate the reward based on state changes."""
        reward = 0.0
        
        # Reward for collecting rings
        rings_diff = curr_state.get('rings', 0) - prev_state.get('rings', 0)
        if rings_diff > 0:
            reward += rings_diff * 10.0
        
        # Reward for progress (moving right)
        x_diff = curr_state.get('x', 0) - prev_state.get('x', 0)
        if x_diff > 0:
            reward += x_diff * 0.1
        
        # Penalty for losing lives
        lives_diff = curr_state.get('lives', 0) - prev_state.get('lives', 0)
        if lives_diff < 0:
            reward -= 100.0
        
        # Small penalty for time passing
        reward -= 0.1
        
        return reward
    
    def _is_done(self, state: Dict[str, Any]) -> bool:
        """Check if the episode is done."""
        # Done if no lives left
        if state.get('lives', 0) <= 0:
            return True
        
        # Done if max steps reached
        if self.current_step >= self.max_steps:
            return True
        
        # Done if level completed (you can add more sophisticated logic here)
        if state.get('level', 0) > 1:  # Assuming level 1 is the starting level
            return True
        
        return False
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset emulator
        self._reset_inputs()
        
        # Wait a bit for the game to settle
        time.sleep(1)
        
        # Reset frame buffer
        self.frame_buffer = []
        
        # Get initial state
        initial_state = self._get_game_state()
        self.current_state = initial_state
        self.previous_state = initial_state
        
        # Process observation
        obs = self._process_observation(initial_state)
        
        # Initialize frame buffer
        for _ in range(self.frame_stack):
            self._update_frame_buffer(obs)
        
        # Reset step counter
        self.current_step = 0
        
        # Get stacked observation
        stacked_obs = self._get_stacked_observation()
        
        return stacked_obs, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Convert action to inputs
        action_name = self.action_mapping.get(action, 'NOOP')
        inputs = [action_name] if action_name != 'NOOP' else []
        
        # Send inputs
        self._send_inputs(inputs)
        
        # Wait for frame to process
        time.sleep(0.016)  # ~60 FPS
        
        # Update step counter
        self.current_step += 1
        
        # Get new state
        self.previous_state = self.current_state
        self.current_state = self._get_game_state()
        
        # Calculate reward
        reward = self._calculate_reward(self.previous_state, self.current_state)
        
        # Check if done
        done = self._is_done(self.current_state)
        
        # Process observation
        obs = self._process_observation(self.current_state)
        self._update_frame_buffer(obs)
        
        # Get stacked observation
        stacked_obs = self._get_stacked_observation()
        
        # Reset inputs for next frame
        self._reset_inputs()
        
        return stacked_obs, reward, done, False, {
            'state': self.current_state,
            'action': action_name
        }
    
    def render(self, mode: str = 'human'):
        """Render the environment (not implemented for file-based)."""
        pass
    
    def close(self):
        """Close the environment."""
        self._close_emulator()
    
    def _close_emulator(self):
        """Close the emulator process."""
        if self.emulator_process:
            self.emulator_process.terminate()
            self.emulator_process.wait()
            self.emulator_process = None
    
    def get_action_meanings(self) -> List[str]:
        """Get the meaning of each action."""
        return list(self.action_mapping.values())
