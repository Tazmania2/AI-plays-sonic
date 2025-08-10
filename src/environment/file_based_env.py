#!/usr/bin/env python3
"""
File-Based Sonic Environment

This module provides a file-based alternative to the standard Sonic environment.
Instead of direct memory reading and input injection, it uses a file-based
communication system where:
1. AI writes commands to a text file
2. Emulator reads the file and executes commands
3. Emulator writes game state logs to another file
4. AI reads the logs and generates new commands

This approach is more robust and can work around input isolation issues.
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

class FileBasedSonicEnvironment(gym.Env):
    """
    File-based Sonic environment for reinforcement learning.
    
    This environment uses file-based communication instead of direct memory access
    and input injection. It's more robust and can work around input isolation issues.
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
        
        self.input_file = self.comm_dir / "ai_inputs.txt"
        self.log_file = self.comm_dir / "game_log.txt"
        self.status_file = self.comm_dir / "status.txt"
        self.completion_file = self.comm_dir / "execution_complete.txt"
        
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
        # Simple observation space for file-based approach
        # We'll use basic game state info instead of screen pixels
        obs_size = 10  # Basic state info: x, y, rings, lives, score, etc.
        
        if self.frame_stack > 1:
            obs_size *= self.frame_stack
            
        return spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
    
    def _get_action_mapping(self) -> Dict[int, str]:
        """Get mapping from action indices to action names."""
        actions = self.action_config.get('available_actions', [
            'NOOP', 'LEFT', 'RIGHT', 'UP', 'DOWN', 
            'A', 'B', 'C', 'START', 'SELECT'
        ])
        
        return {i: action for i, action in enumerate(actions)}
    
    def _launch_emulator(self):
        """Launch the BizHawk emulator with the input player script."""
        if self.emulator_process:
            self._close_emulator()
        
        # Set environment variable for instance ID
        env = os.environ.copy()
        env['BIZHAWK_INSTANCE_ID'] = str(self.instance_id)
        env['BIZHAWK_COMM_BASE'] = str(Path.cwd())
        
        # Create command
        cmd = [
            os.path.join(self.bizhawk_dir, "EmuHawk.exe"),
            f"--lua={self.lua_script}",
            self.rom_path
        ]
        
        print(f"[FileBasedEnv-{self.instance_id}] Launching emulator: {' '.join(cmd)}")
        
        # Launch emulator
        self.emulator_process = subprocess.Popen(
            cmd, 
            cwd=str(Path.cwd()),
            env=env
        )
        
        print(f"[FileBasedEnv-{self.instance_id}] Emulator launched (PID: {self.emulator_process.pid})")
    
    def _wait_for_ready(self, timeout: int = 30):
        """Wait for emulator to be ready."""
        print(f"[FileBasedEnv-{self.instance_id}] Waiting for emulator to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.status_file.exists():
                try:
                    status = self.status_file.read_text().strip()
                    if status == "READY":
                        print(f"[FileBasedEnv-{self.instance_id}] Emulator ready!")
                        return
                except Exception as e:
                    print(f"[FileBasedEnv-{self.instance_id}] Error reading status: {e}")
            
            time.sleep(1)
        
        raise TimeoutError(f"Emulator not ready after {timeout} seconds")
    
    def _write_input_sequence(self, actions: List[str], max_frames: int = 300):
        """Write input sequence to file."""
        input_sequence = []
        current_frame = 0
        
        for action in actions:
            # Add some randomness to frame timing
            frame_delay = random.randint(5, 15)
            current_frame += frame_delay
            
            if current_frame > max_frames:
                break
            
            # Format: "FRAME:ACTION"
            input_sequence.append(f"{current_frame}:{action}")
        
        # Write to input file
        with open(self.input_file, 'w') as f:
            for line in input_sequence:
                f.write(line + '\n')
        
        print(f"[FileBasedEnv-{self.instance_id}] Wrote {len(input_sequence)} inputs")
    
    def _wait_for_execution_complete(self, timeout: int = 60) -> bool:
        """Wait for input execution to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.completion_file.exists():
                try:
                    # Read completion status
                    status = self.completion_file.read_text().strip()
                    if status == "COMPLETE":
                        # Remove completion file
                        self.completion_file.unlink()
                        return True
                except Exception as e:
                    print(f"[FileBasedEnv-{self.instance_id}] Error reading completion: {e}")
            
            time.sleep(0.1)
        
        return False
    
    def _read_game_log(self) -> List[Dict[str, Any]]:
        """Read game state log from file."""
        if not self.log_file.exists():
            return []
        
        try:
            game_states = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            state = json.loads(line)
                            game_states.append(state)
                        except json.JSONDecodeError:
                            continue
            
            return game_states
        except Exception as e:
            print(f"[FileBasedEnv-{self.instance_id}] Error reading game log: {e}")
            return []
    
    def _process_observation(self, game_states: List[Dict[str, Any]]) -> np.ndarray:
        """Process game states into observation array."""
        if not game_states:
            # Return zero observation if no game states
            obs_size = 10
            if self.frame_stack > 1:
                obs_size *= self.frame_stack
            return np.zeros(obs_size, dtype=np.float32)
        
        # Use the last game state
        last_state = game_states[-1]
        
        # Extract basic features
        obs = np.array([
            last_state.get('x', 0),
            last_state.get('y', 0),
            last_state.get('rings', 0),
            last_state.get('lives', 0),
            last_state.get('score', 0),
            last_state.get('level', 0),
            last_state.get('act', 0),
            last_state.get('timer', 0),
            last_state.get('invincibility', 0),
            last_state.get('status', 0)
        ], dtype=np.float32)
        
        # Update frame buffer
        self._update_frame_buffer(obs)
        
        # Return stacked observation
        return self._get_stacked_observation()
    
    def _update_frame_buffer(self, obs: np.ndarray):
        """Update the frame buffer for temporal information."""
        self.frame_buffer.append(obs.copy())
        
        # Keep only the last frame_stack observations
        if len(self.frame_buffer) > self.frame_stack:
            self.frame_buffer.pop(0)
    
    def _get_stacked_observation(self) -> np.ndarray:
        """Get stacked observation from frame buffer."""
        if len(self.frame_buffer) < self.frame_stack:
            # Pad with zeros if not enough frames
            padding = self.frame_stack - len(self.frame_buffer)
            padded_frames = [np.zeros_like(self.frame_buffer[0]) for _ in range(padding)]
            frames = padded_frames + self.frame_buffer
        else:
            frames = self.frame_buffer
        
        # Concatenate frames
        return np.concatenate(frames)
    
    def _calculate_reward(self, prev_state: Dict[str, Any], curr_state: Dict[str, Any]) -> float:
        """Calculate reward based on state changes."""
        if not prev_state or not curr_state:
            return 0.0
        
        reward = 0.0
        
        # Position progress
        prev_x = prev_state.get('x', 0)
        curr_x = curr_state.get('x', 0)
        if curr_x > prev_x:
            reward += 1.0  # Moving right is good
        
        # Ring collection
        prev_rings = prev_state.get('rings', 0)
        curr_rings = curr_state.get('rings', 0)
        if curr_rings > prev_rings:
            reward += 10.0  # Collecting rings is good
        
        # Score increase
        prev_score = prev_state.get('score', 0)
        curr_score = curr_state.get('score', 0)
        if curr_score > prev_score:
            reward += 5.0  # Score increase is good
        
        # Life loss penalty
        prev_lives = prev_state.get('lives', 0)
        curr_lives = curr_state.get('lives', 0)
        if curr_lives < prev_lives:
            reward -= 50.0  # Losing life is bad
        
        # Stuck penalty (no movement)
        if curr_x == prev_x:
            reward -= 0.1  # Small penalty for not moving
        
        return reward
    
    def _is_done(self, state: Dict[str, Any]) -> bool:
        """Check if episode is done."""
        if not state:
            return False
        
        # Check for game over
        lives = state.get('lives', 0)
        if lives <= 0:
            return True
        
        # Check for level completion
        level = state.get('level', 0)
        if level > 0:  # Assuming level 0 is title screen
            return True
        
        # Check for timeout
        timer = state.get('timer', 0)
        if timer > 600:  # 10 minutes timeout
            return True
        
        return False
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        print(f"[FileBasedEnv-{self.instance_id}] Resetting environment...")
        
        # Clear previous state
        self.current_step = 0
        self.frame_buffer = []
        self.previous_state = None
        self.current_state = None
        
        # Clear files
        if self.input_file.exists():
            self.input_file.unlink()
        if self.log_file.exists():
            self.log_file.unlink()
        if self.completion_file.exists():
            self.completion_file.unlink()
        
        # Send reset command
        self._write_input_sequence(['RESET'])
        
        # Wait for reset to complete
        time.sleep(2)
        
        # Get initial observation
        game_states = self._read_game_log()
        obs = self._process_observation(game_states)
        
        info = {
            'episode': 0,
            'step': 0,
            'game_states': game_states
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Convert action to action name
        action_name = self.action_mapping.get(action, 'NOOP')
        
        # Write action to input file
        self._write_input_sequence([action_name])
        
        # Wait for execution to complete
        if not self._wait_for_execution_complete():
            print(f"[FileBasedEnv-{self.instance_id}] Warning: Execution timeout")
        
        # Read game log
        game_states = self._read_game_log()
        
        # Process observation
        obs = self._process_observation(game_states)
        
        # Update state tracking
        self.previous_state = self.current_state
        if game_states:
            self.current_state = game_states[-1]
        
        # Calculate reward
        reward = self._calculate_reward(self.previous_state, self.current_state)
        
        # Check if done
        done = self._is_done(self.current_state)
        
        # Update step counter
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        # Prepare info
        info = {
            'step': self.current_step,
            'action': action_name,
            'game_states': game_states,
            'position': (self.current_state.get('x', 0), self.current_state.get('y', 0)) if self.current_state else (0, 0),
            'score': self.current_state.get('score', 0) if self.current_state else 0,
            'rings': self.current_state.get('rings', 0) if self.current_state else 0,
            'lives': self.current_state.get('lives', 0) if self.current_state else 0
        }
        
        return obs, reward, done, truncated, info
    
    def render(self, mode: str = 'human'):
        """Render the environment (not implemented for file-based approach)."""
        if mode == 'human':
            print(f"[FileBasedEnv-{self.instance_id}] Current state: {self.current_state}")
        return None
    
    def close(self):
        """Close the environment."""
        self._close_emulator()
    
    def _close_emulator(self):
        """Close the emulator process."""
        if self.emulator_process:
            try:
                self.emulator_process.terminate()
                self.emulator_process.wait(timeout=5)
                print(f"[FileBasedEnv-{self.instance_id}] Emulator closed")
            except subprocess.TimeoutExpired:
                self.emulator_process.kill()
                print(f"[FileBasedEnv-{self.instance_id}] Emulator force-killed")
            except Exception as e:
                print(f"[FileBasedEnv-{self.instance_id}] Error closing emulator: {e}")
            finally:
                self.emulator_process = None
    
    def get_action_meanings(self) -> List[str]:
        """Get the meaning of each action."""
        return list(self.action_mapping.values())
