import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple
import time
import os
from pathlib import Path
import random
import threading

from emulator.sonic_emulator import SonicEmulator
from src.utils.reward_calculator import RewardCalculator
from src.utils.observation_processor import ObservationProcessor
from src.utils.direct_input_manager import get_direct_input_manager
from src.utils.input_isolator import get_input_manager as get_file_input_manager

# Global environment counter for input isolation (thread-safe)
_env_counter = 0
_env_counter_lock = threading.Lock()

class DirectInputSonicEnvironment(gym.Env):
    """
    Sonic the Hedgehog environment with direct input injection as primary method.
    
    This environment uses Windows API to send inputs directly to BizHawk windows,
    with file-based communication as a secondary fallback method.
    """
    
    def __init__(self, config: Dict[str, Any], env_id: Optional[int] = None):
        super().__init__()
        
        global _env_counter
        
        # Assign environment ID for input isolation (thread-safe)
        if env_id is None:
            with _env_counter_lock:
                env_id = _env_counter
                _env_counter += 1
        self.env_id = env_id
        
        self.config = config
        self.game_config = config['game']
        self.env_config = config['environment']
        self.obs_config = config['observation']
        self.action_config = config['actions']
        
        # Initialize emulator with instance ID
        emulator_config = config.get('emulator', {})
        core_path = emulator_config.get('core', 'genesis_plus_gx_libretro.dll')
        
        # Use env_id as instance_id for input isolation
        instance_id = self.env_id % 4  # Distribute across 4 instances
        
        self.emulator = SonicEmulator(
            rom_path=self.game_config['rom_path'],
            bizhawk_dir=config.get('bizhawk_dir', r"C:\Program Files (x86)\BizHawk-2.10-win-x64"),
            lua_script_path=config.get('lua_script_path', 'emulator/bizhawk_bridge_fixed.lua'),
            instance_id=instance_id
        )
        
        # Initialize input managers
        self.direct_input_manager = get_direct_input_manager(num_instances=4)
        self.file_input_manager = get_file_input_manager(num_instances=4)
        
        # Set environment ID for input isolation
        if self.direct_input_manager:
            self.direct_input_manager.assign_environment(self.env_id, instance_id)
            print(f"[DirectInputSonicEnvironment-{self.env_id}] Direct input manager initialized")
        else:
            print(f"[DirectInputSonicEnvironment-{self.env_id}] Warning: Direct input manager not available")
        
        if self.file_input_manager:
            print(f"[DirectInputSonicEnvironment-{self.env_id}] File input manager initialized (fallback)")
        else:
            print(f"[DirectInputSonicEnvironment-{self.env_id}] Warning: File input manager not available")
        
        # Set environment ID for emulator
        self.emulator.set_env_id(self.env_id)
        
        # Initialize processors
        self.obs_processor = ObservationProcessor(self.obs_config)
        
        # Use simplified reward calculator if specified
        reward_calculator_type = config.get('reward_calculator', 'complex')
        if reward_calculator_type == 'simplified':
            from src.utils.simplified_reward_calculator import SimplifiedRewardCalculator
            self.reward_calculator = SimplifiedRewardCalculator(config['rewards'])
        else:
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
        
        # Position tracking for stuck detection
        self.last_positions = []
        self.stuck_threshold = 50  # frames
        self.stuck_distance = 10   # pixels
        
        # High score and completion tracking
        self.high_score = 0
        self.completion_count = 0
        self.first_completion = False
        
        # Input method tracking
        self.use_direct_input = True  # Start with direct input
        self.direct_input_failures = 0
        self.max_direct_input_failures = 5  # Switch to file-based after 5 failures
        
        # Launch emulator
        self.emulator.launch()
        
        # Wait for emulator to be ready and detect game state
        print(f"[DirectInputSonicEnvironment-{self.env_id}] Waiting for emulator and ROM to load...")
        self._wait_for_game_ready()
        
        # Handle game startup sequence
        print(f"[DirectInputSonicEnvironment-{self.env_id}] Starting game from current state...")
        self._handle_game_startup()
        
        print(f"[DirectInputSonicEnvironment-{self.env_id}] Game startup sequence completed")
    
    def _wait_for_game_ready(self):
        """Wait for the emulator and ROM to be ready, with intelligent detection."""
        max_wait_time = 30  # Maximum 30 seconds
        check_interval = 1  # Check every second
        elapsed_time = 0
        
        print(f"[DirectInputSonicEnvironment-{self.env_id}] Waiting for game to be ready...")
        
        while elapsed_time < max_wait_time:
            try:
                # Try to get game state to see if emulator is responsive
                state = self.emulator.get_game_state()
                if state and state.get('game_mode') is not None:
                    print(f"[DirectInputSonicEnvironment-{self.env_id}] Game is ready! Game mode: {state.get('game_mode')}")
                    return
            except Exception as e:
                # Game not ready yet, continue waiting
                pass
            
            time.sleep(check_interval)
            elapsed_time += check_interval
            
            if elapsed_time % 5 == 0:  # Print status every 5 seconds
                print(f"[DirectInputSonicEnvironment-{self.env_id}] Still waiting... ({elapsed_time}s elapsed)")
        
        print(f"[DirectInputSonicEnvironment-{self.env_id}] Warning: Game may not be fully ready after {max_wait_time}s")
    
    def _handle_game_startup(self):
        """Handle the game startup sequence based on current game state."""
        max_attempts = 5
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Get current game state
                state = self.emulator.get_game_state()
                if not state:
                    print(f"[DirectInputSonicEnvironment-{self.env_id}] Cannot read game state, trying START...")
                    self._send_start_command()
                    time.sleep(1)
                    attempt += 1
                    continue
                
                game_mode = state.get('game_mode', 0)
                print(f"[DirectInputSonicEnvironment-{self.env_id}] Current game mode: {game_mode}")
                
                # Check if we're already in gameplay
                if self._is_in_gameplay(game_mode):
                    print(f"[DirectInputSonicEnvironment-{self.env_id}] Already in gameplay mode!")
                    return
                
                # Check if we're in demo mode (need to press START to return to menu)
                if self._is_in_demo_mode(game_mode):
                    print(f"[DirectInputSonicEnvironment-{self.env_id}] In demo mode, pressing START to return to menu...")
                    self._send_start_command()
                    time.sleep(2)  # Wait for transition to menu
                    attempt += 1
                    continue
                
                # Check if we're in menu mode (need to press START to begin game)
                if self._is_in_menu_mode(game_mode):
                    print(f"[DirectInputSonicEnvironment-{self.env_id}] In menu mode, pressing START to begin game...")
                    self._send_start_command()
                    time.sleep(2)  # Wait for game to start
                    attempt += 1
                    continue
                
                # Unknown state, try pressing START
                print(f"[DirectInputSonicEnvironment-{self.env_id}] Unknown game mode {game_mode}, trying START...")
                self._send_start_command()
                time.sleep(2)
                attempt += 1
                
            except Exception as e:
                print(f"[DirectInputSonicEnvironment-{self.env_id}] Error during startup: {e}")
                self._send_start_command()
                time.sleep(2)
                attempt += 1
        
        print(f"[DirectInputSonicEnvironment-{self.env_id}] Warning: Could not properly start game after {max_attempts} attempts")
    
    def _send_start_command(self):
        """Send START command using the appropriate input method."""
        if self.direct_input_manager:
            self.direct_input_manager.send_action(self.env_id, 'START', duration=0.2)
        else:
            # Fallback to file-based START command
            self.emulator.step(['START'])
    
    def _is_in_gameplay(self, game_mode: int) -> bool:
        """Check if the game is in active gameplay mode."""
        # Gameplay modes (these may need adjustment based on actual Sonic 1 memory values)
        gameplay_modes = [0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x20, 0x24, 0x28, 0x2C]
        return game_mode in gameplay_modes
    
    def _is_in_demo_mode(self, game_mode: int) -> bool:
        """Check if the game is in demo mode."""
        # Demo mode (this may need adjustment based on actual Sonic 1 memory values)
        demo_modes = [0x04, 0x06]  # Example demo mode values
        return game_mode in demo_modes
    
    def _is_in_menu_mode(self, game_mode: int) -> bool:
        """Check if the game is in menu/title screen mode."""
        # Menu/title screen modes (this may need adjustment based on actual Sonic 1 memory values)
        menu_modes = [0x00, 0x02]  # Example menu mode values
        return game_mode in menu_modes
    
    def _create_action_space(self) -> spaces.Discrete:
        """Create the action space."""
        basic_actions = self.action_config.get('basic', [])
        combinations = self.action_config.get('combinations', [])
        
        # Total number of actions: basic + combinations + NOOP
        num_actions = len(basic_actions) + len(combinations) + 1
        
        return spaces.Discrete(num_actions)
    
    def _create_observation_space(self) -> spaces.Box:
        """Create the observation space."""
        # Get the processed screen dimensions
        screen_height = self.obs_config.get('height', 84)
        screen_width = self.obs_config.get('width', 84)
        frame_stack = self.obs_config.get('frame_stack', 4)
        
        # Create observation space for stacked frames
        obs_shape = (frame_stack, screen_height, screen_width)
        
        return spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8
        )
    
    def _get_action_mapping(self, action: int) -> list:
        """Convert action index to emulator actions."""
        basic_actions = self.action_config.get('basic', [])
        combinations = self.action_config.get('combinations', [])
        
        if action == 0:
            return []  # NOOP
        elif action <= len(basic_actions):
            return [basic_actions[action - 1]]
        else:
            combo_index = action - len(basic_actions) - 1
            if combo_index < len(combinations):
                return combinations[combo_index]
        
        return []  # Default to NOOP
    
    def _process_observation(self, screen: np.ndarray) -> np.ndarray:
        """Process the raw screen observation."""
        return self.obs_processor.process(screen)
    
    def _update_frame_buffer(self, obs: np.ndarray):
        """Update the frame buffer with new observation."""
        self.frame_buffer.append(obs)
        if len(self.frame_buffer) > self.frame_stack:
            self.frame_buffer.pop(0)
    
    def _get_stacked_observation(self) -> np.ndarray:
        """Get the stacked observation from frame buffer."""
        if len(self.frame_buffer) < self.frame_stack:
            # Pad with zeros if not enough frames
            padding = [np.zeros_like(self.frame_buffer[0]) for _ in range(self.frame_stack - len(self.frame_buffer))]
            stacked = padding + self.frame_buffer
        else:
            stacked = self.frame_buffer
        
        return np.stack(stacked, axis=0)
    
    def _get_game_state(self) -> Dict[str, Any]:
        """Get the current game state from the emulator."""
        return self.emulator.get_game_state()
    
    def _calculate_reward(self, prev_state: Dict[str, Any], 
                         curr_state: Dict[str, Any]) -> float:
        """Calculate reward based on state changes."""
        return self.reward_calculator.calculate_reward(prev_state, curr_state)
    
    def _is_done(self, state: Dict[str, Any]) -> bool:
        """Check if the episode is done."""
        if not state:
            return True
        
        # Check for level completion
        if self._is_level_completed(state):
            return True
        
        # Check for stuck condition
        if self._is_stuck(state):
            return True
        
        # Check for max steps
        if self.current_step >= self.max_steps:
            return True
        
        # Check for game over (lives <= 0)
        lives = state.get('lives', 0)
        if lives <= 0:
            return True
        
        return False
    
    def _is_level_completed(self, state: Dict[str, Any]) -> bool:
        """Check if the level was completed successfully."""
        game_mode = state.get('game_mode', 0)
        lives = state.get('lives', 0)
        
        # Check for level completion game modes
        completion_modes = [0x18, 0x1C, 0x20, 0x24, 0x28, 0x2C]
        
        if game_mode in completion_modes and lives > 0:
            return True
        
        # Also check for zone/act progression
        zone = state.get('zone', 0)
        act = state.get('act', 0)
        
        if zone > 1 or act > 1:  # Progressed beyond the first level
            return True
        
        return False
    
    def _is_stuck(self, state: Dict[str, Any]) -> bool:
        """Check if Sonic is stuck in one place."""
        if not state or 'position' not in state:
            return False
        
        position = state['position']
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            return False
        
        x_pos = position[0]
        
        # Add current position to tracking
        self.last_positions.append(x_pos)
        
        # Keep only last N positions
        if len(self.last_positions) > self.stuck_threshold:
            self.last_positions.pop(0)
        
        # Check if we have enough positions to determine if stuck
        if len(self.last_positions) >= self.stuck_threshold:
            # Calculate movement range
            min_x = min(self.last_positions)
            max_x = max(self.last_positions)
            movement_range = max_x - min_x
            
            # If movement range is very small, we're stuck
            if movement_range < self.stuck_distance:
                return True
        
        return False
    
    def _send_inputs(self, actions: list):
        """Send inputs using the appropriate method."""
        if not actions:
            return
        
        try:
            if self.use_direct_input and self.direct_input_manager:
                # Try direct input first
                for action in actions:
                    self.direct_input_manager.send_action(self.env_id, action, duration=0.016)
                
                # Check if direct input is working by monitoring window status
                instance_id = self.env_id % 4
                if not self.direct_input_manager.input_managers[instance_id].is_window_active():
                    self.direct_input_failures += 1
                    print(f"[DirectInputSonicEnvironment-{self.env_id}] Direct input failure {self.direct_input_failures}")
                else:
                    self.direct_input_failures = 0  # Reset on success
                
                # Switch to file-based if too many failures
                if self.direct_input_failures >= self.max_direct_input_failures:
                    print(f"[DirectInputSonicEnvironment-{self.env_id}] Switching to file-based input due to failures")
                    self.use_direct_input = False
                    self.direct_input_failures = 0
            
            elif self.file_input_manager:
                # Use file-based input as fallback
                self.emulator.step(actions)
            
            else:
                # No input method available
                print(f"[DirectInputSonicEnvironment-{self.env_id}] Warning: No input method available")
                time.sleep(0.016)  # Just wait
                
        except Exception as e:
            print(f"[DirectInputSonicEnvironment-{self.env_id}] Input error: {e}")
            # Switch to file-based on error
            if self.use_direct_input:
                print(f"[DirectInputSonicEnvironment-{self.env_id}] Switching to file-based input due to error")
                self.use_direct_input = False
                self.direct_input_failures = 0
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset environment state
        self.current_step = 0
        self.frame_buffer = []
        self.last_positions = []
        self.previous_state = None
        self.current_state = None
        
        # Try to reset using direct input first
        if self.use_direct_input and self.direct_input_manager:
            try:
                # Send multiple START presses to reset
                for _ in range(3):
                    self.direct_input_manager.send_action(self.env_id, 'START', duration=0.2)
                    time.sleep(0.1)
            except Exception as e:
                print(f"[DirectInputSonicEnvironment-{self.env_id}] Direct reset failed: {e}")
                # Fallback to emulator reset
                self.emulator.reset()
        else:
            # Use emulator reset
            self.emulator.reset()
        
        # Wait for reset to complete
        time.sleep(2)
        
        # Get initial observation
        screen = self.emulator.get_screen()
        obs = self._process_observation(screen)
        self._update_frame_buffer(obs)
        
        # Get initial state
        self.current_state = self._get_game_state()
        
        # Get stacked observation
        stacked_obs = self._get_stacked_observation()
        
        # Create info dict
        info = {
            'step': self.current_step,
            'game_state': self.current_state,
            'input_method': 'direct' if self.use_direct_input else 'file',
            'high_score': self.high_score,
            'completion_count': self.completion_count
        }
        
        return stacked_obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Convert action to emulator actions
        emulator_actions = self._get_action_mapping(action)
        
        # Send inputs using appropriate method
        self._send_inputs(emulator_actions)
        
        # Update step counter
        self.current_step += 1
        
        # Get current observation and state
        screen = self.emulator.get_screen()
        obs = self._process_observation(screen)
        self._update_frame_buffer(obs)
        
        # Update state tracking
        self.previous_state = self.current_state
        self.current_state = self._get_game_state()
        
        # Calculate reward
        reward = self._calculate_reward(self.previous_state, self.current_state)
        
        # Check if episode is done
        done = self._is_done(self.current_state)
        
        # Update high score and completion tracking
        if self.current_state:
            score = self.current_state.get('score', 0)
            if score > self.high_score:
                self.high_score = score
            
            if done and self._is_level_completed(self.current_state):
                self.completion_count += 1
                if not self.first_completion:
                    self.first_completion = True
        
        # Get stacked observation
        stacked_obs = self._get_stacked_observation()
        
        # Create info dict
        info = {
            'step': self.current_step,
            'action': action,
            'actions': emulator_actions,
            'game_state': self.current_state,
            'input_method': 'direct' if self.use_direct_input else 'file',
            'high_score': self.high_score,
            'completion_count': self.completion_count,
            'first_completion': self.first_completion
        }
        
        return stacked_obs, reward, done, False, info
    
    def render(self, mode: str = 'human'):
        """Render the current game state."""
        if mode == 'human':
            screen = self.emulator.get_screen()
            cv2.imshow('Sonic AI (Direct Input)', screen)
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
        
        meanings = ['NOOP']
        for action in basic_actions:
            meanings.append(action)
        for combo in combinations:
            meanings.append("+".join(combo))
        
        return meanings
    
    def save_state(self, path: str):
        """Save the current game state."""
        if self.use_direct_input and self.direct_input_manager:
            self.direct_input_manager.send_action(self.env_id, 'SAVE', duration=0.1)
        else:
            self.emulator.save_state(path)
    
    def load_state(self, path: str):
        """Load a saved game state."""
        if self.use_direct_input and self.direct_input_manager:
            self.direct_input_manager.send_action(self.env_id, 'LOAD', duration=0.1)
        else:
            self.emulator.load_state(path)
    
    def check_objective_completed(self) -> bool:
        """Check if the objective has been completed."""
        if not self.current_state:
            return False
        
        # Check for level completion
        if self._is_level_completed(self.current_state):
            return True
        
        # Check for high score achievement
        score = self.current_state.get('score', 0)
        if score >= 50000:  # Example objective
            return True
        
        return False
    
    def get_objective_progress(self) -> float:
        """Get the progress towards the objective (0.0 to 1.0)."""
        if not self.current_state:
            return 0.0
        
        # Calculate progress based on score
        score = self.current_state.get('score', 0)
        max_score = 100000  # Example max score
        score_progress = min(score / max_score, 1.0)
        
        # Calculate progress based on level completion
        zone = self.current_state.get('zone', 0)
        act = self.current_state.get('act', 0)
        level_progress = min((zone * 3 + act) / 30, 1.0)  # Example: 10 zones, 3 acts each
        
        # Return the higher of the two progress values
        return max(score_progress, level_progress)
