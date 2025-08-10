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
from utils.reward_calculator import RewardCalculator
from utils.observation_processor import ObservationProcessor
from utils.input_isolator import get_input_manager

# Global environment counter for input isolation (thread-safe)
_env_counter = 0
_env_counter_lock = threading.Lock()

class SonicEnvironment(gym.Env):
    """
    Sonic the Hedgehog environment for reinforcement learning.
    
    This environment wraps a Sonic emulator and provides a gym-like interface
    for training RL agents to play Sonic games.
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
            lua_script_path=config.get('lua_script_path', 'emulator/bizhawk_bridge.lua'),  # Fixed: use correct script name
            instance_id=instance_id
        )
        
        # Initialize input manager for this environment
        self.input_manager = get_input_manager(num_instances=4, instance_id=instance_id)
        if self.input_manager:
            print(f"[SonicEnvironment-{self.env_id}] Input manager initialized for instance {instance_id}")
        else:
            print(f"[SonicEnvironment-{self.env_id}] Warning: Input manager not available")
        
        # Set environment ID for input isolation
        self.emulator.set_env_id(self.env_id)
        
        # Initialize processors
        self.obs_processor = ObservationProcessor(self.obs_config)
        
        # Use simplified reward calculator if specified
        reward_calculator_type = config.get('reward_calculator', 'complex')
        if reward_calculator_type == 'simplified':
            from utils.simplified_reward_calculator import SimplifiedRewardCalculator
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
        
        # High score tracking for Green Hill Zone completion
        self.high_score = 0
        self.first_completion = False
        self.completion_count = 0
        
        # Position tracking for stuck detection
        self.last_positions = []
        
        print(f"[SonicEnvironment-{self.env_id}] Initialized with instance_id={instance_id}")
        
    def _create_action_space(self) -> spaces.Discrete:
        """Create the action space based on available actions."""
        # Basic actions: NOOP, LEFT, RIGHT, UP, DOWN, A, B, START, SELECT
        # Plus combinations: LEFT+A, RIGHT+A, DOWN+B, LEFT+DOWN+B, RIGHT+DOWN+B
        basic_actions = self.action_config.get('basic', [])
        combinations = self.action_config.get('combinations', [])
        num_actions = len(basic_actions) + len(combinations)
        
        # Validate action space
        if num_actions == 0:
            print("Warning: No actions defined, using default action space")
            basic_actions = ["NOOP", "LEFT", "RIGHT", "A", "B", "START"]
            combinations = [["LEFT", "A"], ["RIGHT", "A"]]
            num_actions = len(basic_actions) + len(combinations)
        
        print(f"[SonicEnvironment-{self.env_id}] Action space: {num_actions} actions")
        print(f"  Basic: {basic_actions}")
        print(f"  Combinations: {combinations}")
        
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
            action_name = basic_actions[action]
            return [action_name]
        else:
            combo_idx = action - len(basic_actions)
            if combo_idx < len(combinations):
                return combinations[combo_idx]
            else:
                print(f"Warning: Invalid action {action}, using NOOP")
                return ["NOOP"]
    
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
        
        # Check objective type
        objective = self.config['game'].get('objective', 'complex')
        
        if objective == 'Move right and survive':
            # Simplified objective: just check time limit and stuck detection
            if self.current_step >= self.max_steps:
                return True
            
            # Stuck detection (optional)
            if self._is_stuck(state):
                return True
            
            return False
        else:
            # Complex objective: check for Green Hill Zone Act 3 completion
            if self._is_green_hill_zone_act3_completed(state):
                self._handle_green_hill_completion(state)
                return True
            
            # Time limit
            if self.current_step >= self.max_steps:
                return True
            
            # Stuck detection (optional)
            if self._is_stuck(state):
                return True
            
            return False
    
    def _is_green_hill_zone_act3_completed(self, state: Dict[str, Any]) -> bool:
        """Check if Green Hill Zone Act 3 has been completed."""
        zone = state.get('zone', 0)
        act = state.get('act', 0)
        game_mode = state.get('game_mode', 0)
        lives = state.get('lives', 0)
        
        # Check if we've completed Green Hill Zone Act 3 (Zone 1, Act 3)
        # or progressed beyond it (Zone 2+ or Act 4+)
        if lives > 0:  # Must still have lives
            if zone > 1:  # Completed Green Hill Zone (Zone 1) and moved to Zone 2
                return True
            elif zone == 1 and act > 3:  # Completed Act 3 and moved to Act 4
                return True
            elif zone == 1 and act == 3 and game_mode in [0x18, 0x1C, 0x20, 0x24, 0x28, 0x2C]:  # Act 3 completion screen
                return True
        
        return False
    
    def _handle_green_hill_completion(self, state: Dict[str, Any]):
        """Handle Green Hill Zone completion and score tracking."""
        score = state.get('score', 0)
        rings = state.get('rings', 0)
        lives = state.get('lives', 0)
        
        self.completion_count += 1
        
        # Check if this is the first completion
        if not self.first_completion:
            self.first_completion = True
            self.high_score = score
            print(f"\n🎉 [SonicEnvironment-{self.env_id}] FIRST GREEN HILL ZONE COMPLETION! 🎉")
            print(f"   Score: {score:,}")
            print(f"   Rings: {rings}")
            print(f"   Lives: {lives}")
            print(f"   Steps: {self.current_step}")
            print(f"   Zone: {state.get('zone', 0)}, Act: {state.get('act', 0)}")
            print("=" * 60)
        
        # Check if this beats the previous high score
        elif score > self.high_score:
            old_high = self.high_score
            self.high_score = score
            print(f"\n🏆 [SonicEnvironment-{self.env_id}] NEW HIGH SCORE! 🏆")
            print(f"   Previous High: {old_high:,}")
            print(f"   New High: {score:,}")
            print(f"   Improvement: +{score - old_high:,}")
            print(f"   Rings: {rings}")
            print(f"   Lives: {lives}")
            print(f"   Steps: {self.current_step}")
            print(f"   Completion #: {self.completion_count}")
            print("=" * 60)
        
        # Regular completion log
        else:
            print(f"[SonicEnvironment-{self.env_id}] Green Hill Zone completed - Score: {score:,}, High Score: {self.high_score:,}, Completion #{self.completion_count}")
    
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
        """Reset the environment with intelligent game state detection."""
        super().reset(seed=seed)
        
        # Reset emulator
        self.emulator.reset()
        
        # Wait for emulator to fully load
        time.sleep(2)
        
        # Intelligent game startup using RAM detection
        self._smart_game_startup()
        
        # Reset frame buffer
        self.frame_buffer = []
        
        # Get initial observation
        screen = self.emulator.get_screen()
        obs = self._process_observation(screen)
        
        # Initialize frame buffer
        for _ in range(self.frame_stack):
            self._update_frame_buffer(obs)
        
        # Reset state tracking
        self.current_step = 0
        self.previous_state = None
        self.current_state = self._get_game_state()
        
        # Reset position tracking for stuck detection
        self.last_positions = []
        
        # Get stacked observation
        stacked_obs = self._get_stacked_observation()
        
        return stacked_obs, {}
    
    def _smart_game_startup(self):
        """Intelligent game startup using RAM detection."""
        print(f"[SonicEnvironment-{self.env_id}] Starting intelligent game startup...")
        
        # First, try to load a save state if available
        if self._try_load_save_state():
            print(f"[SonicEnvironment-{self.env_id}] Successfully loaded save state!")
            return
        
        max_attempts = 50  # Maximum attempts to avoid infinite loops
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            
            # Get current game state
            game_state = self.emulator.get_game_state()
            game_mode = game_state.get('game_mode', 0)
            zone = game_state.get('zone', 0)
            act = game_state.get('act', 0)
            lives = game_state.get('lives', 0)
            
            print(f"[SonicEnvironment-{self.env_id}] Attempt {attempt}: Game mode={game_mode}, Zone={zone}, Act={act}, Lives={lives}")
            
            # Check if we're already in a playable state
            if self._is_in_playable_state(game_mode, zone, act, lives):
                print(f"[SonicEnvironment-{self.env_id}] Already in playable state!")
                return
            
            # Handle different game modes
            if self._handle_game_mode(game_mode):
                time.sleep(0.5)  # Wait for state change
                continue
            
            # If we can't determine the state, use fallback
            if attempt > 30:
                print(f"[SonicEnvironment-{self.env_id}] Using fallback startup method...")
                self._fallback_startup()
                return
        
        print(f"[SonicEnvironment-{self.env_id}] Warning: Could not reach playable state after {max_attempts} attempts")
    
    def _try_load_save_state(self) -> bool:
        """Try to load a save state for faster startup."""
        try:
            # Check if we have a save state file
            save_state_path = f"save_states/sonic_env_{self.env_id}.state"
            if os.path.exists(save_state_path):
                print(f"[SonicEnvironment-{self.env_id}] Found save state, attempting to load...")
                
                # Load the save state
                self.emulator.load_state(save_state_path)
                time.sleep(1)  # Wait for load to complete
                
                # Verify we're in a good state
                game_state = self.emulator.get_game_state()
                game_mode = game_state.get('game_mode', 0)
                lives = game_state.get('lives', 0)
                
                if self._is_in_playable_state(game_mode, 1, 1, lives):
                    print(f"[SonicEnvironment-{self.env_id}] Save state loaded successfully!")
                    return True
                else:
                    print(f"[SonicEnvironment-{self.env_id}] Save state loaded but not in playable state")
                    return False
            else:
                print(f"[SonicEnvironment-{self.env_id}] No save state found, will create one after successful startup")
                return False
                
        except Exception as e:
            print(f"[SonicEnvironment-{self.env_id}] Error loading save state: {e}")
            return False
    
    def _create_save_state(self):
        """Create a save state for future fast startup."""
        try:
            # Ensure save states directory exists
            os.makedirs("save_states", exist_ok=True)
            
            # Create save state
            save_state_path = f"save_states/sonic_env_{self.env_id}.state"
            self.emulator.save_state(save_state_path)
            print(f"[SonicEnvironment-{self.env_id}] Created save state: {save_state_path}")
            
        except Exception as e:
            print(f"[SonicEnvironment-{self.env_id}] Error creating save state: {e}")
    
    def _is_in_playable_state(self, game_mode: int, zone: int, act: int, lives: int) -> bool:
        """Check if we're in a playable game state."""
        # Game modes that indicate we're in a level
        playable_modes = [0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
        
        # Check if we're in a playable mode and have lives
        if game_mode in playable_modes and lives > 0:
            return True
        
        # Also check if we're in a specific zone/act combination that indicates gameplay
        if zone > 0 and act > 0 and lives > 0:
            return True
        
        return False
    
    def _handle_game_mode(self, game_mode: int) -> bool:
        """Handle different game modes intelligently."""
        print(f"[SonicEnvironment-{self.env_id}] Handling game mode: 0x{game_mode:02X}")
        
        # Title screen / Main menu (0x00-0x0B)
        if game_mode <= 0x0B:
            print(f"[SonicEnvironment-{self.env_id}] In title screen/menu, pressing START...")
            self.emulator.step(['START'])
            return True
        
        # Game over screen (0x18-0x1F)
        elif game_mode >= 0x18 and game_mode <= 0x1F:
            print(f"[SonicEnvironment-{self.env_id}] In game over screen, pressing START...")
            self.emulator.step(['START'])
            return True
        
        # Level complete / Act clear (0x20-0x2F)
        elif game_mode >= 0x20 and game_mode <= 0x2F:
            print(f"[SonicEnvironment-{self.env_id}] Level complete, pressing START...")
            self.emulator.step(['START'])
            return True
        
        # Special stage or other modes (0x30+)
        elif game_mode >= 0x30:
            print(f"[SonicEnvironment-{self.env_id}] In special stage/other mode, pressing START...")
            self.emulator.step(['START'])
            return True
        
        # Unknown mode
        else:
            print(f"[SonicEnvironment-{self.env_id}] Unknown game mode 0x{game_mode:02X}, pressing START...")
            self.emulator.step(['START'])
            return True
    
    def _fallback_startup(self):
        """Fallback startup method if intelligent detection fails."""
        print(f"[SonicEnvironment-{self.env_id}] Using fallback startup method...")
        
        # Press START multiple times to navigate through menus
        for i in range(5):
            print(f"[SonicEnvironment-{self.env_id}] Fallback START press {i+1}/5")
            self.emulator.step(['START'])
            time.sleep(0.3)
        
        # Wait a bit more for the game to settle
        time.sleep(1)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Convert action to emulator actions
        emulator_actions = self._get_action_mapping(action)
        
        # Execute actions
        self.emulator.step(emulator_actions)
        
        # Update step counter
        self.current_step += 1
        
        # Get current observation and state
        screen = self.emulator.get_screen()
        obs = self._process_observation(screen)
        self._update_frame_buffer(obs)
        
        # Update state tracking
        self.previous_state = self.current_state
        self.current_state = self._get_game_state()
        
        # Track position for stuck detection
        if self.current_state:
            position = self.current_state.get('position', (0, 0))
            if isinstance(position, (list, tuple)) and len(position) >= 2:
                self.last_positions.append(position[0])  # Track X position
                # Keep only last 100 positions to avoid memory bloat
                if len(self.last_positions) > 100:
                    self.last_positions.pop(0)
        
        # Calculate reward
        reward = self._calculate_reward(self.previous_state, self.current_state)
        
        # Check if episode is done
        done = self._is_done(self.current_state)
        
        # Create save state if we completed Green Hill Zone Act 3 successfully
        if done and self._is_green_hill_zone_act3_completed(self.current_state):
            self._create_save_state()
        
        # Get stacked observation
        stacked_obs = self._get_stacked_observation()
        
        # Create info dict
        info = {
            'step': self.current_step,
            'action': action,
            'actions': emulator_actions,
            'game_state': self.current_state,
            'high_score': self.high_score,
            'completion_count': self.completion_count,
            'first_completion': self.first_completion
        }
        
        return stacked_obs, reward, done, False, info
    
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
        
        meanings = []
        for action in basic_actions:
            meanings.append(action)
        for combo in combinations:
            meanings.append("+".join(combo))
        
        return meanings
    
    def save_state(self, path: str):
        """Save the current game state."""
        self.emulator.save_state(path)
    
    def load_state(self, path: str):
        """Load a saved game state."""
        self.emulator.load_state(path)
    
    def check_objective_completed(self) -> bool:
        """Check if the objective (end of Green Hill Zone Act 3) has been completed."""
        try:
            # Get current game state
            game_state = self.emulator.get_game_state()
            
            # Check for level completion indicators
            # These are typical signs that Sonic has reached the end of the act
            if game_state.get('level_completed', False):
                return True
            
            # Check if we're in the end-of-act sequence
            # This usually involves Sonic running to the right and the screen scrolling
            # or showing the "ACT CLEAR" screen
            if game_state.get('game_state') == 'act_clear':
                return True
            
            # Check if Sonic has reached the end position
            # The end of Green Hill Zone Act 3 is typically around x-position 6000+
            position = game_state.get('position', (0, 0))
            if position[0] > 6000:  # Adjust this threshold based on actual level length
                return True
            
            # Check for end-of-act music or sound effects
            # This would require audio analysis or memory reading
            
            return False
            
        except Exception as e:
            print(f"Error checking objective completion: {e}")
            return False

    def get_objective_progress(self) -> float:
        """Get progress towards the objective (0.0 to 1.0)."""
        try:
            game_state = self.emulator.get_game_state()
            position = game_state.get('position', (0, 0))
            
            # Calculate progress based on x-position
            # Green Hill Zone Act 3 is approximately 7000 pixels long
            max_distance = 7000
            current_distance = max(0, position[0])
            
            progress = min(1.0, current_distance / max_distance)
            return progress
            
        except Exception as e:
            print(f"Error calculating objective progress: {e}")
            return 0.0 