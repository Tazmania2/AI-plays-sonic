import numpy as np
import time
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Stable-Retro (maintained fork of gym-retro) exposes the same "retro" API
try:
    import retro  # provided by stable-retro
except Exception as e:
    retro = None
    print(f"[SonicEmulator] Warning: retro (stable-retro) not available: {e}")


class SonicEmulator:
    """
    Stable-Retro based Sonic emulator wrapper (Colab friendly).
    - No external windows, no BizHawk, no Lua bridge.
    - Buttons are sent as a multi-binary vector in a single step.
    """

    # Default Genesis button order used by Retro
    DEFAULT_BUTTONS = ['B', 'C', 'A', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT']
    ACTION_ALIASES = {
        'NOOP': [],
        'LEFT': ['LEFT'],
        'RIGHT': ['RIGHT'],
        'UP': ['UP'],
        'DOWN': ['DOWN'],
        'A': ['A'],
        'B': ['B'],
        'C': ['C'],
        'START': ['START'],
        'SELECT': ['START'],  # no SELECT on Genesis; map to START if used
    }

    def __init__(
        self,
        rom_path: str = '',
        bizhawk_dir: str = '',
        lua_script_path: str = '',
        port: int = 0,
        instance_id: int = 0,
        retro_game: str = 'SonicTheHedgehog-Genesis',
        retro_state: str = 'GreenHillZone.Act1',
        frame_skip: int = 1,
        render: bool = False,
    ):
        self.instance_id = instance_id
        self.retro_game = retro_game
        self.retro_state = retro_state
        self.frame_skip = max(1, int(frame_skip))
        self.render_enabled = render

        self.env: Optional[retro.RetroEnv] = None if retro is None else retro.make(
            game=self.retro_game,
            state=self.retro_state,
            use_restricted_actions=retro.Actions.DISCRETE  # still returns multi-binary buttons
        )

        if self.env is None:
            raise RuntimeError("stable-retro not available or game/state not found. Make sure to pip install stable-retro and import ROMs with `python -m retro.import roms/`.")

        self.last_obs: Optional[np.ndarray] = None
        self.last_info: Dict[str, Any] = {}
        self.prev_info: Dict[str, Any] = {}
        self.last_speed_x: float = 0.0
        self.obs_shape = self.env.observation_space.shape
        self.buttons = list(self.DEFAULT_BUTTONS)
        self.button_to_idx = {b: i for i, b in enumerate(self.buttons)}
        self.reset()

    def set_env_id(self, env_id: int):
        # API compatibility; not required for Retro
        pass

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        except Exception:
            pass
        self.env = None

    def _actions_to_vector(self, actions: List[str]) -> np.ndarray:
        vec = np.zeros(len(self.buttons), dtype=np.uint8)
        for a in actions:
            keys = self.ACTION_ALIASES.get(a.upper(), [a.upper()])
            for k in keys:
                idx = self.button_to_idx.get(k)
                if idx is not None:
                    vec[idx] = 1
        return vec

    def reset(self) -> np.ndarray:
        obs, info = self.env.reset()
        self.prev_info = {}
        self.last_info = info or {}
        self.last_speed_x = 0.0
        self.last_obs = obs
        return obs

    def get_screen(self) -> np.ndarray:
        return self.last_obs.copy() if self.last_obs is not None else np.zeros(self.obs_shape, dtype=np.uint8)

    def step(self, actions: List[str]):
        if not actions:
            actions = ['NOOP']

        action_vec = self._actions_to_vector(actions)
        total_reward = 0.0
        info_agg = {}
        done = False
        obs = None

        # Simple frame-skip: repeat same action
        for i in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action_vec)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            # capture the latest info
            info_agg = info
            if done:
                break

        self.last_obs = obs
        try:
            # compute simple speed from x delta if present
            prev_x = float(self.last_info.get('x', 0) or 0)
            curr_x = float((info_agg or {}).get('x', 0) or 0)
            self.last_speed_x = curr_x - prev_x
        except Exception:
            self.last_speed_x = 0.0
        self.prev_info = self.last_info
        self.last_info = info_agg or {}
        return obs, total_reward, done, info_agg

    # Minimal game state extraction (best-effort; many values may be 0 without custom scenarios)
    def get_game_state(self) -> Dict[str, Any]:
        state = {
            'position': (0, 0),
            'position_x': 0,
            'position_y': 0,
            'velocity': (0, 0),
            'rings': 0,
            'rings_count': 0,
            'score': 0,
            'lives': 3,
            'zone': 0,
            'act': 0,
            'zone_act': (0, 0),
            'game_mode': 0,
            'game_mode_name': 'Unknown',
            'timer': (0, 0, 0),
            'invincibility': False,
            'shield': False,
            'speed_shoes': False,
            'air_remaining': 0,
            'lamppost_counter': 0,
            'emeralds': 0,
            'angle': 0,
            'status': {'on_ground': False, 'in_air': False, 'underwater': False, 'on_object': False},
            'level_timer_frames': 0,
            'speed': float(self.last_speed_x),
            'on_ground': False,
            'in_air': False,
            'underwater': False,
            'on_object': False
        }
        # Prefer scenario-provided info keys (more reliable than raw RAM)
        info = self.last_info or {}
        x = int(info.get('x', 0) or 0)
        y = int(info.get('y', 0) or 0)
        rings = int(info.get('rings', info.get('ring', 0) or 0))
        lives = int(info.get('lives', 0) or 0)
        score = int(info.get('score,', info.get('score', 0) or 0))
        zone = int(info.get('zone', 0) or 0)
        act = int(info.get('act', 0) or 0)

        state.update({
            'position': (x, y),
            'position_x': x,
            'position_y': y,
            'rings': rings,
            'rings_count': rings,
            'lives': lives if lives > 0 else state['lives'],
            'score': score,
            'zone': zone,
            'act': act,
            'zone_act': (zone, act),
        })
        return state

    def _get_sonic_position(self):
        try:
            x = self._read_memory_safe(0xFFD030, 2, 'position_x')
            y = self._read_memory_safe(0xFFD038, 2, 'position_y')
            return (x, y)
        except Exception as e:
            print(f"[MemoryRead] Error reading Sonic position: {e}")
            return (0, 0)

    def _get_sonic_velocity(self):
        try:
            vx = self._read_memory_safe(0xFFD040, 2, 'speed')
            vy = self._read_memory_safe(0xFFD042, 2, 'speed')
            return (vx, vy)
        except Exception as e:
            print(f"[MemoryRead] Error reading Sonic velocity: {e}")
            return (0, 0)

    def _get_sonic_rings(self):
        try:
            return self._read_memory_safe(0xFFE002, 2, 'rings')
        except Exception as e:
            print(f"[MemoryRead] Error reading rings: {e}")
            return 0

    def _get_sonic_score(self):
        try:
            return self._read_memory_safe(0xFFE000, 4, 'score')
        except Exception as e:
            print(f"[MemoryRead] Error reading score: {e}")
            return 0

    def _get_sonic_lives(self):
        try:
            return self._read_memory_safe(0xFFE004, 2, 'lives')
        except Exception as e:
            print(f"[MemoryRead] Error reading lives: {e}")
            return 3

    def _get_zone_act(self):
        try:
            zone = self._read_memory_safe(0xFFE012, 2, 'zone')
            act = self._read_memory_safe(0xFFE014, 2, 'act')
            return (zone, act)
        except Exception as e:
            print(f"[MemoryRead] Error reading zone/act: {e}")
            return (0, 0)

    def _get_game_mode(self):
        try:
            return self._read_memory_safe(0xFFE00C, 2, 'game_state')
        except Exception as e:
            print(f"[MemoryRead] Error reading game mode: {e}")
            return 0

    def _get_timer(self):
        try:
            minutes = self._read_memory_safe(0xFFE010, 1, 'timer')
            seconds = self._read_memory_safe(0xFFE011, 1, 'timer')
            frames = self._read_memory_safe(0xFFE012, 1, 'timer')
            return (minutes, seconds, frames)
        except Exception as e:
            print(f"[MemoryRead] Error reading timer: {e}")
            return (0, 0, 0)

    def _get_invincibility(self):
        try:
            timer = self._read_memory_safe(0xFFE00E, 2, 'invincibility')
            return timer > 0
        except Exception as e:
            print(f"[MemoryRead] Error reading invincibility: {e}")
            return False

    def _get_shield(self):
        try:
            timer = self._read_memory_safe(0xFFE016, 2, 'shields')
            return timer > 0
        except Exception as e:
            print(f"[MemoryRead] Error reading shield: {e}")
            return False

    def _get_speed_shoes(self):
        try:
            timer = self._read_memory_safe(0xFFE018, 2, 'speed_shoes')
            return timer > 0
        except Exception as e:
            print(f"[MemoryRead] Error reading speed shoes: {e}")
            return False

    def _get_air_remaining(self):
        try:
            return self._read_memory_safe(0xFFE01A, 2, 'air_remaining')
        except Exception as e:
            print(f"[MemoryRead] Error reading air remaining: {e}")
            return 0

    def _get_lamppost_counter(self):
        try:
            return self._read_memory_safe(0xFFE01C, 1, 'lamppost_counter')
        except Exception as e:
            print(f"[MemoryRead] Error reading lamppost counter: {e}")
            return 0

    def _get_emeralds(self):
        try:
            return self._read_memory_safe(0xFFE01E, 1, 'emeralds')
        except Exception as e:
            print(f"[MemoryRead] Error reading emeralds: {e}")
            return 0

    def _get_sonic_angle(self):
        try:
            return self._read_memory_safe(0xFFD042, 1, 'angle')
        except Exception as e:
            print(f"[MemoryRead] Error reading Sonic angle: {e}")
            return 0

    def _get_sonic_status(self):
        try:
            status = self._read_memory_safe(0xFFD044, 1, 'status')
            return self._parse_sonic_status(status)
        except Exception as e:
            print(f"[MemoryRead] Error reading Sonic status: {e}")
            return {'on_ground': False, 'in_air': True, 'underwater': False, 'on_object': False}

    def _get_level_timer_frames(self):
        try:
            return self._read_memory_safe(0xFFE020, 2, 'level_timer_frames')
        except Exception as e:
            print(f"[MemoryRead] Error reading level timer frames: {e}")
            return 0

    # Convenience methods for individual values
    def get_sonic_x(self):
        """Get Sonic's X position."""
        try:
            data = self.read_memory(0xFFD030, 2)
            return int.from_bytes(data, byteorder='big', signed=True)
        except Exception as e:
            print(f"[MemoryRead] Error reading Sonic X position: {e}")
            return 0

    def get_sonic_y(self):
        """Get Sonic's Y position."""
        try:
            data = self.read_memory(0xFFD038, 2)
            return int.from_bytes(data, byteorder='big', signed=True)
        except Exception as e:
            print(f"[MemoryRead] Error reading Sonic Y position: {e}")
            return 0

    def get_rings(self):
        """Get current ring count."""
        try:
            data = self.read_memory(0xFFFE20, 2)
            return int.from_bytes(data, byteorder='big', signed=False)
        except Exception as e:
            print(f"[MemoryRead] Error reading rings: {e}")
            return 0

    def get_lives(self):
        """Get current lives count."""
        try:
            data = self.read_memory(0xFFFE12, 1)
            return int.from_bytes(data, byteorder='big', signed=False)
        except Exception as e:
            print(f"[MemoryRead] Error reading lives: {e}")
            return 0

    def get_score(self):
        """Get current score."""
        try:
            data = self.read_memory(0xFFFE26, 4)
            # Convert BCD to decimal
            score_str = ""
            for byte in data:
                high_nibble = (byte >> 4) & 0xF
                low_nibble = byte & 0xF
                score_str += str(high_nibble) + str(low_nibble)
            return int(score_str) if score_str.isdigit() else 0
        except Exception as e:
            print(f"[MemoryRead] Error reading score: {e}")
            return 0

    def _update_capture_region(self):
        """Update screen capture region to target BizHawk window."""
        if not MSS_AVAILABLE:
            return
            
        try:
            import win32gui
            
            def enum_windows_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    window_text = win32gui.GetWindowText(hwnd)
                    if 'bizhawk' in window_text.lower() or 'emuhawk' in window_text.lower():
                        windows.append(hwnd)
                return True
            
            windows = []
            win32gui.EnumWindows(enum_windows_callback, windows)
            
            if windows:
                # Use the first BizHawk window found
                hwnd = windows[0]
                rect = win32gui.GetWindowRect(hwnd)
                x, y, right, bottom = rect
                
                # Adjust for window borders and title bar
                border_width = 8
                title_height = 30
                
                self.capture_region = {
                    "top": y + title_height,
                    "left": x + border_width,
                    "width": right - x - 2 * border_width,
                    "height": bottom - y - title_height - border_width
                }
                
                print(f"[SonicEmulator-{self.instance_id}] Updated capture region: {self.capture_region}")
            else:
                print(f"[SonicEmulator-{self.instance_id}] No BizHawk window found for capture region")
                
        except Exception as e:
            print(f"[SonicEmulator-{self.instance_id}] Error updating capture region: {e}") 

    def get_game_state(self) -> dict:
        """Get comprehensive game state information."""
        try:
            # Get basic game state
            position = self._get_sonic_position()
            velocity = self._get_sonic_velocity()
            rings = self._get_sonic_rings()
            score = self._get_sonic_score()
            lives = self._get_sonic_lives()
            zone_act = self._get_zone_act()
            game_mode = self._get_game_mode()
            timer = self._get_timer()
            invincibility = self._get_invincibility()
            shield = self._get_shield()
            speed_shoes = self._get_speed_shoes()
            air_remaining = self._get_air_remaining()
            lamppost_counter = self._get_lamppost_counter()
            emeralds = self._get_emeralds()
            angle = self._get_sonic_angle()
            status = self._get_sonic_status()
            level_timer_frames = self._get_level_timer_frames()
            
            # Create comprehensive state dictionary
            state = {
                'position': position,
                'position_x': position[0] if isinstance(position, (list, tuple)) else position,
                'position_y': position[1] if isinstance(position, (list, tuple)) else 0,
                'velocity': velocity,
                'rings': rings,
                'rings_count': rings,  # Alternative key for compatibility
                'score': score,
                'lives': lives,
                'zone': zone_act[0] if isinstance(zone_act, (list, tuple)) else zone_act,
                'act': zone_act[1] if isinstance(zone_act, (list, tuple)) else 0,
                'zone_act': zone_act,
                'game_mode': game_mode,
                'game_mode_name': self._get_game_mode_name(game_mode),
                'timer': timer,
                'invincibility': invincibility,
                'shield': shield,
                'speed_shoes': speed_shoes,
                'air_remaining': air_remaining,
                'lamppost_counter': lamppost_counter,
                'emeralds': emeralds,
                'angle': angle,
                'status': status,
                'level_timer_frames': level_timer_frames,
                'speed': abs(velocity[0]) if isinstance(velocity, (list, tuple)) else abs(velocity),
                'on_ground': status.get('on_ground', False),
                'in_air': status.get('in_air', False),
                'underwater': status.get('underwater', False),
                'on_object': status.get('on_object', False)
            }
            
            return state
            
        except Exception as e:
            print(f"[SonicEmulator-{self.instance_id}] Error getting game state: {e}")
            # Return default state on error
            return {
                'position': (0, 0),
                'position_x': 0,
                'position_y': 0,
                'velocity': (0, 0),
                'rings': 0,
                'rings_count': 0,
                'score': 0,
                'lives': 3,
                'zone': 0,
                'act': 0,
                'zone_act': (0, 0),
                'game_mode': 0,
                'game_mode_name': 'Error',
                'timer': (0, 0, 0),
                'invincibility': False,
                'shield': False,
                'speed_shoes': False,
                'air_remaining': 0,
                'lamppost_counter': 0,
                'emeralds': 0,
                'angle': 0,
                'status': {'on_ground': False, 'in_air': True, 'underwater': False, 'on_object': False},
                'level_timer_frames': 0,
                'speed': 0,
                'on_ground': False,
                'in_air': True,
                'underwater': False,
                'on_object': False
            }

    def _get_game_mode_name(self, mode: int) -> str:
        """Get human-readable name for game mode."""
        mode_names = {
            0x00: "Title Screen",
            0x01: "Sega Logo",
            0x02: "Main Menu",
            0x03: "Level Select",
            0x04: "Options Menu",
            0x05: "Sound Test",
            0x06: "Credits",
            0x07: "Demo Mode",
            0x08: "Level Loading",
            0x09: "Level Intro",
            0x0A: "Level Start",
            0x0B: "Level Ready",
            0x0C: "Level Playing",
            0x0D: "Level Playing (Paused)",
            0x0E: "Level Playing (Special)",
            0x0F: "Level Playing (Boss)",
            0x10: "Level Playing (Mini-Boss)",
            0x11: "Level Playing (Cutscene)",
            0x12: "Level Playing (Transition)",
            0x13: "Level Playing (Ending)",
            0x14: "Level Playing (Special Stage)",
            0x15: "Level Playing (Bonus Stage)",
            0x16: "Level Playing (Continue)",
            0x17: "Level Playing (Game Over)",
            0x18: "Act Clear",
            0x19: "Zone Clear",
            0x1A: "Game Complete",
            0x1B: "Special Stage Clear",
            0x1C: "Boss Defeated",
            0x1D: "Mini-Boss Defeated",
            0x1E: "Continue Screen",
            0x1F: "Game Over Screen",
            0x20: "Level Complete",
            0x21: "Zone Complete",
            0x22: "Game Complete",
            0x23: "Special Stage Complete",
            0x24: "Boss Complete",
            0x25: "Mini-Boss Complete",
            0x26: "Continue Complete",
            0x27: "Game Over Complete",
            0x28: "Level Transition",
            0x29: "Zone Transition",
            0x2A: "Game Transition",
            0x2B: "Special Stage Transition",
            0x2C: "Boss Transition",
            0x2D: "Mini-Boss Transition",
            0x2E: "Continue Transition",
            0x2F: "Game Over Transition"
        }
        
        return mode_names.get(mode, f"Unknown Mode 0x{mode:02X}")

    def _parse_sonic_status(self, status: int) -> dict:
        """Parse Sonic's status byte into readable flags."""
        return {
            'on_ground': bool(status & 0x01),
            'in_air': bool(status & 0x02), 
            'rolling': bool(status & 0x04),
            'jumping': bool(status & 0x08),
            'spinning': bool(status & 0x10),
            'pushing': bool(status & 0x20),
            'underwater': bool(status & 0x40),
            'on_object': bool(status & 0x80)
        } 

    def _validate_memory_value(self, key: str, value: int) -> int:
        """Validate memory value against expected range."""
        if key in self.memory_ranges:
            min_val, max_val = self.memory_ranges[key]
            if value < min_val or value > max_val:
                print(f"[SonicEmulator-{self.instance_id}] Warning: {key} value {value} outside expected range [{min_val}, {max_val}]")
                # Return a reasonable default value
                return min_val if value < min_val else max_val
        return value
    
    def _read_memory_safe(self, address: int, size: int, key: str = None) -> int:
        """Read memory with validation."""
        try:
            data = self.memory_reader.read_memory(address, size)
            if size == 1:
                value = int.from_bytes(data, byteorder='big', signed=False)
            elif size == 2:
                value = int.from_bytes(data, byteorder='big', signed=True)
            elif size == 4:
                value = int.from_bytes(data, byteorder='big', signed=False)
            else:
                value = int.from_bytes(data, byteorder='big', signed=False)
            
            # Validate if key is provided
            if key:
                value = self._validate_memory_value(key, value)
            
            return value
        except Exception as e:
            print(f"[SonicEmulator-{self.instance_id}] Memory read error for {key} at {address:x}: {e}")
            return 0 

    def _init_input_manager(self):
        """Initialize the direct input manager for this instance."""
        if self.input_manager is None:
            try:
                self.input_manager = DirectInputManager(self.instance_id)
                self.input_manager.start()
                print(f"[SonicEmulator-{self.instance_id}] Direct input manager initialized for instance {self.instance_id}")
            except Exception as e:
                print(f"[SonicEmulator-{self.instance_id}] Failed to initialize direct input manager: {e}")
                self.input_manager = None 
