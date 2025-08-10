import numpy as np
import cv2
import time
import subprocess
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import threading
import queue
from src.utils.bizhawk_memory_file import BizHawkMemoryReaderFile
from src.utils.input_isolator import get_input_manager

# For screen capture (cross-platform)
try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False


class SonicEmulator:
    """
    Wrapper for Sonic game emulator.
    
    This class provides a unified interface to control Sonic games
    through various emulators (BizHawk, RetroArch, etc.).
    """
    
    def __init__(self, rom_path: str, bizhawk_dir: str, lua_script_path: str, port: int = 55555, instance_id: int = 0):
        self.rom_path = Path(rom_path)
        self.bizhawk_dir = bizhawk_dir
        self.lua_script_path = lua_script_path
        # self.port = port  # DEPRECATED: No longer used with file-based communication
        self.instance_id = instance_id
        self.process = None
        self.memory_reader = BizHawkMemoryReaderFile(instance_id=instance_id)
        
        # Screen capture
        self.screen_capture = None
        self.capture_region = None
        self.screen_width = 224
        self.screen_height = 256
        
        # Initialize screen capture
        if MSS_AVAILABLE:
            try:
                self.screen_capture = mss.mss()
                # Default capture region (will be updated when window is found)
                self.capture_region = {"top": 0, "left": 0, "width": self.screen_width, "height": self.screen_height}
                print(f"[SonicEmulator-{self.instance_id}] Screen capture initialized with MSS")
            except Exception as e:
                print(f"[SonicEmulator-{self.instance_id}] Failed to initialize MSS: {e}")
                self.screen_capture = None
                self.capture_region = None
        else:
            print(f"[SonicEmulator-{self.instance_id}] MSS not available, screen capture disabled")
        
        # Input control - use ONLY isolated input system
        self.input_manager = None  # Will be initialized when needed
        # self.env_id = None  # DEPRECATED: Not needed with file-based system
        
        # Game state memory addresses (Sonic 1 actual addresses)
        self.memory_addresses = {
            'score': 0xFFE000,      # Score (4 bytes)
            'rings': 0xFFE002,      # Rings (2 bytes)
            'lives': 0xFFE004,      # Lives (2 bytes)
            'level': 0xFFE006,      # Current level (2 bytes)
            'position_x': 0xFFD030, # Sonic X position (2 bytes, signed)
            'position_y': 0xFFD038, # Sonic Y position (2 bytes, signed)
            'game_state': 0xFFE00C, # Game state/status (2 bytes)
            'invincibility': 0xFFE00E, # Invincibility timer (2 bytes)
            'speed': 0xFFD040,      # Sonic speed (2 bytes)
            'timer': 0xFFE010,      # Level timer (2 bytes)
            'zone': 0xFFE012,       # Current zone (2 bytes)
            'act': 0xFFE014,        # Current act (2 bytes)
            'shields': 0xFFE016,    # Shield status (2 bytes)
            'speed_shoes': 0xFFE018, # Speed shoes timer (2 bytes)
            'air_remaining': 0xFFE01A, # Air remaining underwater (2 bytes)
            'lamppost_counter': 0xFFE01C, # Lamppost counter (2 bytes)
            'emeralds': 0xFFE01E,   # Chaos emeralds collected (2 bytes)
            'angle': 0xFFD042,      # Sonic's angle (2 bytes)
            'status': 0xFFD044,     # Sonic's status (2 bytes)
            'level_timer_frames': 0xFFE020  # Level timer in frames (2 bytes)
        }
        
        # Memory address validation ranges
        self.memory_ranges = {
            'score': (0, 999999),           # Score range
            'rings': (0, 999),              # Rings range
            'lives': (0, 99),               # Lives range
            'level': (0, 15),               # Level range
            'position_x': (-32768, 32767),  # Signed 16-bit
            'position_y': (-32768, 32767),  # Signed 16-bit
            'game_state': (0, 65535),       # Unsigned 16-bit
            'invincibility': (0, 65535),    # Timer range
            'speed': (-32768, 32767),       # Signed 16-bit
            'timer': (0, 65535),            # Timer range
            'zone': (0, 15),                # Zone range
            'act': (0, 3),                  # Act range
            'shields': (0, 65535),          # Shield timer
            'speed_shoes': (0, 65535),      # Speed shoes timer
            'air_remaining': (0, 65535),    # Air timer
            'lamppost_counter': (0, 255),   # Counter range
            'emeralds': (0, 7),             # Emerald count
            'angle': (0, 255),              # Angle range
            'status': (0, 255),             # Status flags
            'level_timer_frames': (0, 65535) # Frame timer
        }
        
        # Initialize emulator
        self.launch()
        
        # Update capture region after launch
        self._update_capture_region()
        
        # Initialize input manager
        self._init_input_manager()
    
    def set_env_id(self, env_id: int):
        """Set the environment ID for input isolation."""
        # self.env_id = env_id  # DEPRECATED: Not needed with file-based system
        
        # Initialize input manager if not already done (shared per process)
        if self.input_manager is None:
            try:
                # Use a shared input manager for this process
                # The input manager should already be created by the main process
                self.input_manager = get_input_manager(num_instances=4)  # Support up to 4 instances
                print(f"[SonicEmulator-{self.instance_id}] Input manager initialized")
            except Exception as e:
                print(f"[SonicEmulator-{self.instance_id}] Failed to initialize input manager: {e}")
                self.input_manager = None
        
        # Assign environment to input manager
        if self.input_manager is not None:
            try:
                # self.input_manager.assign_environment(env_id, self.instance_id)  # DEPRECATED: Not needed with file-based system
                print(f"[SonicEmulator-{self.instance_id}] Assigned to environment {env_id}")
            except Exception as e:
                print(f"[SonicEmulator-{self.instance_id}] Failed to assign environment: {e}")
        else:
            print(f"[SonicEmulator-{self.instance_id}] Warning: No input manager available")
        
    def launch(self):
        if self.process:
            self.close()
        
        # Set environment variable for instance ID
        env = os.environ.copy()
        env['BIZHAWK_INSTANCE_ID'] = str(self.instance_id)
        # Share base directory for communication files with Lua script
        env['BIZHAWK_COMM_BASE'] = os.getcwd()
        
        cmd = [
            os.path.join(self.bizhawk_dir, "EmuHawk.exe"),
            f"--lua={self.lua_script_path}",
            str(self.rom_path)
        ]
        print(f"[SonicEmulator-{self.instance_id}] Launching BizHawk with command: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, env=env)
        time.sleep(5)  # Wait for BizHawk to start
    
    def read_memory(self, address, size):
        return self.memory_reader.read_memory(address, size)
    
    def close(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
    
    def get_screen(self) -> np.ndarray:
        """Capture the current screen from the emulator."""
        if not self.screen_capture or not self.capture_region:
            # Return a blank image if screen capture is not available
            print(f"[SonicEmulator-{self.instance_id}] Screen capture not available, returning blank image")
            return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        try:
            # Capture screen region
            screenshot = self.screen_capture.grab(self.capture_region)
            
            # Convert to numpy array
            img = np.array(screenshot)
            
            # Convert from BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Resize to target dimensions
            img = cv2.resize(img, (self.screen_width, self.screen_height))
            
            return img
        except Exception as e:
            print(f"[SonicEmulator-{self.instance_id}] Screen capture error: {e}")
            # Return a blank image on error
            return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
    
    def step(self, actions: List[str]):
        """Execute actions in the emulator using input manager."""
        if not actions:
            return
        
        try:
            # Use input manager if available
            if self.input_manager is not None:
                # Send actions through input manager
                for action in actions:
                    if action in ['LEFT', 'RIGHT', 'UP', 'DOWN', 'A', 'B', 'C', 'START']:
                        self.input_manager.send_action(action, duration=0.016)
                        print(f"[SonicEmulator-{self.instance_id}] Sent action via input manager: {action}")
            else:
                # Fallback to Lua bridge
                input_commands = []
                
                # Reset all inputs first
                self.memory_reader._send_command("ACTION:RESET_INPUTS")
                
                # Set the requested inputs to true
                for action in actions:
                    if action in ['LEFT', 'RIGHT', 'UP', 'DOWN', 'A', 'B', 'C', 'START']:
                        input_commands.append(f"{action}:true")
                
                if input_commands:
                    # Send inputs to Lua bridge
                    inputs_str = "|".join(input_commands)
                    self.memory_reader._send_command(f"ACTION:SET_INPUTS|INPUTS:{inputs_str}")
                    
                    print(f"[SonicEmulator-{self.instance_id}] Sent inputs via Lua bridge: {inputs_str}")
                
                # Small delay to let inputs take effect
                time.sleep(0.016)
            
        except Exception as e:
            print(f"[SonicEmulator-{self.instance_id}] Input error: {e}")
            # Fallback: just wait
            time.sleep(0.016)
    
    def reset(self):
        """Reset the emulator to the beginning of the game."""
        try:
            # Use input manager for reset if available
            if self.input_manager is not None:
                print(f"[SonicEmulator-{self.instance_id}] Using input manager for reset")
                # Send multiple START presses to reset
                for _ in range(3):
                    self.input_manager.send_action('START', duration=0.2)
                    time.sleep(0.1)
            else:
                # Fallback to Lua bridge
                print(f"[SonicEmulator-{self.instance_id}] Using Lua bridge for reset")
                self.memory_reader._send_command("ACTION:RESET_INPUTS")
                # Send START command
                self.memory_reader._send_command("ACTION:SET_INPUTS|INPUTS:START:true")
                time.sleep(0.2)
                self.memory_reader._send_command("ACTION:RESET_INPUTS")
        except Exception as e:
            print(f"[SonicEmulator-{self.instance_id}] Reset failed: {e}")
            # Fallback: just wait
            time.sleep(2)
        
        # Wait for reset to complete
        time.sleep(2)
    
    def save_state(self, path: str):
        """Save the current game state."""
        if self.input_manager is not None:
            self.input_manager.send_action('F5', duration=0.1)
        else:
            print(f"[SonicEmulator-{self.instance_id}] No input manager available, cannot save state")
        
        # Wait for save to complete
        time.sleep(0.5)
    
    def load_state(self, path: str):
        """Load a saved game state using input isolation."""
        if self.input_manager is not None:
            self.input_manager.send_action('F7', duration=0.1)
        else:
            print(f"[SonicEmulator-{self.instance_id}] Warning: No input manager available, skipping load")
        
        # Wait for load to complete
        time.sleep(0.5)
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()
    
    def _detect_act_clear_screen(self) -> bool:
        """Detect if we're in the ACT CLEAR screen."""
        try:
            # This would require screen capture and image analysis
            # For now, return False as a placeholder
            # In a real implementation, you would:
            # 1. Capture the screen
            # 2. Look for "ACT CLEAR" text or specific visual patterns
            # 3. Check for the characteristic end-of-act music/sounds
            return False
        except Exception as e:
            print(f"Error detecting act clear screen: {e}")
            return False

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
        """Initialize the input manager if it hasn't been initialized yet."""
        if self.input_manager is None:
            try:
                # Use instance-specific input manager
                self.input_manager = get_input_manager(num_instances=4, instance_id=self.instance_id)
                print(f"[SonicEmulator-{self.instance_id}] Input manager initialized for instance {self.instance_id}")
            except Exception as e:
                print(f"[SonicEmulator-{self.instance_id}] Failed to initialize input manager: {e}")
                self.input_manager = None 