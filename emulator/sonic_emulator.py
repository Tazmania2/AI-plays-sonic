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

# For screen capture (cross-platform)
try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

# For keyboard/mouse control
try:
    from pynput import keyboard, mouse
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


class SonicEmulator:
    """
    Wrapper for Sonic game emulator.
    
    This class provides a unified interface to control Sonic games
    through various emulators (BizHawk, RetroArch, etc.).
    """
    
    def __init__(self, rom_path: str, screen_width: int = 224, screen_height: int = 256, core_path: str = None):
        self.rom_path = Path(rom_path)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.core_path = core_path  # Add core_path to the emulator
        
        # Emulator process
        self.emulator_process = None
        self.emulator_type = self._detect_emulator()
        
        # Screen capture
        self.screen_capture = None
        self.capture_region = None
        
        # Input control
        self.keyboard_controller = None
        self.mouse_controller = None
        
        # Game state memory addresses (example for Sonic 1)
        self.memory_addresses = {
            'score': 0xFFE000,
            'rings': 0xFFE002,
            'lives': 0xFFE004,
            'level': 0xFFE006,
            'position_x': 0xFFE008,
            'position_y': 0xFFE00A,
            'game_state': 0xFFE00C,
            'invincibility': 0xFFE00E,
            'speed': 0xFFE010
        }
        
        # Action mappings (updated for user's RetroArch key bindings)
        from pynput.keyboard import Key, KeyCode
        self.action_mappings = {
            'NOOP': [],
            'UP': [Key.up],
            'DOWN': [Key.down],
            'LEFT': [Key.left],
            'RIGHT': [Key.right],
            'A': [KeyCode.from_char('x')],  # A (direita)
            'B': [KeyCode.from_char('z')],  # B (baixo)
            'Y': [KeyCode.from_char('a')],  # Y (esquerda)
            'X': [KeyCode.from_char('s')],  # X (topo)
            'START': [Key.enter],
            'SELECT': [Key.shift_r],
            'L': [KeyCode.from_char('q')],
            'R': [KeyCode.from_char('w')]
        }
        
        # Initialize emulator
        self._initialize_emulator()
        
    def _detect_emulator(self) -> str:
        """Detect which emulator to use based on available software."""
        # Check for common emulators
        emulators = {
            'bizhawk': ['EmuHawk.exe', 'bizhawk'],
            'retroarch': ['retroarch.exe', 'retroarch'],
            'fusion': ['Fusion.exe', 'kega'],
            'gens': ['gens.exe', 'gens']
        }
        
        for emulator, executables in emulators.items():
            for exe in executables:
                if self._find_executable(exe):
                    return emulator
        
        # Default to BizHawk if available
        return 'bizhawk'
    
    def _find_executable(self, name: str) -> bool:
        """Check if an executable is available in PATH or common locations."""
        import shutil
        
        # Check PATH
        if shutil.which(name):
            return True
        
        # Check common installation directories
        common_paths = [
            "C:\\Program Files\\BizHawk",
            "C:\\Program Files (x86)\\BizHawk",
            "C:\\Program Files\\RetroArch",
            "C:\\Program Files (x86)\\RetroArch",
            "C:\\RetroArch-Win64",  # User's specific installation
            os.path.expanduser("~/AppData/Local/Programs/BizHawk"),
            os.path.expanduser("~/AppData/Local/Programs/RetroArch")
        ]
        
        for path in common_paths:
            if os.path.exists(os.path.join(path, name)):
                return True
        
        return False
    
    def _initialize_emulator(self):
        """Initialize the emulator and set up screen capture."""
        if not self.rom_path.exists():
            raise FileNotFoundError(f"ROM file not found: {self.rom_path}")
        
        # Start emulator
        self._start_emulator()
        
        # Wait for emulator to load
        time.sleep(3)
        
        # Set up screen capture
        self._setup_screen_capture()
        
        # Set up input control
        self._setup_input_control()
        
        # Load ROM
        self._load_rom()
        
    def _start_emulator(self):
        """Start the emulator process."""
        if self.emulator_type == 'bizhawk':
            self._start_bizhawk()
        elif self.emulator_type == 'retroarch':
            self._start_retroarch()
        else:
            raise NotImplementedError(f"Emulator type {self.emulator_type} not supported")
    
    def _start_bizhawk(self):
        """Start BizHawk emulator."""
        try:
            # Try to find BizHawk executable
            bizhawk_path = None
            possible_paths = [
                "C:\\Program Files\\BizHawk\\EmuHawk.exe",
                "C:\\Program Files (x86)\\BizHawk\\EmuHawk.exe",
                os.path.expanduser("~/AppData/Local/Programs/BizHawk/EmuHawk.exe")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    bizhawk_path = path
                    break
            
            if not bizhawk_path:
                raise FileNotFoundError("BizHawk not found. Please install BizHawk first.")
            
            # Start BizHawk
            self.emulator_process = subprocess.Popen([
                bizhawk_path,
                "--loadrom", str(self.rom_path),
                "--windowed",
                "--autostart"
            ])
            
        except Exception as e:
            print(f"Failed to start BizHawk: {e}")
            print("Please install BizHawk or use a different emulator.")
            raise
    
    def _start_retroarch(self):
        """Start RetroArch emulator."""
        try:
            retroarch_path = None
            possible_paths = [
                "C:\\Program Files\\RetroArch\\retroarch.exe",
                "C:\\Program Files (x86)\\RetroArch\\retroarch.exe",
                "C:\\RetroArch-Win64\\retroarch.exe",  # User's specific installation
                os.path.expanduser("~/AppData/Local/Programs/RetroArch/retroarch.exe")
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    retroarch_path = path
                    break
            if not retroarch_path:
                raise FileNotFoundError("RetroArch not found. Please install RetroArch or use a different emulator.")
            # Use the core path from config or default location
            core_path = self.core_path or "cores/genesis_plus_gx_libretro.dll"
            if not os.path.isabs(core_path):
                # Try common core locations
                core_possible_paths = [
                    os.path.join(os.path.dirname(retroarch_path), "cores", core_path),
                    os.path.join("C:\\RetroArch-Win64\\cores", core_path),
                    os.path.join("C:\\Program Files\\RetroArch\\cores", core_path),
                    os.path.join("C:\\Program Files (x86)\\RetroArch\\cores", core_path)
                ]
                for cpath in core_possible_paths:
                    if os.path.exists(cpath):
                        core_path = cpath
                        break
            if not os.path.exists(core_path):
                raise FileNotFoundError(f"RetroArch core not found: {core_path}")
            # Start RetroArch with correct order
            self.emulator_process = subprocess.Popen([
                retroarch_path,
                "-L", core_path,
                str(self.rom_path)
            ])
        except Exception as e:
            print(f"Failed to start RetroArch: {e}")
            print("Please install RetroArch or use a different emulator.")
            raise
    
    def _setup_screen_capture(self):
        """Set up screen capture for the emulator window."""
        if not MSS_AVAILABLE:
            raise ImportError("mss library required for screen capture. Install with: pip install mss")
        
        self.screen_capture = mss.mss()
        
        # Find emulator window
        self.capture_region = self._find_emulator_window()
        
        if not self.capture_region:
            raise RuntimeError("Could not find emulator window")
    
    def _find_emulator_window(self) -> Optional[Dict[str, int]]:
        """Find the emulator window on screen."""
        import win32gui
        import win32con
        
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if any(name in window_text.lower() for name in ['bizhawk', 'retroarch', 'sonic']):
                    rect = win32gui.GetWindowRect(hwnd)
                    windows.append({
                        'hwnd': hwnd,
                        'left': rect[0],
                        'top': rect[1],
                        'width': rect[2] - rect[0],
                        'height': rect[3] - rect[1],
                        'title': window_text
                    })
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        if windows:
            # Use the first found window
            window = windows[0]
            return {
                'left': window['left'],
                'top': window['top'],
                'width': window['width'],
                'height': window['height']
            }
        
        return None
    
    def _setup_input_control(self):
        """Set up keyboard and mouse control."""
        if not PYNPUT_AVAILABLE:
            raise ImportError("pynput library required for input control. Install with: pip install pynput")
        
        self.keyboard_controller = keyboard.Controller()
        self.mouse_controller = mouse.Controller()
    
    def _load_rom(self):
        """Load the ROM file into the emulator."""
        # ROM should already be loaded when starting emulator
        # Wait a bit for the game to fully load
        time.sleep(2)
    
    def get_screen(self) -> np.ndarray:
        """Capture the current screen from the emulator."""
        if not self.capture_region:
            raise RuntimeError("Screen capture not initialized")
        
        # Capture screen region
        screenshot = self.screen_capture.grab(self.capture_region)
        
        # Convert to numpy array
        img = np.array(screenshot)
        
        # Convert from BGRA to BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Resize to target dimensions
        img = cv2.resize(img, (self.screen_width, self.screen_height))
        
        return img
    
    def step(self, actions: List[str]):
        """Execute actions in the emulator."""
        if not self.keyboard_controller:
            raise RuntimeError("Input control not initialized")
        
        # Release all keys first
        self._release_all_keys()
        
        # Press the specified keys
        for action in actions:
            if action in self.action_mappings:
                keys = self.action_mappings[action]
                for key in keys:
                    self.keyboard_controller.press(key)
        
        # Small delay to ensure action is registered
        time.sleep(0.016)  # ~60 FPS
    
    def _release_all_keys(self):
        """Release all currently pressed keys."""
        for keys in self.action_mappings.values():
            for key in keys:
                try:
                    self.keyboard_controller.release(key)
                except:
                    pass
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get the current game state from memory."""
        # This is a simplified version. In a real implementation,
        # you would read memory addresses from the emulator process.
        
        # For now, return basic state information
        # In practice, you'd use memory reading libraries or emulator APIs
        return {
            'score': 0,  # Would read from memory
            'rings': 0,  # Would read from memory
            'lives': 3,  # Would read from memory
            'level': 1,  # Would read from memory
            'position': (0, 0),  # Would read from memory
            'game_state': 'playing',  # Would read from memory
            'invincibility': False,  # Would read from memory
            'speed': 0,  # Would read from memory
            'level_completed': False,  # Would read from memory
            'game_over': False  # Would read from memory
        }
    
    def reset(self):
        """Reset the emulator to the beginning of the game."""
        # Press F1 to reset (common emulator reset key)
        self.keyboard_controller.press(keyboard.Key.f1)
        time.sleep(0.1)
        self.keyboard_controller.release(keyboard.Key.f1)
        
        # Wait for reset to complete
        time.sleep(2)
    
    def save_state(self, path: str):
        """Save the current game state."""
        # Press F5 to save state (common emulator save key)
        self.keyboard_controller.press(keyboard.Key.f5)
        time.sleep(0.1)
        self.keyboard_controller.release(keyboard.Key.f5)
        
        # Wait for save to complete
        time.sleep(0.5)
    
    def load_state(self, path: str):
        """Load a saved game state."""
        # Press F7 to load state (common emulator load key)
        self.keyboard_controller.press(keyboard.Key.f7)
        time.sleep(0.1)
        self.keyboard_controller.release(keyboard.Key.f7)
        
        # Wait for load to complete
        time.sleep(0.5)
    
    def close(self):
        """Close the emulator."""
        if self.emulator_process:
            self.emulator_process.terminate()
            self.emulator_process.wait()
        
        if self.screen_capture:
            self.screen_capture.close()
        
        if self.keyboard_controller:
            self._release_all_keys()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close() 