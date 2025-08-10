import win32gui
import win32con
import win32api
import win32process
import psutil
import time
import threading
from typing import Dict, List, Optional, Tuple
import ctypes
from ctypes import wintypes
import numpy as np

# Windows API constants
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_CHAR = 0x0102
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105

# Virtual key codes
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_RETURN = 0x0D
VK_SPACE = 0x20
VK_X = 0x58
VK_Z = 0x5A
VK_A = 0x41
VK_S = 0x53
VK_Q = 0x51
VK_W = 0x57
VK_C = 0x43
VK_P = 0x50
VK_F1 = 0x70
VK_F5 = 0x74
VK_F7 = 0x76

class DirectInputManager:
    """
    Direct input injection system using Windows API.
    This provides the primary input method for AI control of BizHawk.
    """
    
    def __init__(self, instance_id: int = 0):
        self.instance_id = instance_id
        self.target_hwnd = None
        self.target_process_id = None
        self.input_thread = None
        self.input_queue = []
        self.max_queue_size = 100  # Prevent queue overflow
        self.running = False
        self.lock = threading.Lock()
        
        # Action to virtual key mapping (BizHawk Genesis/Mega Drive)
        self.action_to_vk = {
            'NOOP': None,
            'LEFT': VK_LEFT,
            'RIGHT': VK_RIGHT,
            'UP': VK_UP,
            'DOWN': VK_DOWN,
            'A': VK_Z,      # Genesis A button (Jump)
            'B': VK_X,      # Genesis B button (Spin dash)
            'C': VK_C,      # Genesis C button
            'START': VK_RETURN,
            'SELECT': VK_SPACE,
            'L': VK_Q,      # Left shoulder
            'R': VK_W,      # Right shoulder
            # Alternative mappings for different emulator configurations
            'JUMP': VK_Z,   # Alternative jump key
            'SPIN': VK_X,   # Alternative spin key
            'PAUSE': VK_P,  # Pause key
            'RESET': VK_F1, # Reset key
            'SAVE': VK_F5,  # Save state
            'LOAD': VK_F7   # Load state
        }
        
        # Find and target the specific BizHawk instance
        self._find_target_window()
    
    def _find_target_window(self):
        """Find the BizHawk window for this specific instance."""
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                window_class = win32gui.GetClassName(hwnd)
                try:
                    _, process_id = win32process.GetWindowThreadProcessId(hwnd)
                    process = psutil.Process(process_id)
                    process_name = process.name()
                except Exception:
                    process_name = ''
                
                # Accept both BizHawk.Client.EmuHawk.exe and EmuHawk.exe
                pname = process_name.lower()
                if pname in ('bizhawk.client.emuhawk.exe', 'emuhawk.exe'):
                    # More flexible title matching - look for game-related content
                    title = window_text.lower()
                    # Accept windows that contain game-related terms but exclude file explorer
                    if (('sonic' in title or 'genesis' in title or 'hedgehog' in title or 
                         'game' in title or 'emulator' in title or 'bizhawk' in title) and
                        'explorador de arquivos' not in title and
                        'file explorer' not in title and
                        'explorer' not in title):
                        windows.append((hwnd, process_id, window_text, window_class, process_name))
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        # Sort by process ID and then by window handle for consistent ordering
        windows.sort(key=lambda x: (x[1], x[0]))
        
        if len(windows) > self.instance_id:
            self.target_hwnd = windows[self.instance_id][0]
            self.target_process_id = windows[self.instance_id][1]
            print(f"[DirectInputManager-{self.instance_id}] Targeted window: {windows[self.instance_id][2]} (PID: {self.target_process_id}, Process: {windows[self.instance_id][4]})")
        else:
            print(f"[DirectInputManager-{self.instance_id}] No BizHawk game window found for instance {self.instance_id}")
            print(f"[DirectInputManager-{self.instance_id}] Available BizHawk game windows: {[w[2] for w in windows]}")
            # Print all visible windows and process names for debugging
            all_windows = []
            def debug_enum(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    text = win32gui.GetWindowText(hwnd)
                    try:
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        pname = psutil.Process(pid).name()
                    except Exception:
                        pname = ''
                    all_windows.append(f'"{text}" (Process: {pname})')
                return True
            win32gui.EnumWindows(debug_enum, None)
            print(f"[DirectInputManager-{self.instance_id}] All visible windows: {all_windows}")
    
    def refresh_window_target(self):
        """Refresh the window target (useful if windows change)."""
        print(f"[DirectInputManager-{self.instance_id}] Refreshing window target...")
        self._find_target_window()
    
    def _send_input_to_window(self, vk_code: int, key_down: bool = True):
        """Send input directly to the target window."""
        if not self.target_hwnd or not win32gui.IsWindow(self.target_hwnd):
            return False
        
        try:
            # Try to bring window to foreground (but don't fail if it doesn't work)
            try:
                win32gui.SetForegroundWindow(self.target_hwnd)
                win32gui.ShowWindow(self.target_hwnd, win32con.SW_RESTORE)
            except:
                pass  # Continue even if focus fails
            
            # Small delay to ensure window is ready
            time.sleep(0.01)
            
            # Send the key message
            message = WM_KEYDOWN if key_down else WM_KEYUP
            result = win32gui.PostMessage(self.target_hwnd, message, vk_code, 0)
            
            if result == 0:
                print(f"[DirectInputManager-{self.instance_id}] Failed to send message to window")
                return False
            
            return True
        except Exception as e:
            print(f"[DirectInputManager-{self.instance_id}] Error sending input: {e}")
            return False
    
    def _input_worker(self):
        """Background thread for processing input queue."""
        while self.running:
            with self.lock:
                if self.input_queue:
                    action, duration = self.input_queue.pop(0)
                    
                    # Process the action
                    vk_code = self.action_to_vk.get(action)
                    if vk_code is not None:
                        # Press key
                        success = self._send_input_to_window(vk_code, True)
                        if success:
                            print(f"[DirectInputManager-{self.instance_id}] Sent {action} (key down)")
                        
                        # Hold for specified duration
                        time.sleep(duration)
                        
                        # Release key
                        success = self._send_input_to_window(vk_code, False)
                        if success:
                            print(f"[DirectInputManager-{self.instance_id}] Sent {action} (key up)")
                    else:
                        # NOOP or unknown action
                        time.sleep(duration)
                else:
                    time.sleep(0.001)  # 1ms sleep when no input
    
    def start(self):
        """Start the input processing thread."""
        if not self.running:
            self.running = True
            self.input_thread = threading.Thread(target=self._input_worker, daemon=True)
            self.input_thread.start()
            print(f"[DirectInputManager-{self.instance_id}] Input thread started")
    
    def stop(self):
        """Stop the input processing thread."""
        self.running = False
        if self.input_thread:
            self.input_thread.join(timeout=1.0)
            print(f"[DirectInputManager-{self.instance_id}] Input thread stopped")
    
    def send_action(self, action: str, duration: float = 0.016):
        """Send a single action to the target window."""
        if len(self.input_queue) < self.max_queue_size:
            with self.lock:
                self.input_queue.append((action, duration))
    
    def send_actions(self, actions: List[str], duration: float = 0.016):
        """Send multiple actions to the target window."""
        for action in actions:
            self.send_action(action, duration)
    
    def clear_queue(self):
        """Clear the input queue."""
        with self.lock:
            self.input_queue.clear()
    
    def is_window_active(self) -> bool:
        """Check if the target window is still active."""
        return (self.target_hwnd is not None and 
                win32gui.IsWindow(self.target_hwnd) and
                win32gui.IsWindowVisible(self.target_hwnd))


class MultiInstanceDirectInputManager:
    """Manages multiple direct input managers for parallel training environments."""
    
    def __init__(self, num_instances: int = 4):
        self.num_instances = num_instances
        self.input_managers = {}
        self.env_to_instance = {}  # Map environment ID to instance ID
        
        # Create input managers
        for i in range(num_instances):
            self.input_managers[i] = DirectInputManager(i)
            self.input_managers[i].start()
        
        print(f"[MultiInstanceDirectInputManager] Created {num_instances} direct input managers")
    
    def assign_environment(self, env_id: int, instance_id: int):
        """Assign an environment to a specific input instance."""
        if instance_id < self.num_instances:
            self.env_to_instance[env_id] = instance_id
            print(f"[MultiInstanceDirectInputManager] Assigned environment {env_id} to instance {instance_id}")
    
    def send_action(self, env_id: int, action: str, duration: float = 0.016):
        """Send action to the specific environment's input instance."""
        if env_id in self.env_to_instance:
            instance_id = self.env_to_instance[env_id]
            self.input_managers[instance_id].send_action(action, duration)
        else:
            print(f"[MultiInstanceDirectInputManager] Environment {env_id} not assigned to any instance")
    
    def send_actions(self, env_id: int, actions: List[str], duration: float = 0.016):
        """Send multiple actions to the specific environment's input instance."""
        if env_id in self.env_to_instance:
            instance_id = self.env_to_instance[env_id]
            self.input_managers[instance_id].send_actions(actions, duration)
        else:
            print(f"[MultiInstanceDirectInputManager] Environment {env_id} not assigned to any instance")
    
    def get_instance_status(self) -> Dict[int, bool]:
        """Get the status of all input instances."""
        return {i: manager.is_window_active() for i, manager in self.input_managers.items()}
    
    def shutdown(self):
        """Shutdown all input managers."""
        for manager in self.input_managers.values():
            manager.stop()
        print("[MultiInstanceDirectInputManager] All direct input managers shutdown")


# Global direct input manager instance
_global_direct_input_manager = None

def get_direct_input_manager(num_instances: int = 4) -> MultiInstanceDirectInputManager:
    """Get or create the global direct input manager."""
    global _global_direct_input_manager
    if _global_direct_input_manager is None:
        _global_direct_input_manager = MultiInstanceDirectInputManager(num_instances)
    return _global_direct_input_manager

def shutdown_direct_input_manager():
    """Shutdown the global direct input manager."""
    global _global_direct_input_manager
    if _global_direct_input_manager is not None:
        _global_direct_input_manager.shutdown()
        _global_direct_input_manager = None
