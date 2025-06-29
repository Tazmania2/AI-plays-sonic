#!/usr/bin/env python3
"""
BizHawk Emulator Interface
Communicates with BizHawk via file-based Lua bridge for input injection and memory reading.
"""

import os
import json
import time
import threading
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BizHawkEmulator:
    """
    BizHawk emulator interface using file-based Lua bridge for communication.
    """
    
    def __init__(self, 
                 bizhawk_path: str = r"C:\Program Files (x86)\BizHawk-2.10-win-x64\EmuHawk.exe",
                 lua_script: str = "emulator/bizhawk_bridge.lua",
                 rom_path: str = "roms/Sonic The Hedgehog (USA, Europe).md",
                 comm_dir: str = "D:/AI tests/bizhawk_comm",
                 timeout: float = 5.0):
        """
        Initialize BizHawk emulator interface.
        
        Args:
            bizhawk_path: Path to BizHawk executable
            lua_script: Path to Lua bridge script
            rom_path: Path to Sonic ROM
            comm_dir: Communication directory for file-based bridge
            timeout: Communication timeout
        """
        self.bizhawk_path = bizhawk_path
        self.lua_script = lua_script
        self.rom_path = rom_path
        self.comm_dir = comm_dir
        self.timeout = timeout
        
        # File paths for communication
        self.request_file = os.path.join(comm_dir, "request.txt")
        self.response_file = os.path.join(comm_dir, "response.txt")
        self.status_file = os.path.join(comm_dir, "status.txt")
        
        self.is_connected = False
        self.process = None
        self.lock = threading.Lock()
        
        # Validate paths
        self._validate_paths()
        
    def _validate_paths(self):
        """Validate that all required files exist."""
        if not os.path.exists(self.bizhawk_path):
            raise FileNotFoundError(f"BizHawk not found at: {self.bizhawk_path}")
        
        if not os.path.exists(self.lua_script):
            raise FileNotFoundError(f"Lua script not found at: {self.lua_script}")
            
        if not os.path.exists(self.rom_path):
            raise FileNotFoundError(f"ROM not found at: {self.rom_path}")
    
    def start(self) -> bool:
        """
        Start BizHawk with the Lua bridge script.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            logger.info("Starting BizHawk with Lua bridge...")
            
            # Build command
            cmd = [
                self.bizhawk_path,
                "--lua=" + os.path.abspath(self.lua_script),
                os.path.abspath(self.rom_path)
            ]
            
            # Start BizHawk process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            
            logger.info(f"BizHawk started with PID: {self.process.pid}")
            
            # Wait for Lua script to initialize and create status file
            return self._wait_for_bridge()
            
        except Exception as e:
            logger.error(f"Failed to start BizHawk: {e}")
            return False
    
    def _wait_for_bridge(self) -> bool:
        """
        Wait for the Lua bridge to be ready by checking for status file.
        
        Returns:
            True if bridge is ready, False otherwise
        """
        logger.info("Waiting for Lua bridge to initialize...")
        
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            if os.path.exists(self.status_file):
                try:
                    with open(self.status_file, 'r') as f:
                        status = f.read().strip()
                    if status == "READY":
                        logger.info("Lua bridge is ready")
                        self.is_connected = True
                        return True
                except:
                    pass
            time.sleep(0.1)
        
        logger.error("Timeout waiting for Lua bridge")
        return False
    
    def _send_command(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send a command to BizHawk via file-based communication.
        
        Args:
            command: Command dictionary
            
        Returns:
            Response dictionary or None if failed
        """
        if not self.is_connected:
            logger.error("Not connected to BizHawk bridge")
            return None
        
        try:
            with self.lock:
                # Convert command to simple string format
                command_str = self._encode_command(command)
                
                # Write request file
                with open(self.request_file, 'w') as f:
                    f.write(command_str)
                
                # Wait for response with longer timeout
                start_time = time.time()
                timeout = 10.0  # Increased timeout
                
                while time.time() - start_time < timeout:
                    if os.path.exists(self.response_file):
                        # Read response
                        with open(self.response_file, 'r') as f:
                            response_data = f.read().strip()
                        
                        # Delete response file
                        os.remove(self.response_file)
                        
                        # Parse response
                        response = self._decode_response(response_data)
                        return response
                    
                    time.sleep(0.05)  # Increased delay
                
                logger.error("Timeout waiting for response")
                return None
                    
        except Exception as e:
            logger.error(f"Communication error: {e}")
            self.is_connected = False
            return None
        
        return None
    
    def _encode_command(self, command: Dict[str, Any]) -> str:
        """Encode command dictionary to simple string format."""
        parts = []
        
        if command.get("action") == "ping":
            parts.append("ACTION:PING")
        elif command.get("action") == "get_state":
            parts.append("ACTION:GET_STATE")
        elif command.get("action") == "set_input":
            parts.append("ACTION:SET_INPUT")
            parts.append(f"INPUT:{command.get('input', '')}")
            parts.append(f"STATE:{str(command.get('state', False)).lower()}")
        elif command.get("action") == "set_inputs":
            parts.append("ACTION:SET_INPUTS")
            inputs = command.get("inputs", {})
            input_parts = []
            for input_name, state in inputs.items():
                input_parts.append(f"{input_name}:{str(state).lower()}")
            parts.append(f"INPUTS:{'|'.join(input_parts)}")
        elif command.get("action") == "reset_inputs":
            parts.append("ACTION:RESET_INPUTS")
        
        return "|".join(parts)
    
    def _decode_response(self, response_str: str) -> Dict[str, Any]:
        """Decode response string to dictionary."""
        response = {"success": False, "data": None, "error": None}
        
        for part in response_str.split("|"):
            if part.startswith("SUCCESS:"):
                response["success"] = part.split(":", 1)[1] == "true"
            elif part.startswith("DATA:"):
                response["data"] = part.split(":", 1)[1]
            elif part.startswith("ERROR:"):
                response["error"] = part.split(":", 1)[1]
        
        return response
    
    def ping(self) -> bool:
        """
        Test connection to BizHawk bridge.
        
        Returns:
            True if connection is working, False otherwise
        """
        response = self._send_command({"action": "ping"})
        if response and response.get("success"):
            logger.info("Ping successful")
            return True
        else:
            logger.error("Ping failed")
            return False
    
    def get_game_state(self) -> Optional[Dict[str, Any]]:
        """
        Get current game state from BizHawk.
        
        Returns:
            Game state dictionary or None if failed
        """
        response = self._send_command({"action": "get_state"})
        if response and response.get("success"):
            # Parse the simplified state string: "x:123,y:456,rings:5,lives:3,level:1"
            state_data = response.get("data", "")
            state = {}
            
            for part in state_data.split(","):
                if ":" in part:
                    key, value = part.split(":", 1)
                    try:
                        state[key] = int(value)
                    except ValueError:
                        state[key] = value
            
            return state
        else:
            logger.error(f"Failed to get game state: {response}")
            return None
    
    def set_input(self, input_name: str, state: bool) -> bool:
        """
        Set a single input state.
        
        Args:
            input_name: Input name (UP, DOWN, LEFT, RIGHT, A, B, C, START)
            state: True for pressed, False for released
            
        Returns:
            True if successful, False otherwise
        """
        response = self._send_command({
            "action": "set_input",
            "input": input_name.upper(),
            "state": state
        })
        
        if response and response.get("success"):
            logger.debug(f"Set {input_name} = {state}")
            return True
        else:
            logger.error(f"Failed to set input {input_name}: {response}")
            return False
    
    def set_inputs(self, inputs: Dict[str, bool]) -> bool:
        """
        Set multiple input states at once.
        
        Args:
            inputs: Dictionary of input_name -> state
            
        Returns:
            True if successful, False otherwise
        """
        # Convert input names to uppercase
        inputs_upper = {k.upper(): v for k, v in inputs.items()}
        
        response = self._send_command({
            "action": "set_inputs",
            "inputs": inputs_upper
        })
        
        if response and response.get("success"):
            logger.debug(f"Set inputs: {inputs_upper}")
            return True
        else:
            logger.error(f"Failed to set inputs: {response}")
            return False
    
    def reset_inputs(self) -> bool:
        """
        Reset all inputs to released state.
        
        Returns:
            True if successful, False otherwise
        """
        response = self._send_command({"action": "reset_inputs"})
        if response and response.get("success"):
            logger.debug("Reset all inputs")
            return True
        else:
            logger.error(f"Failed to reset inputs: {response}")
            return False
    
    def step(self, action: Optional[Dict[str, bool]] = None) -> Optional[Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Dictionary of input states (optional)
            
        Returns:
            Game state after step or None if failed
        """
        # Set inputs if provided
        if action:
            self.set_inputs(action)
        
        # Get game state
        return self.get_game_state()
    
    def close(self):
        """Close the emulator and clean up."""
        logger.info("Closing BizHawk emulator...")
        
        # Reset connection status
        self.is_connected = False
        
        # Terminate BizHawk process
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except:
                pass
            self.process = None
        
        logger.info("BizHawk emulator closed")
    
    def __enter__(self):
        """Context manager entry."""
        if not self.start():
            raise RuntimeError("Failed to start BizHawk emulator")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Test function
def test_bizhawk_emulator():
    """Test the BizHawk emulator interface."""
    print("🧪 Testing BizHawk Emulator Interface")
    print("=" * 50)
    
    try:
        # Create emulator instance
        emu = BizHawkEmulator()
        
        # Start emulator
        if not emu.start():
            print("❌ Failed to start BizHawk")
            return
        
        print("✅ BizHawk started successfully")
        
        # Test ping
        if not emu.ping():
            print("❌ Ping failed")
            return
        
        print("✅ Connection established")
        
        # Test input injection
        print("\n🎮 Testing input injection...")
        
        # Test individual inputs
        for input_name in ["LEFT", "RIGHT", "A", "B", "START"]:
            print(f"  Testing {input_name}...")
            emu.set_input(input_name, True)
            time.sleep(0.1)
            emu.set_input(input_name, False)
            time.sleep(0.1)
        
        # Test multiple inputs
        print("  Testing multiple inputs...")
        emu.set_inputs({
            "LEFT": True,
            "A": True
        })
        time.sleep(0.2)
        emu.reset_inputs()
        
        # Get game state
        print("\n📊 Getting game state...")
        state = emu.get_game_state()
        if state:
            print(f"  Sonic X: {state.get('x', 'N/A')}")
            print(f"  Sonic Y: {state.get('y', 'N/A')}")
            print(f"  Rings: {state.get('rings', 'N/A')}")
            print(f"  Lives: {state.get('lives', 'N/A')}")
            print(f"  Level: {state.get('level', 'N/A')}")
        else:
            print("  ❌ Failed to get game state")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        # Clean up
        if 'emu' in locals():
            emu.close()


if __name__ == "__main__":
    test_bizhawk_emulator() 