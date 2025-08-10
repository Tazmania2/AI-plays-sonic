#!/usr/bin/env python3
"""
Test script to verify that the fixed Lua bridge properly handles inputs.
"""

import os
import time
import subprocess
from pathlib import Path

def test_fixed_inputs():
    """Test sending inputs with the fixed Lua bridge."""
    print("Testing fixed BizHawk input injection...")
    
    # Set up communication directory
    comm_dir = Path("bizhawk_comm_0")
    comm_dir.mkdir(exist_ok=True)
    
    request_file = comm_dir / "request.txt"
    response_file = comm_dir / "response.txt"
    status_file = comm_dir / "status.txt"
    
    # Clean up any existing files
    for file in [request_file, response_file, status_file]:
        if file.exists():
            file.unlink()
    
    # Find BizHawk and ROM
    bizhawk_exe = r"C:\Program Files (x86)\BizHawk-2.10-win-x64\EmuHawk.exe"
    rom_file = "roms/sonic1.md"
    lua_script = "emulator/bizhawk_bridge_fixed.lua"  # Use the fixed version
    
    if not all(os.path.exists(p) for p in [bizhawk_exe, rom_file, lua_script]):
        print("Required files not found!")
        return False
    
    # Set environment variables
    env = os.environ.copy()
    env['BIZHAWK_INSTANCE_ID'] = '0'
    env['BIZHAWK_COMM_BASE'] = os.getcwd()
    
    print(f"Launching BizHawk with fixed Lua bridge...")
    
    # Launch BizHawk
    cmd = [bizhawk_exe, f"--lua={lua_script}", rom_file]
    process = subprocess.Popen(cmd, env=env)
    
    print("Waiting for BizHawk to start...")
    
    # Wait for status file to appear
    max_wait = 30
    for i in range(max_wait):
        if status_file.exists():
            with open(status_file, 'r') as f:
                status = f.read().strip()
            print(f"✓ Status file found: {status}")
            break
        time.sleep(1)
        print(f"Waiting... ({i+1}/{max_wait})")
    else:
        print("✗ Status file not found after 30 seconds")
        process.terminate()
        return False
    
    # Wait a bit more for the game to fully load
    print("Waiting for game to load...")
    time.sleep(5)
    
    # Test different input sequences with proper format
    test_inputs = [
        ("RIGHT:true", "Moving right"),
        ("A:true", "Jumping"),
        ("RIGHT:true|A:true", "Jumping right"),
        ("LEFT:true", "Moving left"),
        ("DOWN:true|B:true", "Spin dash"),
        ("START:true", "Start button"),
    ]
    
    for inputs, description in test_inputs:
        print(f"\nTesting: {description}")
        print(f"Sending inputs: {inputs}")
        
        # Send input command
        with open(request_file, 'w') as f:
            f.write(f"ACTION:SET_INPUTS|INPUTS:{inputs}\n")
        
        # Wait for response
        max_wait = 5
        response_received = False
        for i in range(max_wait):
            if response_file.exists():
                with open(response_file, 'r') as f:
                    response = f.read().strip()
                print(f"✓ Response: {response}")
                response_received = True
                break
            time.sleep(1)
        
        if not response_received:
            print("✗ No response received")
        
        # Hold input for a moment
        time.sleep(1.0)
        
        # Reset inputs
        with open(request_file, 'w') as f:
            f.write("ACTION:RESET_INPUTS\n")
        
        time.sleep(0.5)
    
    # Test continuous right movement
    print("\nTesting continuous right movement...")
    for i in range(20):  # 2 seconds at 10fps
        with open(request_file, 'w') as f:
            f.write("ACTION:SET_INPUTS|INPUTS:RIGHT:true\n")
        time.sleep(0.1)
    
    # Reset
    with open(request_file, 'w') as f:
        f.write("ACTION:RESET_INPUTS\n")
    
    print("\nFixed input test completed!")
    print("Check the BizHawk window to see if Sonic is responding to inputs.")
    print("Press Enter to continue...")
    input()
    
    # Clean up
    print("Cleaning up...")
    process.terminate()
    process.wait()
    
    # Clean up files
    for file in [request_file, response_file, status_file]:
        if file.exists():
            file.unlink()
    
    print("✓ Fixed input test completed!")
    return True

if __name__ == "__main__":
    test_fixed_inputs()
