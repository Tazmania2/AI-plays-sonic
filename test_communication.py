#!/usr/bin/env python3
"""
Simple test script to verify file-based communication with BizHawk.
"""

import os
import time
import subprocess
from pathlib import Path

def test_communication():
    """Test the file-based communication system."""
    print("Testing BizHawk file-based communication...")
    
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
    lua_script = "emulator/bizhawk_bridge.lua"
    
    if not all(os.path.exists(p) for p in [bizhawk_exe, rom_file, lua_script]):
        print("Required files not found!")
        return False
    
    # Set environment variables
    env = os.environ.copy()
    env['BIZHAWK_INSTANCE_ID'] = '0'
    env['BIZHAWK_COMM_BASE'] = os.getcwd()
    
    print(f"Launching BizHawk with:")
    print(f"  BizHawk: {bizhawk_exe}")
    print(f"  ROM: {rom_file}")
    print(f"  Lua: {lua_script}")
    print(f"  Comm dir: {comm_dir}")
    
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
    
    # Test sending a ping command
    print("\nTesting ping command...")
    with open(request_file, 'w') as f:
        f.write("ACTION:PING\n")
    
    # Wait for response
    max_wait = 10
    for i in range(max_wait):
        if response_file.exists():
            with open(response_file, 'r') as f:
                response = f.read().strip()
            print(f"✓ Response received: {response}")
            break
        time.sleep(1)
        print(f"Waiting for response... ({i+1}/{max_wait})")
    else:
        print("✗ No response received")
        process.terminate()
        return False
    
    # Test sending input command
    print("\nTesting input command...")
    with open(request_file, 'w') as f:
        f.write("ACTION:SET_INPUTS|INPUTS:RIGHT:true\n")
    
    # Wait for response
    max_wait = 10
    for i in range(max_wait):
        if response_file.exists():
            with open(response_file, 'r') as f:
                response = f.read().strip()
            print(f"✓ Input response: {response}")
            break
        time.sleep(1)
        print(f"Waiting for input response... ({i+1}/{max_wait})")
    else:
        print("✗ No input response received")
        process.terminate()
        return False
    
    # Clean up
    print("\nCleaning up...")
    process.terminate()
    process.wait()
    
    # Clean up files
    for file in [request_file, response_file, status_file]:
        if file.exists():
            file.unlink()
    
    print("✓ Communication test completed successfully!")
    return True

if __name__ == "__main__":
    test_communication()
