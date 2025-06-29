#!/usr/bin/env python3
"""
Test script to verify the file-based BizHawk bridge communication.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

BIZHAWK_DIR = r"C:\Program Files (x86)\BizHawk-2.10-win-x64"
LUA_SCRIPT = os.path.join(os.path.dirname(__file__), "emulator", "bizhawk_bridge.lua")
ROM_PATH = os.path.join(os.path.dirname(__file__), "roms", "Sonic The Hedgehog (USA, Europe).md")

COMM_DIR = r"D:\AI tests\bizhawk_comm"
REQUEST_FILE = os.path.join(COMM_DIR, "request.txt")
RESPONSE_FILE = os.path.join(COMM_DIR, "response.txt")
STATUS_FILE = os.path.join(COMM_DIR, "status.txt")

def check_files():
    """Check the status of communication files."""
    if os.path.exists(COMM_DIR):
        files = os.listdir(COMM_DIR)
        print(f"Files in comm directory: {files}")
        
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                status = f.read().strip()
            print(f"Status: '{status}'")
        
        if os.path.exists(REQUEST_FILE):
            with open(REQUEST_FILE, 'r') as f:
                content = f.read().strip()
            print(f"Request: '{content}'")
        
        if os.path.exists(RESPONSE_FILE):
            with open(RESPONSE_FILE, 'r') as f:
                content = f.read().strip()
            print(f"Response: '{content}'")
            return True
    return False

def test_manual_communication():
    """Test manual communication with the bridge."""
    print("\n🧪 Testing manual communication...")
    
    # Test ping
    print("Testing PING command...")
    with open(REQUEST_FILE, 'w') as f:
        f.write("ACTION:PING")
    
    # Wait for response
    timeout = 5.0
    start_time = time.time()
    while not os.path.exists(RESPONSE_FILE):
        if time.time() - start_time > timeout:
            print("❌ Timeout waiting for PING response")
            return False
        time.sleep(0.1)
    
    # Read response
    with open(RESPONSE_FILE, 'r') as f:
        response = f.read().strip()
    print(f"PING response: {response}")
    
    # Clean up
    os.remove(RESPONSE_FILE)
    
    # Test get state
    print("Testing GET_STATE command...")
    with open(REQUEST_FILE, 'w') as f:
        f.write("ACTION:GET_STATE")
    
    # Wait for response
    start_time = time.time()
    while not os.path.exists(RESPONSE_FILE):
        if time.time() - start_time > timeout:
            print("❌ Timeout waiting for GET_STATE response")
            return False
        time.sleep(0.1)
    
    # Read response
    with open(RESPONSE_FILE, 'r') as f:
        response = f.read().strip()
    print(f"GET_STATE response: {response}")
    
    # Clean up
    os.remove(RESPONSE_FILE)
    
    return True

def main():
    print("🧪 Testing File-Based Bridge")
    print("=" * 40)
    
    # Check if required files exist
    print("📁 Checking required files...")
    if not os.path.exists(BIZHAWK_DIR):
        print(f"❌ BizHawk directory not found: {BIZHAWK_DIR}")
        return
    
    if not os.path.exists(LUA_SCRIPT):
        print(f"❌ Lua script not found: {LUA_SCRIPT}")
        return
    
    if not os.path.exists(ROM_PATH):
        print(f"❌ ROM file not found: {ROM_PATH}")
        return
    
    print("✅ All required files found")
    
    # Clean up old communication directory
    if os.path.exists(COMM_DIR):
        import shutil
        shutil.rmtree(COMM_DIR)
        print("✅ Cleaned up old communication directory")
    
    # Launch BizHawk with file-based bridge
    cmd = [
        os.path.join(BIZHAWK_DIR, "EmuHawk.exe"),
        f"--lua={LUA_SCRIPT}",
        str(ROM_PATH)
    ]
    
    print(f"🚀 Launching BizHawk: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd)
        print("✅ BizHawk launched")
        
        # Wait for bridge to start (wait for status file)
        print("⏳ Waiting for bridge to start (status file)...")
        for i in range(30):
            if os.path.exists(STATUS_FILE):
                print(f"✅ Status file found after {i+1} seconds")
                break
            time.sleep(1)
        else:
            print("❌ Status file not found after 30 seconds")
            process.terminate()
            process.wait()
            return
        
        # Check initial state
        print("\n📋 Initial state:")
        check_files()
        
        # Test manual communication
        if test_manual_communication():
            print("✅ Manual communication test successful!")
        else:
            print("❌ Manual communication test failed!")
        
        # Final state
        print("\n📋 Final state:")
        check_files()
        
        # Close BizHawk
        process.terminate()
        process.wait()
        print("✅ BizHawk closed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'process' in locals():
            process.terminate()
            process.wait()

if __name__ == "__main__":
    main() 