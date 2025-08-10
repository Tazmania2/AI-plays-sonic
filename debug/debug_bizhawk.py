#!/usr/bin/env python3
"""
Debug script for BizHawk file-based communication.
"""

import os
import time
import subprocess

COMM_DIR = "D:/AI tests/bizhawk_comm"
REQUEST_FILE = os.path.join(COMM_DIR, "request.txt")
RESPONSE_FILE = os.path.join(COMM_DIR, "response.txt")
STATUS_FILE = os.path.join(COMM_DIR, "status.txt")

def check_files():
    """Check the status of communication files."""
    print("🔍 Checking communication files...")
    print(f"Communication directory: {COMM_DIR}")
    
    if os.path.exists(COMM_DIR):
        print("✅ Communication directory exists")
        files = os.listdir(COMM_DIR)
        print(f"Files in directory: {files}")
        
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                status = f.read().strip()
            print(f"Status file content: '{status}'")
        else:
            print("❌ Status file not found")
            
        if os.path.exists(REQUEST_FILE):
            print("⚠️  Request file exists (should be deleted after processing)")
            with open(REQUEST_FILE, 'r') as f:
                content = f.read().strip()
            print(f"Request file content: '{content}'")
        else:
            print("✅ Request file not found (good)")
            
        if os.path.exists(RESPONSE_FILE):
            print("⚠️  Response file exists")
            try:
                with open(RESPONSE_FILE, 'rb') as f:
                    content = f.read()
                print(f"Response file content (hex): {content.hex()}")
            except:
                with open(RESPONSE_FILE, 'r') as f:
                    content = f.read()
                print(f"Response file content: '{content}'")
        else:
            print("✅ Response file not found (good)")
    else:
        print("❌ Communication directory not found")

def test_manual_request():
    """Test a manual memory read request."""
    print("\n🧪 Testing manual memory read request...")
    
    # Create request
    request = "read FFE20 2\n"  # Read rings
    print(f"Writing request: '{request.strip()}'")
    
    with open(REQUEST_FILE, 'w') as f:
        f.write(request)
    
    print("Request written. Waiting for response...")
    
    # Wait for response
    timeout = 10.0
    start_time = time.time()
    
    while not os.path.exists(RESPONSE_FILE):
        if time.time() - start_time > timeout:
            print("❌ Timeout waiting for response")
            return False
        time.sleep(0.1)
        print(".", end="", flush=True)
    
    print("\n✅ Response received!")
    
    # Read response
    try:
        with open(RESPONSE_FILE, 'rb') as f:
            data = f.read()
        print(f"Response data (hex): {data.hex()}")
        print(f"Response size: {len(data)} bytes")
        
        # Clean up
        os.remove(RESPONSE_FILE)
        print("✅ Response file cleaned up")
        
        return True
    except Exception as e:
        print(f"❌ Error reading response: {e}")
        return False

def main():
    print("🔧 BizHawk File-Based Communication Debug")
    print("=" * 50)
    
    # Check current state
    check_files()
    
    # Test manual request
    if test_manual_request():
        print("\n✅ Manual test successful!")
    else:
        print("\n❌ Manual test failed!")
    
    # Check final state
    print("\n📋 Final file state:")
    check_files()

if __name__ == "__main__":
    main() 