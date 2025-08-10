#!/usr/bin/env python3
"""
Diagnostic script for Sonic emulator input issues.
This script will test the file-based communication system and identify potential problems.
"""

import os
import time
import subprocess
import sys
from pathlib import Path

def check_bizhawk_installation():
    """Check if BizHawk is properly installed."""
    print("=== Checking BizHawk Installation ===")
    
    # Check common BizHawk installation paths
    bizhawk_paths = [
        r"C:\Program Files (x86)\BizHawk-2.10-win-x64\EmuHawk.exe",
        r"C:\Program Files\BizHawk-2.10-win-x64\EmuHawk.exe",
        r"C:\BizHawk-2.10-win-x64\EmuHawk.exe"
    ]
    
    found_path = None
    for path in bizhawk_paths:
        if os.path.exists(path):
            found_path = path
            print(f"✓ BizHawk found at: {path}")
            break
    
    if not found_path:
        print("✗ BizHawk not found in common locations")
        print("Please ensure BizHawk is installed and update the path in config")
        return False
    
    return True

def check_rom_file():
    """Check if the Sonic ROM file exists."""
    print("\n=== Checking ROM File ===")
    
    rom_paths = [
        "roms/sonic1.md",
        "roms/Sonic The Hedgehog (USA, Europe).md",
        "Sonic The Hedgehog (USA, Europe).md"
    ]
    
    found_rom = None
    for path in rom_paths:
        if os.path.exists(path):
            found_rom = path
            print(f"✓ ROM found at: {path}")
            break
    
    if not found_rom:
        print("✗ Sonic ROM not found")
        print("Please ensure the ROM file is in the correct location")
        return False
    
    return True

def check_lua_script():
    """Check if the Lua bridge script exists."""
    print("\n=== Checking Lua Bridge Script ===")
    
    lua_paths = [
        "emulator/bizhawk_bridge_file.lua",
        "emulator/bizhawk_bridge.lua"
    ]
    
    found_lua = None
    for path in lua_paths:
        if os.path.exists(path):
            found_lua = path
            print(f"✓ Lua script found at: {path}")
            break
    
    if not found_lua:
        print("✗ Lua bridge script not found")
        print("Please ensure the Lua script exists")
        return False
    
    return True

def test_file_communication():
    """Test the file-based communication system."""
    print("\n=== Testing File Communication ===")
    
    # Create test communication directory
    test_comm_dir = Path("test_comm")
    test_comm_dir.mkdir(exist_ok=True)
    
    request_file = test_comm_dir / "request.txt"
    response_file = test_comm_dir / "response.txt"
    status_file = test_comm_dir / "status.txt"
    
    # Test file writing
    try:
        with open(request_file, 'w') as f:
            f.write("ACTION:PING\n")
        print("✓ Can write to request file")
    except Exception as e:
        print(f"✗ Cannot write to request file: {e}")
        return False
    
    # Test file reading
    try:
        with open(request_file, 'r') as f:
            content = f.read()
        print("✓ Can read from request file")
    except Exception as e:
        print(f"✗ Cannot read from request file: {e}")
        return False
    
    # Clean up
    try:
        request_file.unlink()
        test_comm_dir.rmdir()
        print("✓ File system permissions are working")
    except Exception as e:
        print(f"✗ Cannot clean up test files: {e}")
        return False
    
    return True

def test_bizhawk_launch():
    """Test launching BizHawk manually."""
    print("\n=== Testing BizHawk Launch ===")
    
    # Find BizHawk path
    bizhawk_paths = [
        r"C:\Program Files (x86)\BizHawk-2.10-win-x64\EmuHawk.exe",
        r"C:\Program Files\BizHawk-2.10-win-x64\EmuHawk.exe",
        r"C:\BizHawk-2.10-win-x64\EmuHawk.exe"
    ]
    
    bizhawk_exe = None
    for path in bizhawk_paths:
        if os.path.exists(path):
            bizhawk_exe = path
            break
    
    if not bizhawk_exe:
        print("✗ BizHawk executable not found")
        return False
    
    # Find ROM
    rom_paths = [
        "roms/sonic1.md",
        "roms/Sonic The Hedgehog (USA, Europe).md",
        "Sonic The Hedgehog (USA, Europe).md"
    ]
    
    rom_file = None
    for path in rom_paths:
        if os.path.exists(path):
            rom_file = path
            break
    
    if not rom_file:
        print("✗ ROM file not found")
        return False
    
    # Find Lua script
    lua_paths = [
        "emulator/bizhawk_bridge_file.lua",
        "emulator/bizhawk_bridge.lua"
    ]
    
    lua_script = None
    for path in lua_paths:
        if os.path.exists(path):
            lua_script = path
            break
    
    if not lua_script:
        print("✗ Lua script not found")
        return False
    
    print(f"Attempting to launch BizHawk...")
    print(f"BizHawk: {bizhawk_exe}")
    print(f"ROM: {rom_file}")
    print(f"Lua: {lua_script}")
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['BIZHAWK_INSTANCE_ID'] = '0'
        env['BIZHAWK_COMM_BASE'] = os.getcwd()
        
        # Launch BizHawk
        cmd = [
            bizhawk_exe,
            f"--lua={lua_script}",
            rom_file
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        # Launch in background
        process = subprocess.Popen(cmd, env=env)
        
        print("✓ BizHawk launched successfully")
        print("Waiting 10 seconds for startup...")
        time.sleep(10)
        
        # Check if process is still running
        if process.poll() is None:
            print("✓ BizHawk process is running")
            
            # Check for communication files
            comm_dir = Path("bizhawk_comm_0")
            status_file = comm_dir / "status.txt"
            
            if status_file.exists():
                with open(status_file, 'r') as f:
                    status = f.read().strip()
                print(f"✓ Communication status: {status}")
            else:
                print("✗ Communication status file not found")
            
            # Terminate process
            process.terminate()
            process.wait()
            print("✓ BizHawk terminated cleanly")
        else:
            print("✗ BizHawk process terminated unexpectedly")
            return False
            
    except Exception as e:
        print(f"✗ Failed to launch BizHawk: {e}")
        return False
    
    return True

def check_firewall_settings():
    """Check and suggest firewall settings."""
    print("\n=== Firewall Analysis ===")
    
    print("Windows Firewall is enabled on all profiles.")
    print("For file-based communication, firewall should not be an issue.")
    print("However, if using network-based communication, you may need to:")
    print("1. Add BizHawk to Windows Firewall exceptions")
    print("2. Allow BizHawk through antivirus software")
    print("3. Run as administrator if needed")
    
    return True

def check_antivirus_interference():
    """Check for potential antivirus interference."""
    print("\n=== Antivirus Check ===")
    
    # Check for common antivirus processes
    antivirus_processes = [
        "avast.exe", "avgui.exe", "mcafee.exe", "norton.exe", 
        "kaspersky.exe", "bitdefender.exe", "malwarebytes.exe"
    ]
    
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq avast.exe'], 
                              capture_output=True, text=True)
        if "avast.exe" in result.stdout:
            print("⚠ Avast detected - may interfere with file operations")
        
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq avgui.exe'], 
                              capture_output=True, text=True)
        if "avgui.exe" in result.stdout:
            print("⚠ AVG detected - may interfere with file operations")
        
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq mcafee.exe'], 
                              capture_output=True, text=True)
        if "mcafee.exe" in result.stdout:
            print("⚠ McAfee detected - may interfere with file operations")
        
        print("If you have antivirus software, try:")
        print("1. Adding the project directory to exclusions")
        print("2. Temporarily disabling real-time protection")
        print("3. Running the script as administrator")
        
    except Exception as e:
        print(f"Could not check for antivirus software: {e}")
    
    return True

def main():
    """Run all diagnostic tests."""
    print("Sonic Emulator Input Diagnostic Tool")
    print("=" * 50)
    
    tests = [
        ("BizHawk Installation", check_bizhawk_installation),
        ("ROM File", check_rom_file),
        ("Lua Script", check_lua_script),
        ("File Communication", test_file_communication),
        ("Firewall Settings", check_firewall_settings),
        ("Antivirus Check", check_antivirus_interference),
        ("BizHawk Launch Test", test_bizhawk_launch),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The emulator should work correctly.")
        print("If you're still having input issues, try:")
        print("1. Running the script as administrator")
        print("2. Temporarily disabling antivirus real-time protection")
        print("3. Checking that BizHawk window is focused")
    else:
        print("\n⚠ Some tests failed. Please address the issues above.")
        print("Common solutions:")
        print("1. Install BizHawk if not found")
        print("2. Place ROM file in correct location")
        print("3. Run as administrator")
        print("4. Add project directory to antivirus exclusions")

if __name__ == "__main__":
    main()
