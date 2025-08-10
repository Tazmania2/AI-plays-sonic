#!/usr/bin/env python3
"""
Debug Input System - Step by Step Diagnostic

This script will test the input isolation system step by step to identify
why inputs aren't being sent to the emulators.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.input_isolator import get_input_manager, shutdown_input_manager
from emulator.sonic_emulator import SonicEmulator

BIZHAWK_DIR = r"C:\Program Files (x86)\BizHawk-2.10-win-x64"
LUA_SCRIPT = os.path.join(os.path.dirname(__file__), "emulator", "bizhawk_bridge.lua")
ROM_PATH = os.path.join(os.path.dirname(__file__), "roms", "Sonic The Hedgehog (USA, Europe).md")

def test_step_1_basic_input_manager():
    """Test 1: Basic input manager creation and initialization."""
    print("\n" + "="*60)
    print("STEP 1: Testing Basic Input Manager")
    print("="*60)
    
    try:
        # Create input manager
        print("Creating input manager...")
        input_manager = get_input_manager(2)  # Just 2 instances for testing
        print("✅ Input manager created successfully")
        
        # Check instance status
        print("Checking instance status...")
        status = input_manager.get_instance_status()
        print(f"Instance status: {status}")
        
        # Test basic functionality
        print("Testing basic send_action...")
        input_manager.send_action(0, 'RIGHT', duration=0.1)
        print("✅ send_action called successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in step 1: {e}")
        return False

def test_step_2_bizhawk_window_detection():
    """Test 2: BizHawk window detection."""
    print("\n" + "="*60)
    print("STEP 2: Testing BizHawk Window Detection")
    print("="*60)
    
    try:
        # Check if BizHawk directory exists
        if not os.path.exists(BIZHAWK_DIR):
            print(f"❌ BizHawk directory not found: {BIZHAWK_DIR}")
            return False
        
        print(f"✅ BizHawk directory found: {BIZHAWK_DIR}")
        
        # Check if ROM exists
        if not os.path.exists(ROM_PATH):
            print(f"❌ ROM file not found: {ROM_PATH}")
            return False
        
        print(f"✅ ROM file found: {ROM_PATH}")
        
        # Check if Lua script exists
        if not os.path.exists(LUA_SCRIPT):
            print(f"❌ Lua script not found: {LUA_SCRIPT}")
            return False
        
        print(f"✅ Lua script found: {LUA_SCRIPT}")
        
        # Launch a single BizHawk instance
        print("Launching BizHawk instance...")
        cmd = [
            os.path.join(BIZHAWK_DIR, "EmuHawk.exe"),
            f"--lua={LUA_SCRIPT}",
            str(ROM_PATH)
        ]
        
        print(f"Command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        
        # Wait for BizHawk to start
        print("Waiting for BizHawk to start...")
        time.sleep(10)
        
        # Test window detection
        print("Testing window detection...")
        input_manager = get_input_manager(1)
        status = input_manager.get_instance_status()
        print(f"Window detection status: {status}")
        
        if status.get(0, False):
            print("✅ BizHawk window detected successfully")
            return True, process
        else:
            print("❌ BizHawk window not detected")
            return False, process
        
    except Exception as e:
        print(f"❌ Error in step 2: {e}")
        return False, None

def test_step_3_emulator_integration():
    """Test 3: Emulator integration with input system."""
    print("\n" + "="*60)
    print("STEP 3: Testing Emulator Integration")
    print("="*60)
    
    try:
        # Create emulator instance
        print("Creating emulator instance...")
        emulator = SonicEmulator(
            rom_path=ROM_PATH,
            bizhawk_dir=BIZHAWK_DIR,
            lua_script_path=LUA_SCRIPT,
            instance_id=0
        )
        print("✅ Emulator instance created")
        
        # Set environment ID
        print("Setting environment ID...")
        emulator.set_env_id(0)
        print("✅ Environment ID set")
        
        # Wait for emulator to start
        print("Waiting for emulator to start...")
        time.sleep(5)
        
        # Test sending actions
        print("Testing action sending...")
        test_actions = ['RIGHT', 'A', 'LEFT', 'B']
        
        for action in test_actions:
            print(f"  Sending action: {action}")
            emulator.step([action])
            time.sleep(0.5)  # Wait to see the effect
        
        print("✅ Action sending test completed")
        return True, emulator
        
    except Exception as e:
        print(f"❌ Error in step 3: {e}")
        return False, None

def test_step_4_input_verification():
    """Test 4: Verify inputs are actually being sent."""
    print("\n" + "="*60)
    print("STEP 4: Input Verification")
    print("="*60)
    
    try:
        # Get input manager
        input_manager = get_input_manager(1)
        
        # Test direct input sending
        print("Testing direct input sending...")
        
        # Send a series of inputs and check if they're processed
        test_sequence = [
            ('RIGHT', 0.2),
            ('A', 0.1),
            ('LEFT', 0.2),
            ('B', 0.1),
            ('START', 0.1)
        ]
        
        for action, duration in test_sequence:
            print(f"  Sending {action} for {duration}s...")
            input_manager.send_action(0, action, duration)
            time.sleep(duration + 0.1)  # Wait for action to complete
        
        print("✅ Direct input test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error in step 4: {e}")
        return False

def test_step_5_window_focus():
    """Test 5: Window focus and input targeting."""
    print("\n" + "="*60)
    print("STEP 5: Window Focus Testing")
    print("="*60)
    
    try:
        import win32gui
        import win32con
        
        # Find BizHawk windows
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if any(pattern in window_text.lower() for pattern in ['bizhawk', 'emuhawk', 'sonic', 'genesis']):
                    windows.append((hwnd, window_text))
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        print(f"Found {len(windows)} BizHawk windows:")
        for hwnd, title in windows:
            print(f"  - {title} (HWND: {hwnd})")
        
        if windows:
            # Test focusing the first window
            hwnd, title = windows[0]
            print(f"Testing focus on: {title}")
            
            # Bring window to foreground
            win32gui.SetForegroundWindow(hwnd)
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            
            print("✅ Window focus test completed")
            return True
        else:
            print("❌ No BizHawk windows found")
            return False
        
    except Exception as e:
        print(f"❌ Error in step 5: {e}")
        return False

def main():
    """Main diagnostic function."""
    print("🔧 Sonic Input System Diagnostic")
    print("="*60)
    print("This script will test the input system step by step.")
    print("Make sure you have BizHawk installed and a Sonic ROM available.")
    print()
    
    # Track results
    results = []
    processes_to_cleanup = []
    
    # Step 1: Basic input manager
    print("Starting Step 1...")
    result1 = test_step_1_basic_input_manager()
    results.append(("Basic Input Manager", result1))
    
    # Step 2: BizHawk window detection
    print("Starting Step 2...")
    result2, process = test_step_2_bizhawk_window_detection()
    results.append(("BizHawk Window Detection", result2))
    if process:
        processes_to_cleanup.append(process)
    
    # Step 3: Emulator integration
    print("Starting Step 3...")
    result3, emulator = test_step_3_emulator_integration()
    results.append(("Emulator Integration", result3))
    
    # Step 4: Input verification
    print("Starting Step 4...")
    result4 = test_step_4_input_verification()
    results.append(("Input Verification", result4))
    
    # Step 5: Window focus
    print("Starting Step 5...")
    result5 = test_step_5_window_focus()
    results.append(("Window Focus", result5))
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! Input system should be working.")
        print("If you're still not seeing inputs, the issue might be:")
        print("1. BizHawk window not receiving focus")
        print("2. Input timing issues")
        print("3. BizHawk configuration problems")
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        print("Common solutions:")
        print("1. Make sure BizHawk is installed correctly")
        print("2. Check that the ROM file exists and is valid")
        print("3. Ensure BizHawk windows are visible and not minimized")
        print("4. Try running as administrator")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    shutdown_input_manager()
    
    for process in processes_to_cleanup:
        try:
            process.terminate()
            process.wait(timeout=5)
            print("✅ Process terminated")
        except:
            print("⚠️  Could not terminate process")
    
    if emulator:
        try:
            emulator.close()
            print("✅ Emulator closed")
        except:
            print("⚠️  Could not close emulator")
    
    print("\nDiagnostic complete!")

if __name__ == "__main__":
    main() 