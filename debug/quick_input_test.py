#!/usr/bin/env python3
"""
Quick Input Test - Fast diagnostic for common input issues
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def check_basic_requirements():
    """Check if basic requirements are met."""
    print("🔍 Checking basic requirements...")
    
    # Check BizHawk
    bizhawk_dir = r"C:\Program Files (x86)\BizHawk-2.10-win-x64"
    if not os.path.exists(bizhawk_dir):
        print(f"❌ BizHawk not found at: {bizhawk_dir}")
        return False
    print(f"✅ BizHawk found: {bizhawk_dir}")
    
    # Check ROM
    rom_path = os.path.join(os.path.dirname(__file__), "roms", "Sonic The Hedgehog (USA, Europe).md")
    if not os.path.exists(rom_path):
        print(f"❌ ROM not found at: {rom_path}")
        return False
    print(f"✅ ROM found: {rom_path}")
    
    # Check Lua script
    lua_script = os.path.join(os.path.dirname(__file__), "emulator", "bizhawk_bridge.lua")
    if not os.path.exists(lua_script):
        print(f"❌ Lua script not found at: {lua_script}")
        return False
    print(f"✅ Lua script found: {lua_script}")
    
    return True

def test_window_detection():
    """Test if BizHawk windows can be detected."""
    print("\n🔍 Testing window detection...")
    
    try:
        import win32gui
        
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if any(pattern in window_text.lower() for pattern in ['bizhawk', 'emuhawk', 'sonic', 'genesis']):
                    windows.append((hwnd, window_text))
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        if windows:
            print(f"✅ Found {len(windows)} BizHawk windows:")
            for hwnd, title in windows:
                print(f"   - {title}")
            return True
        else:
            print("❌ No BizHawk windows detected")
            return False
            
    except Exception as e:
        print(f"❌ Error detecting windows: {e}")
        return False

def test_input_manager():
    """Test input manager creation."""
    print("\n🔍 Testing input manager...")
    
    try:
        from utils.input_isolator import get_input_manager, shutdown_input_manager
        
        # Create input manager
        input_manager = get_input_manager(1)
        print("✅ Input manager created")
        
        # Check status
        status = input_manager.get_instance_status()
        print(f"   Status: {status}")
        
        # Cleanup
        shutdown_input_manager()
        print("✅ Input manager cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with input manager: {e}")
        return False

def test_manual_input():
    """Test manual input sending."""
    print("\n🔍 Testing manual input...")
    
    try:
        import win32gui
        import win32con
        import win32api
        
        # Find BizHawk window
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if any(pattern in window_text.lower() for pattern in ['bizhawk', 'emuhawk', 'sonic', 'genesis']):
                    windows.append(hwnd)
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        if not windows:
            print("❌ No BizHawk windows found for manual input test")
            return False
        
        hwnd = windows[0]
        print(f"✅ Testing manual input on window {hwnd}")
        
        # Focus window
        win32gui.SetForegroundWindow(hwnd)
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        time.sleep(0.5)
        
        # Send a test key
        print("   Sending RIGHT arrow key...")
        win32gui.PostMessage(hwnd, 0x0100, 0x27, 0)  # WM_KEYDOWN, VK_RIGHT
        time.sleep(0.1)
        win32gui.PostMessage(hwnd, 0x0101, 0x27, 0)  # WM_KEYUP, VK_RIGHT
        
        print("✅ Manual input test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error with manual input: {e}")
        return False

def main():
    """Main test function."""
    print("⚡ Quick Input Test")
    print("="*50)
    
    # Check requirements
    if not check_basic_requirements():
        print("\n❌ Basic requirements not met. Please fix these issues first.")
        return
    
    # Test window detection
    if not test_window_detection():
        print("\n⚠️  Window detection failed. This might be the issue.")
        print("   Try launching BizHawk manually first.")
    
    # Test input manager
    if not test_input_manager():
        print("\n⚠️  Input manager failed. This is likely the issue.")
    
    # Test manual input
    if not test_manual_input():
        print("\n⚠️  Manual input failed. This suggests a window focus issue.")
    
    print("\n" + "="*50)
    print("QUICK TEST COMPLETE")
    print("="*50)
    print("If all tests passed but you still don't see inputs:")
    print("1. Make sure BizHawk windows are visible and not minimized")
    print("2. Try running this script as administrator")
    print("3. Check if BizHawk is configured to accept external inputs")
    print("4. Run the full diagnostic: python debug_input_system.py")

if __name__ == "__main__":
    main() 