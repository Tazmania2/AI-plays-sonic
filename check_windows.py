#!/usr/bin/env python3
"""
Check Windows - See what windows are detected by the input system
"""

import win32gui
import win32process

def enum_windows_callback(hwnd, windows):
    if win32gui.IsWindowVisible(hwnd):
        window_text = win32gui.GetWindowText(hwnd)
        window_class = win32gui.GetClassName(hwnd)
        try:
            _, process_id = win32process.GetWindowThreadProcessId(hwnd)
        except:
            process_id = "Unknown"
        
        windows.append((hwnd, window_text, window_class, process_id))
    return True

def main():
    print("🔍 Detecting all visible windows...")
    print("="*60)
    
    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)
    
    # Look for BizHawk-related windows
    bizhawk_windows = []
    for hwnd, title, class_name, pid in windows:
        if any(pattern in title.lower() for pattern in ['bizhawk', 'emuhawk', 'sonic', 'genesis', 'megadrive']):
            bizhawk_windows.append((hwnd, title, class_name, pid))
    
    print(f"Found {len(bizhawk_windows)} BizHawk-related windows:")
    print("-" * 60)
    
    for i, (hwnd, title, class_name, pid) in enumerate(bizhawk_windows):
        print(f"{i}: {title}")
        print(f"   HWND: {hwnd}")
        print(f"   Class: {class_name}")
        print(f"   PID: {pid}")
        print()
    
    if not bizhawk_windows:
        print("❌ No BizHawk windows found!")
        print("\nAll visible windows:")
        print("-" * 60)
        for i, (hwnd, title, class_name, pid) in enumerate(windows[:20]):  # Show first 20
            print(f"{i}: {title}")
        if len(windows) > 20:
            print(f"... and {len(windows) - 20} more windows")
    
    print("="*60)
    print("If no BizHawk windows are found:")
    print("1. Make sure BizHawk is running")
    print("2. Check if BizHawk window titles contain expected keywords")
    print("3. Try launching BizHawk manually first")

if __name__ == "__main__":
    main() 