#!/usr/bin/env python3
"""
Launch RetroArch in GUI mode to enable network command interface.
"""

import subprocess
import os
import time

def launch_retroarch_gui():
    """Launch RetroArch in GUI mode."""
    print("🎮 Launching RetroArch in GUI mode...")
    print("📋 Instructions:")
    print("1. Wait for RetroArch to open")
    print("2. Go to Settings > Network")
    print("3. Enable 'Network Command Interface'")
    print("4. Set 'Network Command Port' to 55355")
    print("5. Save the configuration")
    print("6. Close RetroArch")
    print("7. Then run your training script again")
    print()
    
    retroarch_path = "C:\\RetroArch-Win64\\retroarch.exe"
    config_path = "D:\\AI tests\\retroarch.cfg"
    
    if not os.path.exists(retroarch_path):
        print(f"❌ RetroArch not found at: {retroarch_path}")
        return False
    
    try:
        # Launch RetroArch with config but without loading a game
        cmd = [retroarch_path, "--config", config_path]
        print(f"Running: {' '.join(cmd)}")
        
        process = subprocess.Popen(cmd)
        print("✅ RetroArch launched! Please follow the instructions above.")
        print("Press Ctrl+C to stop this script when you're done.")
        
        # Wait for user to finish
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping RetroArch...")
            process.terminate()
            process.wait()
        
        return True
        
    except Exception as e:
        print(f"❌ Error launching RetroArch: {e}")
        return False

if __name__ == "__main__":
    launch_retroarch_gui() 