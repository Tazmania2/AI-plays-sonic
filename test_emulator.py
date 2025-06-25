#!/usr/bin/env python3
"""
Test script to verify emulator setup and ROM file.
"""

import os
import sys
from pathlib import Path
import subprocess
import time

def check_rom_file(rom_path):
    """Check if the ROM file exists and is valid."""
    print(f"Checking ROM file: {rom_path}")
    
    if not os.path.exists(rom_path):
        print(f"❌ ROM file not found: {rom_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(rom_path)
    print(f"📁 ROM file size: {file_size:,} bytes")
    
    # Check file extension
    file_ext = Path(rom_path).suffix.lower()
    valid_extensions = ['.md', '.gen', '.bin', '.smd', '.zip', '.7z']
    
    if file_ext not in valid_extensions:
        print(f"⚠️  Warning: File extension '{file_ext}' is not a typical ROM extension")
        print(f"   Valid extensions: {valid_extensions}")
    
    # Try to read first few bytes to check if it's a valid file
    try:
        with open(rom_path, 'rb') as f:
            header = f.read(16)
            print(f"🔍 File header: {header.hex()}")
            
            # Check for common ROM signatures
            if header.startswith(b'SEGA'):
                print("✅ Valid Sega Genesis ROM detected!")
                return True
            elif header.startswith(b'SEGA MEGA DRIVE'):
                print("✅ Valid Sega Mega Drive ROM detected!")
                return True
            else:
                print("⚠️  File doesn't appear to be a standard Sega ROM")
                print("   But will try to load it anyway (might be compressed/modified)")
                return True  # Return True anyway to test loading
    except Exception as e:
        print(f"❌ Error reading ROM file: {e}")
        return False

def check_emulator_installation_and_core():
    """Check if RetroArch and the core are installed."""
    print("\nChecking emulator installation and core...")
    
    # Check common RetroArch paths
    retroarch_paths = [
        "C:\\RetroArch-Win64\\retroarch.exe",
        "C:\\Program Files\\RetroArch\\retroarch.exe",
        "C:\\Program Files (x86)\\RetroArch\\retroarch.exe",
        os.path.expanduser("~/AppData/Local/Programs/RetroArch/retroarch.exe")
    ]
    core_paths = [
        "C:\\RetroArch-Win64\\cores\\genesis_plus_gx_libretro.dll",
        "C:\\Program Files\\RetroArch\\cores\\genesis_plus_gx_libretro.dll",
        "C:\\Program Files (x86)\\RetroArch\\cores\\genesis_plus_gx_libretro.dll",
        os.path.expanduser("~/AppData/Local/Programs/RetroArch/cores/genesis_plus_gx_libretro.dll")
    ]
    retroarch_path = None
    core_path = None
    for path in retroarch_paths:
        if os.path.exists(path):
            retroarch_path = path
            print(f"✅ RetroArch found at: {path}")
            break
    for path in core_paths:
        if os.path.exists(path):
            core_path = path
            print(f"✅ Core found at: {path}")
            break
    if not retroarch_path:
        print("❌ RetroArch not found in common locations")
        print("   Please install RetroArch from: https://www.retroarch.com/")
    if not core_path:
        print("❌ Core not found in common locations")
        print("   Please download the Genesis Plus GX core in RetroArch.")
    return retroarch_path, core_path

def test_emulator_startup(retroarch_path, core_path, rom_path):
    """Test if RetroArch can start with the core and ROM."""
    print(f"\nTesting emulator startup...")
    try:
        # Get the path to the config file in the project directory
        config_path = os.path.join(os.path.dirname(__file__), "retroarch.cfg")
        if not os.path.exists(config_path):
            print(f"Warning: Config file not found at {config_path}, using default RetroArch config")
            config_path = None
        
        # Build command with config file
        cmd = [retroarch_path]
        if config_path:
            cmd.extend(["--config", config_path])
        cmd.extend(["-L", core_path, rom_path])
        
        print(f"Starting RetroArch with command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        time.sleep(5)
        if process.poll() is None:
            print("✅ RetroArch started successfully!")
            print("   (You should see the emulator window)")
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
            return True
        else:
            print("❌ RetroArch failed to start")
            return False
    except Exception as e:
        print(f"❌ Error starting RetroArch: {e}")
        return False

def main():
    """Main test function."""
    print("🎮 Sonic AI Emulator Test")
    print("=" * 50)
    
    # Check ROM file
    rom_path = "roms/sonic1.md"
    rom_valid = check_rom_file(rom_path)
    
    # Check emulator
    retroarch_path, core_path = check_emulator_installation_and_core()
    
    if retroarch_path and core_path and rom_valid:
        print("\n" + "=" * 50)
        print("🚀 Testing full setup...")
        
        # Test emulator startup
        emulator_works = test_emulator_startup(retroarch_path, core_path, rom_path)
        
        if emulator_works:
            print("\n✅ All tests passed! The emulator setup is working.")
            print("   You can now run the training script.")
        else:
            print("\n❌ Emulator test failed. Please check the installation.")
    else:
        print("\n❌ Setup incomplete. Please fix the issues above.")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main() 