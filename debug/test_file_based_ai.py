#!/usr/bin/env python3
"""
Test script for the file-based AI input system.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ai_input_controller import AIInputController

def test_file_based_system():
    """Test the file-based AI input system."""
    print("🧪 Testing File-Based AI Input System")
    print("=" * 50)
    
    # Check if files exist
    bizhawk_dir = r"C:\Program Files (x86)\BizHawk-2.10-win-x64"
    lua_script = "emulator/input_player.lua"
    rom_path = "roms/Sonic The Hedgehog (USA, Europe).md"
    
    if not os.path.exists(bizhawk_dir):
        print(f"❌ BizHawk directory not found: {bizhawk_dir}")
        return False
    
    if not os.path.exists(lua_script):
        print(f"❌ Lua script not found: {lua_script}")
        return False
    
    if not os.path.exists(rom_path):
        print(f"❌ ROM file not found: {rom_path}")
        return False
    
    print("✅ All required files found")
    
    # Create controller
    controller = AIInputController(
        bizhawk_dir=bizhawk_dir,
        lua_script=lua_script,
        rom_path=rom_path,
        working_dir="test_ai_training"
    )
    
    try:
        # Start the emulator first
        print("\n🚀 Starting BizHawk emulator...")
        controller.start_emulator()
        
        # Wait a bit for emulator to fully load
        print("⏳ Waiting for emulator to load...")
        time.sleep(10)
        
        # Test a single episode
        print("\n🎮 Testing single episode...")
        success = controller.run_training_episode(1, 1)
        
        if success:
            print("✅ Episode completed successfully!")
            
            # Read and display results
            game_states = controller.read_game_log()
            if game_states:
                print(f"\n📊 Game log contains {len(game_states)} states")
                print("First few states:")
                for i, state in enumerate(game_states[:5]):
                    print(f"  Frame {state['frame']}: X={state['x']}, Y={state['y']}, Rings={state['rings']}")
            
            return True
        else:
            print("❌ Episode failed")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    finally:
        # Cleanup
        if controller.process:
            controller.process.terminate()
            print("✅ Emulator closed")

def main():
    success = test_file_based_system()
    
    if success:
        print("\n🎉 File-based AI system is working!")
        print("   You can now run full training with: python ai_input_controller.py")
    else:
        print("\n❌ File-based AI system needs debugging")

if __name__ == "__main__":
    main() 