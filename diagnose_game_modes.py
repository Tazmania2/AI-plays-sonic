#!/usr/bin/env python3
"""
Diagnostic script to identify game mode values for Sonic 1.
This helps determine the correct values for menu, demo, and gameplay modes.
"""

import sys
import os
import time
import yaml

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from emulator.sonic_emulator import SonicEmulator

def diagnose_game_modes():
    """Diagnose game modes by monitoring the game state."""
    print("=== Sonic 1 Game Mode Diagnosis ===")
    
    # Load configuration
    config_path = "configs/training_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded successfully")
    
    try:
        # Create emulator
        print("\nCreating SonicEmulator...")
        emulator = SonicEmulator(
            rom_path=config['game']['rom_path'],
            bizhawk_dir=config.get('bizhawk_dir', r"C:\Program Files (x86)\BizHawk-2.10-win-x64"),
            lua_script_path=config.get('lua_script_path', 'emulator/bizhawk_bridge_fixed.lua'),
            instance_id=0
        )
        
        # Launch emulator
        print("Launching BizHawk...")
        emulator.launch()
        
        print("\n=== Game Mode Monitoring ===")
        print("Please observe the game and note the following phases:")
        print("1. ROM loading (should show 'None' or errors)")
        print("2. Title screen/menu (note the game_mode value)")
        print("3. Demo mode (note the game_mode value)")
        print("4. Gameplay (note the game_mode value)")
        print("\nPress Ctrl+C to stop monitoring...")
        
        # Monitor game state
        last_game_mode = None
        mode_changes = []
        
        try:
            while True:
                try:
                    state = emulator.get_game_state()
                    if state:
                        game_mode = state.get('game_mode')
                        if game_mode != last_game_mode:
                            timestamp = time.strftime("%H:%M:%S")
                            print(f"[{timestamp}] Game mode changed: {last_game_mode} -> {game_mode}")
                            mode_changes.append({
                                'timestamp': timestamp,
                                'old_mode': last_game_mode,
                                'new_mode': game_mode,
                                'state': state
                            })
                            last_game_mode = game_mode
                        
                        # Print current state every 10 seconds
                        if int(time.time()) % 10 == 0:
                            print(f"[{timestamp}] Current state: mode={game_mode}, score={state.get('score', 'N/A')}, lives={state.get('lives', 'N/A')}")
                    
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error reading state: {e}")
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            print("\n\n=== Monitoring Stopped ===")
            print("Game mode changes detected:")
            for change in mode_changes:
                print(f"  {change['timestamp']}: {change['old_mode']} -> {change['new_mode']}")
            
            print("\n=== Summary ===")
            unique_modes = set(change['new_mode'] for change in mode_changes if change['new_mode'] is not None)
            print(f"Unique game modes observed: {sorted(unique_modes)}")
            
            # Try to categorize modes
            print("\n=== Suggested Mode Categories ===")
            for mode in sorted(unique_modes):
                print(f"Game mode {mode}:")
                # Find a state with this mode
                for change in mode_changes:
                    if change['new_mode'] == mode:
                        state = change['state']
                        print(f"  - Score: {state.get('score', 'N/A')}")
                        print(f"  - Lives: {state.get('lives', 'N/A')}")
                        print(f"  - Zone: {state.get('zone', 'N/A')}")
                        print(f"  - Act: {state.get('act', 'N/A')}")
                        break
        
        # Clean up
        emulator.close()
        return True
        
    except Exception as e:
        print(f"Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = diagnose_game_modes()
    if success:
        print("\n✅ Game mode diagnosis completed!")
    else:
        print("\n❌ Game mode diagnosis failed!")
        sys.exit(1)
