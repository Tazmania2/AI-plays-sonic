#!/usr/bin/env python3
"""
Test script for the updated startup sequence in DirectInputSonicEnvironment.
This tests the 15-second wait and START command functionality.
"""

import sys
import os
import time
import yaml

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environment.direct_input_env import DirectInputSonicEnvironment

def test_startup_sequence():
    """Test the startup sequence with proper timing."""
    print("=== Testing DirectInputSonicEnvironment Startup Sequence ===")
    
    # Load configuration
    config_path = "configs/training_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded successfully")
    
    try:
        # Create environment (this will trigger the startup sequence)
        print("\nCreating DirectInputSonicEnvironment...")
        print("This will launch BizHawk, wait 15 seconds, and send START command...")
        
        env = DirectInputSonicEnvironment(config, env_id=0)
        print("Environment created successfully!")
        
        # Test a few basic actions
        print("\nTesting basic actions...")
        
        # Test NOOP (action 0)
        print("Sending NOOP action...")
        obs, reward, done, truncated, info = env.step(0)
        print(f"NOOP result: reward={reward}, done={done}, input_method={info.get('input_method')}")
        
        # Test RIGHT (action 1)
        print("Sending RIGHT action...")
        obs, reward, done, truncated, info = env.step(1)
        print(f"RIGHT result: reward={reward}, done={done}, input_method={info.get('input_method')}")
        
        # Test A (jump) - assuming it's action 4
        print("Sending A (jump) action...")
        obs, reward, done, truncated, info = env.step(4)
        print(f"A result: reward={reward}, done={done}, input_method={info.get('input_method')}")
        
        # Test RIGHT+A combination - assuming it's action 8
        print("Sending RIGHT+A combination...")
        obs, reward, done, truncated, info = env.step(8)
        print(f"RIGHT+A result: reward={reward}, done={done}, input_method={info.get('input_method')}")
        
        # Get action meanings
        action_meanings = env.get_action_meanings()
        print(f"\nAvailable actions: {action_meanings}")
        
        # Test reset
        print("\nTesting reset...")
        obs, info = env.reset()
        print(f"Reset completed: input_method={info.get('input_method')}")
        
        print("\n=== Startup Sequence Test Completed Successfully! ===")
        print("If Sonic is now moving in the emulator, the startup sequence is working correctly.")
        
        # Clean up
        env.close()
        return True
        
    except Exception as e:
        print(f"Error during startup sequence test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_startup_sequence()
    if success:
        print("\n✅ Startup sequence test passed!")
    else:
        print("\n❌ Startup sequence test failed!")
        sys.exit(1)
