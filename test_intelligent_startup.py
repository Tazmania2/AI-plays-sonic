#!/usr/bin/env python3
"""
Test script for the intelligent startup sequence in DirectInputSonicEnvironment.
This tests the adaptive game state detection and startup handling.
"""

import sys
import os
import time
import yaml

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environment.direct_input_env import DirectInputSonicEnvironment

def test_intelligent_startup():
    """Test the intelligent startup sequence with game state detection."""
    print("=== Testing Intelligent Startup Sequence ===")
    
    # Load configuration
    config_path = "configs/training_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded successfully")
    
    try:
        # Create environment (this will trigger the intelligent startup sequence)
        print("\nCreating DirectInputSonicEnvironment...")
        print("This will launch BizHawk and intelligently handle game startup...")
        
        env = DirectInputSonicEnvironment(config, env_id=0)
        print("Environment created successfully!")
        
        # Test a few basic actions to verify the game is working
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
        
        # Get action meanings
        action_meanings = env.get_action_meanings()
        print(f"\nAvailable actions: {action_meanings}")
        
        # Test reset
        print("\nTesting reset...")
        obs, info = env.reset()
        print(f"Reset completed: input_method={info.get('input_method')}")
        
        print("\n=== Intelligent Startup Test Completed Successfully! ===")
        print("If Sonic is now moving in the emulator, the intelligent startup sequence is working correctly.")
        
        # Clean up
        env.close()
        return True
        
    except Exception as e:
        print(f"Error during intelligent startup test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_intelligent_startup()
    if success:
        print("\n✅ Intelligent startup test passed!")
    else:
        print("\n❌ Intelligent startup test failed!")
        sys.exit(1)
