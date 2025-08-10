#!/usr/bin/env python3
"""
Test script to verify the standard environment fixes.
"""

import os
import sys
import yaml
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from environment.sonic_env import SonicEnvironment

def test_standard_environment(config_path: str):
    """Test the standard environment with fixes."""
    print("🧪 Testing Standard Environment (Fixed)")
    print("=" * 50)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        # Create standard environment
        print("Creating standard environment...")
        env = SonicEnvironment(config)
        
        print("✅ Environment created successfully")
        
        # Test reset
        print("\n🔄 Testing environment reset...")
        obs, info = env.reset()
        print(f"✅ Reset successful. Observation shape: {obs.shape}")
        print(f"   Info: {info}")
        
        # Test a few random actions
        print("\n🎮 Testing random actions...")
        for step in range(5):
            # Take random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            print(f"Step {step + 1}: Action={action}, Reward={reward:.2f}, Done={done}")
            print(f"   Position: {info.get('position', (0, 0))}")
            print(f"   Score: {info.get('score', 0)}, Rings: {info.get('rings', 0)}")
            
            if done or truncated:
                print("Episode ended")
                break
        
        # Close environment
        env.close()
        print("✅ Environment closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    config_path = "configs/training_config.yaml"
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    success = test_standard_environment(config_path)
    
    if success:
        print("\n🎉 Standard environment test successful!")
        print("   The regular command system is now working!")
    else:
        print("\n❌ Standard environment test failed")
        print("   Check the error messages above")

if __name__ == "__main__":
    main()
