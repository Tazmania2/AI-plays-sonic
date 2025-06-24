#!/usr/bin/env python3
"""
Test script to verify the training environment setup.
"""

import yaml
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from environment.sonic_env import SonicEnvironment

def test_environment_creation():
    """Test if the training environment can be created."""
    print("🧪 Testing Training Environment Setup")
    print("=" * 50)
    
    try:
        # Load config
        with open('configs/training_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Config loaded successfully")
        
        # Create environment
        print("Creating Sonic environment...")
        env = SonicEnvironment(config)
        print("✅ Environment created successfully")
        
        # Test reset
        print("Testing environment reset...")
        obs, info = env.reset()
        print(f"✅ Environment reset successful")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation dtype: {obs.dtype}")
        
        # Test step
        print("Testing environment step...")
        obs, reward, terminated, truncated, info = env.step(0)  # NOOP action
        print(f"✅ Environment step successful")
        print(f"   Reward: {reward}")
        print(f"   Terminated: {terminated}")
        print(f"   Truncated: {truncated}")
        
        # Test action space
        print(f"✅ Action space: {env.action_space}")
        print(f"✅ Observation space: {env.observation_space}")
        
        # Close environment
        env.close()
        print("✅ Environment closed successfully")
        
        print("\n🎉 All environment tests passed!")
        print("   The AI training environment is ready to use.")
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_environment_creation() 