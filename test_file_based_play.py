#!/usr/bin/env python3
"""
Test script for file-based Sonic AI play system.

This script demonstrates the file-based approach as an alternative to the standard
memory-based approach. It shows how the AI can control Sonic through file-based
communication.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from environment.file_based_env import FileBasedSonicEnvironment
from environment.sonic_env import SonicEnvironment

def test_file_based_environment(config_path: str, instance_id: int = 0):
    """Test the file-based environment."""
    print("🧪 Testing File-Based Environment")
    print("=" * 50)
    
    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        # Create file-based environment
        print(f"Creating file-based environment with instance ID: {instance_id}")
        env = FileBasedSonicEnvironment(config, instance_id=instance_id)
        
        print("✅ Environment created successfully")
        
        # Test reset
        print("\n🔄 Testing environment reset...")
        obs, info = env.reset()
        print(f"✅ Reset successful. Observation shape: {obs.shape}")
        print(f"   Info: {info}")
        
        # Test a few random actions
        print("\n🎮 Testing random actions...")
        for step in range(10):
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

def compare_environments(config_path: str):
    """Compare file-based and standard environments."""
    print("🔍 Comparing File-Based vs Standard Environments")
    print("=" * 60)
    
    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    results = {}
    
    # Test file-based environment
    print("\n📁 Testing File-Based Environment...")
    try:
        file_env = FileBasedSonicEnvironment(config, instance_id=0)
        obs, info = file_env.reset()
        
        print(f"✅ File-based environment works")
        print(f"   Action space: {file_env.action_space}")
        print(f"   Observation space: {file_env.observation_space}")
        print(f"   Action meanings: {file_env.get_action_meanings()}")
        
        # Test one step
        action = file_env.action_space.sample()
        obs, reward, done, truncated, info = file_env.step(action)
        print(f"   Step test: Action={action}, Reward={reward:.2f}")
        
        file_env.close()
        results['file_based'] = True
        
    except Exception as e:
        print(f"❌ File-based environment failed: {e}")
        results['file_based'] = False
    
    # Test standard environment
    print("\n💾 Testing Standard Environment...")
    try:
        std_env = SonicEnvironment(config)
        obs, info = std_env.reset()
        
        print(f"✅ Standard environment works")
        print(f"   Action space: {std_env.action_space}")
        print(f"   Observation space: {std_env.observation_space}")
        print(f"   Action meanings: {std_env.get_action_meanings()}")
        
        # Test one step
        action = std_env.action_space.sample()
        obs, reward, done, truncated, info = std_env.step(action)
        print(f"   Step test: Action={action}, Reward={reward:.2f}")
        
        std_env.close()
        results['standard'] = True
        
    except Exception as e:
        print(f"❌ Standard environment failed: {e}")
        results['standard'] = False
    
    # Summary
    print("\n📊 Environment Comparison Summary")
    print("=" * 40)
    print(f"File-based environment: {'✅ Working' if results.get('file_based') else '❌ Failed'}")
    print(f"Standard environment: {'✅ Working' if results.get('standard') else '❌ Failed'}")
    
    if results.get('file_based') and results.get('standard'):
        print("\n🎉 Both environments are working!")
        print("   You can use either approach for training/playing")
    elif results.get('file_based'):
        print("\n✅ File-based environment is working!")
        print("   Use --file-based flag for training/playing")
    elif results.get('standard'):
        print("\n✅ Standard environment is working!")
        print("   Use standard approach for training/playing")
    else:
        print("\n❌ Both environments failed")
        print("   Check your setup and configuration")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test file-based Sonic environment")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--instance-id", type=int, default=0,
                       help="Instance ID for file-based environment")
    parser.add_argument("--compare", action="store_true",
                       help="Compare file-based and standard environments")
    
    args = parser.parse_args()
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return
    
    if args.compare:
        compare_environments(args.config)
    else:
        success = test_file_based_environment(args.config, args.instance_id)
        
        if success:
            print("\n🎉 File-based environment test successful!")
            print("   You can now use --file-based flag with play_sonic.py")
        else:
            print("\n❌ File-based environment test failed")
            print("   Check the error messages above")

if __name__ == "__main__":
    main()
