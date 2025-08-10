#!/usr/bin/env python3
"""
Test script for the simplified file-based environment.
"""

import os
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from src.environment.simple_file_based_env import SimpleFileBasedSonicEnvironment

def test_simple_environment():
    """Test the simplified file-based environment."""
    print("Testing simplified file-based environment...")
    
    # Load configuration
    config_path = "configs/training_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    print("Creating environment...")
    env = SimpleFileBasedSonicEnvironment(config, instance_id=0)
    
    print("Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Available actions: {env.get_action_meanings()}")
    
    # Test reset
    print("\nTesting reset...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    
    # Test a few steps
    print("\nTesting steps...")
    for step in range(10):
        # Choose a random action
        action = env.action_space.sample()
        action_name = env.get_action_meanings()[action]
        
        print(f"Step {step + 1}: Action {action} ({action_name})")
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"  Reward: {reward:.2f}")
        print(f"  Done: {done}")
        print(f"  State: {info.get('state', {})}")
        
        if done:
            print("Episode finished!")
            break
    
    # Clean up
    print("\nCleaning up...")
    env.close()
    
    print("✓ Simple environment test completed!")

if __name__ == "__main__":
    test_simple_environment()
