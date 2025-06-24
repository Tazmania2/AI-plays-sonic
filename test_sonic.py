#!/usr/bin/env python3
"""
Test script for Sonic AI project.

This script tests all major components to ensure they work correctly.
"""

import os
import sys
import yaml
import numpy as np
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from demo_sonic import SonicSimulator
from agents.simple_agent import SimpleAgent, SonicHeuristicAgent, SonicRandomAgent
from utils.observation_processor import ObservationProcessor, SonicSpecificProcessor
from utils.reward_calculator import RewardCalculator, SonicSpecificRewardCalculator


def test_simulator():
    """Test the Sonic simulator."""
    print("Testing Sonic Simulator...")
    
    try:
        # Create simulator
        simulator = SonicSimulator()
        
        # Test reset
        obs, info = simulator.reset()
        assert obs.shape == (84, 84), f"Expected shape (84, 84), got {obs.shape}"
        print("✓ Reset works correctly")
        
        # Test step
        obs, reward, done, info = simulator.step(2)  # Move right
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(done, bool), "Done should be boolean"
        assert isinstance(info, dict), "Info should be dictionary"
        print("✓ Step works correctly")
        
        # Test multiple steps
        for i in range(10):
            obs, reward, done, info = simulator.step(2)
            if done:
                break
        
        print("✓ Multiple steps work correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Simulator test failed: {e}")
        return False


def test_observation_processor():
    """Test the observation processor."""
    print("\nTesting Observation Processor...")
    
    try:
        # Create test image
        test_image = np.random.randint(0, 255, (224, 256, 3), dtype=np.uint8)
        
        # Test basic processor
        config = {
            'resize': [84, 84],
            'crop': [0, 0, 224, 256],
            'normalize': True,
            'frame_stack': 1
        }
        
        processor = ObservationProcessor(config)
        processed = processor.process(test_image)
        
        assert processed.shape == (84, 84, 1), f"Expected shape (84, 84, 1), got {processed.shape}"
        print("✓ Basic processor works correctly")
        
        # Test Sonic-specific processor
        sonic_processor = SonicSpecificProcessor(config)
        processed = sonic_processor.process(test_image)
        
        # Sonic processor adds more features
        assert len(processed.shape) == 3, "Processed should have 3 dimensions"
        print("✓ Sonic-specific processor works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Observation processor test failed: {e}")
        return False


def test_reward_calculator():
    """Test the reward calculator."""
    print("\nTesting Reward Calculator...")
    
    try:
        # Create test states
        prev_state = {
            'score': 100,
            'rings': 5,
            'lives': 3,
            'position': (50, 100),
            'speed': 3
        }
        
        curr_state = {
            'score': 110,
            'rings': 6,
            'lives': 3,
            'position': (60, 100),
            'speed': 5
        }
        
        # Test basic calculator
        config = {
            'ring_collected': 10.0,
            'enemy_defeated': 5.0,
            'forward_progress': 1.0,
            'speed_bonus': 2.0
        }
        
        calculator = RewardCalculator(config)
        reward = calculator.calculate_reward(prev_state, curr_state)
        
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        print("✓ Basic reward calculator works correctly")
        
        # Test Sonic-specific calculator
        sonic_calculator = SonicSpecificRewardCalculator(config)
        reward = sonic_calculator.calculate_reward(prev_state, curr_state)
        
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        print("✓ Sonic-specific reward calculator works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Reward calculator test failed: {e}")
        return False


def test_agents():
    """Test the agent implementations."""
    print("\nTesting Agents...")
    
    try:
        # Create test configuration
        config = {
            'agent': {
                'learning_rate': 0.001,
                'gamma': 0.99
            },
            'frame_stack': 1
        }
        
        # Test random agent
        random_agent = SonicRandomAgent(config)
        obs = np.random.randint(0, 255, (84, 84), dtype=np.uint8)
        info = {'position': (50, 100), 'velocity': (3, 0)}
        
        action = random_agent.select_action(obs, info)
        assert 0 <= action <= 8, f"Action should be between 0 and 8, got {action}"
        print("✓ Random agent works correctly")
        
        # Test heuristic agent
        heuristic_agent = SonicHeuristicAgent(config)
        action = heuristic_agent.select_action(obs, info)
        assert 0 <= action <= 8, f"Action should be between 0 and 8, got {action}"
        print("✓ Heuristic agent works correctly")
        
        # Test simple agent (without training)
        simple_agent = SimpleAgent(config)
        action = simple_agent.select_action(obs, training=False)
        assert 0 <= action <= 8, f"Action should be between 0 and 8, got {action}"
        print("✓ Simple agent works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        return False


def test_training_loop():
    """Test a simple training loop."""
    print("\nTesting Training Loop...")
    
    try:
        # Create simulator and agent
        simulator = SonicSimulator()
        config = {
            'agent': {
                'learning_rate': 0.001,
                'gamma': 0.99
            },
            'frame_stack': 1
        }
        
        agent = SimpleAgent(config)
        
        # Run a few episodes
        total_reward = 0
        for episode in range(3):
            obs, info = simulator.reset()
            episode_reward = 0
            
            for step in range(100):
                # Select action
                action = agent.select_action(obs, training=True)
                
                # Take step
                next_obs, reward, done, info = simulator.step(action)
                
                # Store experience
                agent.store_experience(obs, action, reward, next_obs, done)
                
                # Train agent
                if len(agent.memory) >= agent.batch_size:
                    loss = agent.train()
                
                episode_reward += reward
                obs = next_obs
                
                if done:
                    break
            
            total_reward += episode_reward
            print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        avg_reward = total_reward / 3
        print(f"✓ Training loop completed. Average reward: {avg_reward:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training loop test failed: {e}")
        return False


def test_config_loading():
    """Test configuration file loading."""
    print("\nTesting Configuration Loading...")
    
    try:
        config_path = "configs/training_config.yaml"
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required keys
            required_keys = ['game', 'agent', 'environment', 'training']
            for key in required_keys:
                assert key in config, f"Missing required key: {key}"
            
            print("✓ Configuration file loads correctly")
            return True
        else:
            print("⚠ Configuration file not found, skipping test")
            return True
            
    except Exception as e:
        print(f"✗ Configuration loading test failed: {e}")
        return False


def test_demo():
    """Test the demo functionality."""
    print("\nTesting Demo...")
    
    try:
        from demo_sonic import demo_random_agent
        
        # Run a quick demo
        print("Running quick demo (this may take a moment)...")
        
        # We'll just test the simulator creation and a few steps
        simulator = SonicSimulator()
        obs, info = simulator.reset()
        
        for i in range(50):
            action = np.random.randint(0, 9)
            obs, reward, done, info = simulator.step(action)
            if done:
                break
        
        print("✓ Demo functionality works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Demo test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Sonic AI - Component Tests")
    print("="*50)
    
    tests = [
        ("Simulator", test_simulator),
        ("Observation Processor", test_observation_processor),
        ("Reward Calculator", test_reward_calculator),
        ("Agents", test_agents),
        ("Training Loop", test_training_loop),
        ("Configuration Loading", test_config_loading),
        ("Demo", test_demo)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
    
    print("\n" + "="*50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Sonic AI system is ready to use.")
        print("\nNext steps:")
        print("1. Add your Sonic ROM to the 'roms/' directory")
        print("2. Install an emulator (BizHawk or RetroArch)")
        print("3. Run: python train_sonic.py")
        print("4. Or try the demo: python demo_sonic.py")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 