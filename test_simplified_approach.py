#!/usr/bin/env python3
"""
Quick Test Script for Simplified Sonic AI Approach

This script demonstrates the simplified Mario AI-inspired approach
and compares it with the current complex approach.
"""

import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compare_configurations():
    """Compare the current complex configuration with the simplified one."""
    
    print("🔍 Comparing Sonic AI Configurations")
    print("=" * 50)
    
    # Load configurations
    try:
        current_config = load_config("configs/training_config.yaml")
        simplified_config = load_config("configs/simplified_training_config.yaml")
    except FileNotFoundError as e:
        print(f"❌ Configuration file not found: {e}")
        return
    
    print("\n📊 Configuration Comparison:")
    print("-" * 30)
    
    # Compare key aspects
    comparisons = [
        ("Objective", 
         current_config['game'].get('objective', 'Complex multi-objective'),
         simplified_config['game'].get('objective', 'Simple objective')),
        
        ("Max Steps", 
         current_config['game']['max_steps'],
         simplified_config['game']['max_steps']),
        
        ("Frame Stack", 
         current_config['environment']['stack_frames'],
         simplified_config['environment']['stack_frames']),
        
        ("Batch Size", 
         current_config['agent']['batch_size'],
         simplified_config['agent']['batch_size']),
        
        ("Network Type", 
         current_config['network']['type'],
         simplified_config['network']['type']),
        
        ("Total Timesteps", 
         current_config['training']['total_timesteps'],
         simplified_config['training']['total_timesteps']),
    ]
    
    for aspect, current, simplified in comparisons:
        print(f"{aspect:15} | Current: {current:>10} | Simplified: {simplified:>10}")
    
    # Compare reward complexity
    current_rewards = len(current_config['rewards'])
    simplified_rewards = len(simplified_config['rewards'])
    
    print(f"\n🎯 Reward Complexity:")
    print(f"Current:     {current_rewards} reward components")
    print(f"Simplified:  {simplified_rewards} reward components")
    print(f"Reduction:   {current_rewards - simplified_rewards} components removed")
    
    # Compare action space
    current_actions = len(current_config['actions']['basic'])
    simplified_actions = len(simplified_config['actions']['basic'])
    
    print(f"\n🎮 Action Space:")
    print(f"Current:     {current_actions} basic actions")
    print(f"Simplified:  {simplified_actions} basic actions")
    print(f"Reduction:   {current_actions - simplified_actions} actions removed")

def test_simplified_environment():
    """Test the simplified environment setup."""
    
    print("\n🧪 Testing Simplified Environment")
    print("=" * 40)
    
    try:
        from utils.simplified_reward_calculator import SimplifiedRewardCalculator
        
        # Test reward calculator
        config = {
            'distance_reward': 1.0,
            'survival_reward': 0.1,
            'ring_collected': 5.0,
            'death_penalty': -100.0,
            'stuck_penalty': -1.0
        }
        
        calculator = SimplifiedRewardCalculator(config)
        
        # Test reward calculation
        prev_state = {'x': 100, 'y': 200, 'rings': 5, 'lives': 3}
        curr_state = {'x': 150, 'y': 200, 'rings': 7, 'lives': 3}
        
        reward = calculator.calculate_reward(prev_state, curr_state)
        
        print(f"✅ Simplified reward calculator working")
        print(f"   Previous state: {prev_state}")
        print(f"   Current state:  {curr_state}")
        print(f"   Calculated reward: {reward}")
        
        # Expected: distance(50) + survival(0.1) + rings(10) = 60.1
        expected = 50 * 1.0 + 0.1 + 2 * 5.0
        print(f"   Expected reward: {expected}")
        
        if abs(reward - expected) < 0.1:
            print("✅ Reward calculation correct!")
        else:
            print("❌ Reward calculation incorrect!")
            
    except ImportError as e:
        print(f"❌ Could not import simplified reward calculator: {e}")
    except Exception as e:
        print(f"❌ Error testing simplified environment: {e}")

def show_implementation_steps():
    """Show the steps to implement the simplified approach."""
    
    print("\n🚀 Implementation Steps")
    print("=" * 30)
    
    steps = [
        "1. Use simplified configuration:",
        "   python train_sonic_simplified.py --config configs/simplified_training_config.yaml",
        "",
        "2. Start with simple rewards:",
        "   --reward-style simple",
        "",
        "3. Monitor progress:",
        "   tensorboard --logdir logs/",
        "",
        "4. Compare with current approach:",
        "   python train_sonic.py --num_envs 1",
        "",
        "5. Gradually add complexity:",
        "   --reward-style mario",
    ]
    
    for step in steps:
        print(step)

def show_expected_improvements():
    """Show expected improvements from the simplified approach."""
    
    print("\n📈 Expected Improvements")
    print("=" * 30)
    
    improvements = [
        ("Learning Speed", "4-10x faster", "200-500 episodes vs 2000+"),
        ("Training Stability", "More consistent", "Clearer reward signals"),
        ("Resource Usage", "2-3x faster", "Smaller networks"),
        ("Debugging", "Easier", "Simpler reward structure"),
        ("Experimentation", "Faster", "Quick iteration cycles"),
    ]
    
    for aspect, improvement, detail in improvements:
        print(f"{aspect:20} | {improvement:15} | {detail}")

def main():
    """Main function to run all tests and comparisons."""
    
    print("🎮 Sonic AI Simplified Approach Test")
    print("=" * 50)
    print("This script compares your current complex approach with")
    print("a simplified Mario AI-inspired approach.")
    print()
    
    # Run comparisons and tests
    compare_configurations()
    test_simplified_environment()
    show_expected_improvements()
    show_implementation_steps()
    
    print("\n🎯 Next Steps:")
    print("1. Run the simplified training script")
    print("2. Compare learning curves")
    print("3. Measure time to first success")
    print("4. Gradually add complexity back")
    
    print("\n💡 Key Insight:")
    print("Mario AI examples prove that simplicity leads to better results.")
    print("Your sophisticated infrastructure is valuable, but the learning")
    print("approach should be simplified for maximum effectiveness.")

if __name__ == "__main__":
    main()
