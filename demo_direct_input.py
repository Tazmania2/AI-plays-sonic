#!/usr/bin/env python3
"""
Demonstration script for the Direct Input System

This script shows how to use the new direct input system that provides
Windows API-based input injection as the primary method, with file-based
communication as a secondary fallback.
"""

import sys
import yaml
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from environment.direct_input_env import DirectInputSonicEnvironment
from utils.direct_input_manager import get_direct_input_manager, shutdown_direct_input_manager

def demo_direct_input_basic():
    """Demonstrate basic direct input functionality."""
    print("🎮 Direct Input System Demonstration")
    print("=" * 50)
    
    try:
        # Load configuration
        config_path = Path("configs/training_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Configuration loaded")
        
        # Create direct input environment
        env = DirectInputSonicEnvironment(config, env_id=0)
        print("✅ Direct input environment created")
        
        # Show initial input method
        print(f"📊 Initial input method: {'direct' if env.use_direct_input else 'file'}")
        
        # Reset environment
        obs, info = env.reset()
        print(f"🔄 Environment reset - Input method: {info.get('input_method', 'unknown')}")
        
        # Take some steps to demonstrate input method switching
        print("\n🎯 Taking steps to demonstrate input method...")
        for step in range(10):
            # Take random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            input_method = info.get('input_method', 'unknown')
            print(f"  Step {step + 1}: Action={action}, Input={input_method}, Reward={reward:.2f}")
            
            if done:
                print("  Episode finished, resetting...")
                obs, info = env.reset()
        
        # Close environment
        env.close()
        print("✅ Environment closed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_input_manager():
    """Demonstrate the direct input manager functionality."""
    print("\n🔧 Direct Input Manager Demonstration")
    print("=" * 50)
    
    try:
        # Get direct input manager
        input_manager = get_direct_input_manager(num_instances=2)
        print("✅ Direct input manager created")
        
        # Check instance status
        status = input_manager.get_instance_status()
        print(f"📊 Instance status: {status}")
        
        # Show window detection
        for i in range(2):
            instance = input_manager.input_managers[i]
            if instance.target_hwnd:
                print(f"✅ Instance {i}: Found BizHawk window {instance.target_hwnd}")
            else:
                print(f"⚠️  Instance {i}: No BizHawk window found (expected if BizHawk not running)")
        
        # Test input sending (this won't work without BizHawk running, but shows the API)
        print("\n🎮 Testing input API...")
        try:
            input_manager.send_action(0, 'RIGHT', duration=0.1)
            print("✅ Input sent successfully")
        except Exception as e:
            print(f"⚠️  Input failed (expected without BizHawk): {e}")
        
        # Cleanup
        shutdown_direct_input_manager()
        print("✅ Input manager shutdown")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in input manager demonstration: {e}")
        return False

def show_usage_examples():
    """Show usage examples for the direct input system."""
    print("\n📖 Usage Examples")
    print("=" * 50)
    
    print("1. Using play_sonic.py with direct input:")
    print("   python play_sonic.py --model path/to/model --direct-input")
    print()
    
    print("2. Using play_sonic.py with file-based input (fallback):")
    print("   python play_sonic.py --model path/to/model --file-based")
    print()
    
    print("3. Using play_sonic.py with standard environment:")
    print("   python play_sonic.py --model path/to/model")
    print()
    
    print("4. Creating direct input environment in code:")
    print("   from environment.direct_input_env import DirectInputSonicEnvironment")
    print("   env = DirectInputSonicEnvironment(config, env_id=0)")
    print()
    
    print("5. Using direct input manager directly:")
    print("   from utils.direct_input_manager import get_direct_input_manager")
    print("   input_manager = get_direct_input_manager(num_instances=4)")
    print("   input_manager.send_action(env_id, 'RIGHT', duration=0.016)")
    print()

def main():
    """Main demonstration function."""
    print("🚀 Direct Input System - Complete Demonstration")
    print("=" * 60)
    
    # Demo 1: Basic functionality
    success1 = demo_direct_input_basic()
    
    # Demo 2: Input manager
    success2 = demo_input_manager()
    
    # Show usage examples
    show_usage_examples()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Demonstration Summary:")
    print(f"  Basic Functionality: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"  Input Manager: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if all([success1, success2]):
        print("\n🎉 All demonstrations passed!")
        print("\n💡 Key Features:")
        print("  • Primary: Windows API-based direct input injection")
        print("  • Secondary: File-based communication as fallback")
        print("  • Automatic switching between input methods")
        print("  • Multi-instance support for parallel environments")
        print("  • Thread-safe input queuing and processing")
        print("  • Window detection and targeting")
    else:
        print("\n⚠️  Some demonstrations failed. Check the output above.")
    
    return all([success1, success2])

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
