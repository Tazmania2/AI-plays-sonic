#!/usr/bin/env python3
"""
Test script for the direct input system.
This script tests the Windows API-based direct input injection to BizHawk.
"""

import sys
import time
import yaml
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.direct_input_manager import get_direct_input_manager, shutdown_direct_input_manager
from environment.direct_input_env import DirectInputSonicEnvironment

def test_direct_input_manager():
    """Test the direct input manager functionality."""
    print("🔍 Testing Direct Input Manager...")
    
    try:
        # Get the direct input manager
        input_manager = get_direct_input_manager(num_instances=1)
        
        if not input_manager:
            print("❌ Failed to create direct input manager")
            return False
        
        print("✅ Direct input manager created successfully")
        
        # Check instance status
        status = input_manager.get_instance_status()
        print(f"📊 Instance status: {status}")
        
        # Test window detection
        instance_0 = input_manager.input_managers[0]
        if instance_0.target_hwnd:
            print(f"✅ Found BizHawk window: {instance_0.target_hwnd}")
        else:
            print("⚠️  No BizHawk window found - this is expected if BizHawk is not running")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing direct input manager: {e}")
        return False

def test_direct_input_environment():
    """Test the direct input environment."""
    print("\n🔍 Testing Direct Input Environment...")
    
    try:
        # Load configuration
        config_path = Path("configs/training_config.yaml")
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Configuration loaded successfully")
        
        # Create environment
        env = DirectInputSonicEnvironment(config, env_id=0)
        print("✅ Direct input environment created successfully")
        
        # Test reset
        print("🔄 Testing environment reset...")
        obs, info = env.reset()
        print(f"✅ Reset successful - Observation shape: {obs.shape}")
        print(f"📊 Info: {info}")
        
        # Test a few steps
        print("🎮 Testing environment steps...")
        for step in range(5):
            # Take random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            print(f"  Step {step + 1}: Action={action}, Reward={reward:.2f}, Done={done}")
            print(f"    Input method: {info.get('input_method', 'unknown')}")
            
            if done:
                print("  Episode finished, resetting...")
                obs, info = env.reset()
        
        # Close environment
        env.close()
        print("✅ Environment closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing direct input environment: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_input_method_switching():
    """Test the automatic switching between input methods."""
    print("\n🔍 Testing Input Method Switching...")
    
    try:
        # Load configuration
        config_path = Path("configs/training_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create environment
        env = DirectInputSonicEnvironment(config, env_id=1)
        print("✅ Environment created for input method switching test")
        
        # Check initial input method
        print(f"📊 Initial input method: {'direct' if env.use_direct_input else 'file'}")
        
        # Test reset to see input method
        obs, info = env.reset()
        print(f"📊 Reset input method: {info.get('input_method', 'unknown')}")
        
        # Take a few steps and monitor input method
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            input_method = info.get('input_method', 'unknown')
            print(f"  Step {step + 1}: Input method = {input_method}")
            
            if done:
                obs, info = env.reset()
        
        env.close()
        print("✅ Input method switching test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing input method switching: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting Direct Input System Tests")
    print("=" * 50)
    
    # Test 1: Direct Input Manager
    success1 = test_direct_input_manager()
    
    # Test 2: Direct Input Environment
    success2 = test_direct_input_environment()
    
    # Test 3: Input Method Switching
    success3 = test_input_method_switching()
    
    # Cleanup
    try:
        shutdown_direct_input_manager()
        print("✅ Direct input manager shutdown")
    except:
        pass
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"  Direct Input Manager: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"  Direct Input Environment: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"  Input Method Switching: {'✅ PASS' if success3 else '❌ FAIL'}")
    
    if all([success1, success2, success3]):
        print("\n🎉 All tests passed! Direct input system is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
