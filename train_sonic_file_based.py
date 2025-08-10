#!/usr/bin/env python3
"""
File-Based Sonic Training Script

This script demonstrates training an AI model using the file-based environment approach.
It provides an alternative to the standard memory-based training when input isolation
issues occur.
"""

import os
import sys
import yaml
import time
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from environment.file_based_env import FileBasedSonicEnvironment


class FileBasedTrainingCallback(BaseCallback):
    """Custom callback for file-based training."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Track episode statistics
        if self.locals.get('dones'):
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
        else:
            self.current_episode_reward += self.locals.get('rewards', [0])[0]
            self.current_episode_length += 1
        
        # Log every 1000 steps
        if self.n_calls % 1000 == 0:
            if self.episode_rewards:
                avg_reward = np.mean(self.episode_rewards[-100:])  # Last 100 episodes
                avg_length = np.mean(self.episode_lengths[-100:])
                print(f"Step {self.n_calls}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.1f}")
        
        return True


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_file_based_env(config: Dict[str, Any], instance_id: int = 0):
    """Create a file-based environment."""
    def make_env():
        return FileBasedSonicEnvironment(config, instance_id=instance_id)
    return make_env


def train_file_based_model(config: Dict[str, Any], 
                          model_type: str = 'ppo',
                          total_timesteps: int = 100000,
                          num_envs: int = 1,
                          output_dir: str = "models/file_based_training"):
    """Train a model using the file-based environment."""
    
    print(f"🚀 Starting File-Based Training")
    print(f"   Model type: {model_type.upper()}")
    print(f"   Total timesteps: {total_timesteps}")
    print(f"   Number of environments: {num_envs}")
    print(f"   Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environments
    print("\n📁 Creating environments...")
    env_fns = [create_file_based_env(config, i) for i in range(num_envs)]
    
    if num_envs == 1:
        env = env_fns[0]()
    else:
        env = DummyVecEnv(env_fns)
        # Optional: normalize observations and rewards
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    print(f"✅ Created {num_envs} environment(s)")
    
    # Create model
    print(f"\n🤖 Creating {model_type.upper()} model...")
    
    if model_type.lower() == 'ppo':
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log=f"{output_dir}/tensorboard_logs"
        )
    elif model_type.lower() == 'a2c':
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            tensorboard_log=f"{output_dir}/tensorboard_logs"
        )
    elif model_type.lower() == 'dqn':
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            tensorboard_log=f"{output_dir}/tensorboard_logs"
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    print(f"✅ Created {model_type.upper()} model")
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // num_envs, 1),
        save_path=f"{output_dir}/checkpoints",
        name_prefix=f"sonic_{model_type}_file_based"
    )
    callbacks.append(checkpoint_callback)
    
    # Custom callback for logging
    training_callback = FileBasedTrainingCallback()
    callbacks.append(training_callback)
    
    # Train the model
    print(f"\n🎮 Starting training...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.1f} seconds")
        
        # Save final model
        final_model_path = f"{output_dir}/sonic_{model_type}_file_based_final"
        model.save(final_model_path)
        print(f"💾 Final model saved to: {final_model_path}")
        
        # Save environment normalization if using VecNormalize
        if hasattr(env, 'save'):
            env_path = f"{output_dir}/vec_normalize.pkl"
            env.save(env_path)
            print(f"💾 Environment normalization saved to: {env_path}")
        
        return model, env
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return model, env
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    finally:
        # Close environment
        env.close()


def test_trained_model(model, env, num_episodes: int = 3):
    """Test the trained model."""
    print(f"\n🧪 Testing trained model with {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        obs = env.reset()
        total_reward = 0
        step_count = 0
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"  Step {step_count}: Reward = {total_reward:.2f}")
            
            if done or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        print(f"  Episode finished: Total reward = {total_reward:.2f}, Steps = {step_count}")
    
    # Print summary
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\n📊 Test Results:")
    print(f"   Average reward: {avg_reward:.2f}")
    print(f"   Average episode length: {avg_length:.1f}")
    print(f"   Best episode reward: {max(episode_rewards):.2f}")
    print(f"   Worst episode reward: {min(episode_rewards):.2f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train Sonic AI using file-based environment")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model-type", type=str, choices=["ppo", "a2c", "dqn"], default="ppo",
                       help="Type of model to train")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total timesteps for training")
    parser.add_argument("--num-envs", type=int, default=1,
                       help="Number of parallel environments")
    parser.add_argument("--output-dir", type=str, default="models/file_based_training",
                       help="Output directory for models and logs")
    parser.add_argument("--test-only", action="store_true",
                       help="Only test an existing model (don't train)")
    parser.add_argument("--model-path", type=str,
                       help="Path to existing model for testing")
    
    args = parser.parse_args()
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    if args.test_only:
        # Test existing model
        if not args.model_path or not os.path.exists(args.model_path):
            print(f"❌ Model path not provided or model not found: {args.model_path}")
            return
        
        print(f"🧪 Testing existing model: {args.model_path}")
        
        # Create environment
        env = FileBasedSonicEnvironment(config, instance_id=0)
        
        # Load model
        if args.model_type.lower() == 'ppo':
            model = PPO.load(args.model_path, env=env)
        elif args.model_type.lower() == 'a2c':
            model = A2C.load(args.model_path, env=env)
        elif args.model_type.lower() == 'dqn':
            model = DQN.load(args.model_path, env=env)
        
        # Test model
        test_trained_model(model, env)
        env.close()
        
    else:
        # Train new model
        try:
            model, env = train_file_based_model(
                config=config,
                model_type=args.model_type,
                total_timesteps=args.timesteps,
                num_envs=args.num_envs,
                output_dir=args.output_dir
            )
            
            # Test the trained model
            test_trained_model(model, env)
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
