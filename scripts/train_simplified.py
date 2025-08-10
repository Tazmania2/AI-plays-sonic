#!/usr/bin/env python3
"""
Simplified Sonic AI Training Script - Mario AI Inspired

This script implements a simplified training approach based on successful Mario AI examples.
Key principles:
1. Simple, clear objectives
2. Distance-based rewards
3. Minimal complexity for faster learning
4. Focus on core gameplay mechanics
"""

import argparse
import os
import sys
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, Any
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from environment.sonic_env import SonicEnvironment
from utils.simplified_reward_calculator import SimplifiedRewardCalculator, MarioStyleRewardCalculator

def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging for the training session."""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"simplified_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_simplified_environment(config: Dict[str, Any], env_id: int = 0) -> SonicEnvironment:
    """Create a simplified Sonic environment."""
    # Override reward calculator with simplified version
    config['reward_calculator'] = 'simplified'
    
    # Ensure simplified objective is set
    config['game']['objective'] = 'Move right and survive'
    
    env = SonicEnvironment(config, env_id=env_id)
    return Monitor(env)

def create_simplified_agent(env, config: Dict[str, Any]) -> PPO:
    """Create a simplified PPO agent."""
    agent_config = config['agent']
    
    # Simplified network architecture
    policy_kwargs = {
        'net_arch': [dict(pi=agent_config.get('mlp_layers', [256, 128]), 
                          vf=agent_config.get('mlp_layers', [256, 128]))]
    }
    
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=agent_config['learning_rate'],
        n_steps=agent_config['n_steps'],
        batch_size=agent_config['batch_size'],
        n_epochs=4,  # Reduced for faster updates
        gamma=agent_config['gamma'],
        gae_lambda=agent_config['gae_lambda'],
        clip_range=agent_config['clip_range'],
        ent_coef=agent_config['ent_coef'],
        vf_coef=agent_config['vf_coef'],
        max_grad_norm=agent_config['max_grad_norm'],
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=config['training']['log_dir']
    )

def create_simplified_callbacks(config: Dict[str, Any], eval_env) -> list:
    """Create simplified callbacks for training."""
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_interval'],
        save_path=config['training']['checkpoint_dir'],
        name_prefix="simplified_sonic"
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config['training']['checkpoint_dir'],
        log_path=config['training']['log_dir'],
        eval_freq=config['evaluation']['eval_freq'],
        n_eval_episodes=config['evaluation']['n_eval_episodes'],
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    return callbacks

def train_simplified_agent(agent, env, config: Dict[str, Any], logger: logging.Logger):
    """Train the simplified agent."""
    training_config = config['training']
    
    logger.info("Starting simplified Sonic AI training...")
    logger.info(f"Objective: {config['game']['objective']}")
    logger.info(f"Total timesteps: {training_config['total_timesteps']}")
    logger.info(f"Learning rate: {config['agent']['learning_rate']}")
    logger.info(f"Batch size: {config['agent']['batch_size']}")
    
    # Create callbacks
    callbacks = create_simplified_callbacks(config, env)
    
    # Train the agent
    start_time = time.time()
    
    try:
        agent.learn(
            total_timesteps=training_config['total_timesteps'],
            callback=callbacks,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        final_model_path = Path(config['training']['checkpoint_dir']) / "simplified_sonic_final"
        agent.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save intermediate model
        final_model_path = Path(config['training']['checkpoint_dir']) / "simplified_sonic_interrupted"
        agent.save(final_model_path)
        logger.info(f"Intermediate model saved to: {final_model_path}")

def main():
    """Main function for simplified training."""
    parser = argparse.ArgumentParser(description="Simplified Sonic AI Training")
    parser.add_argument("--config", type=str, default="configs/simplified_training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--reward-style", type=str, choices=["simple", "mario"], default="simple",
                       help="Reward style: simple or mario-style")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of episodes to train")
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return
    
    config = load_config(args.config)
    
    # Adjust timesteps based on episodes
    config['training']['total_timesteps'] = args.episodes * config['game']['max_steps']
    
    # Setup logging
    logger = setup_logging(config['training']['log_dir'])
    
    # Log configuration
    logger.info("Simplified Sonic AI Training Configuration:")
    logger.info(f"Reward style: {args.reward_style}")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Max steps per episode: {config['game']['max_steps']}")
    logger.info(f"Total timesteps: {config['training']['total_timesteps']}")
    
    # Create environment
    logger.info("Creating simplified environment...")
    env = create_simplified_environment(config)
    
    # Create agent
    logger.info("Creating simplified agent...")
    agent = create_simplified_agent(env, config)
    
    # Train agent
    train_simplified_agent(agent, env, config, logger)
    
    # Cleanup
    env.close()
    logger.info("Training session completed")

if __name__ == "__main__":
    main()
