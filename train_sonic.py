#!/usr/bin/env python3
"""
Sonic AI Training Script

Main script for training reinforcement learning agents to play Sonic the Hedgehog.
Inspired by Pokemon Red Experiments by PWhiddy.
"""

import argparse
import os
import sys
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from environment.sonic_env import SonicEnvironment
from utils.observation_processor import SonicSpecificProcessor
from utils.reward_calculator import SonicSpecificRewardCalculator
from visualization.training_visualizer import TrainingVisualizer


def setup_logging(log_dir: str, level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_environment(config: Dict[str, Any], render: bool = False) -> SonicEnvironment:
    """Create and configure the Sonic environment."""
    # Update config for rendering if needed
    if render:
        config['game']['render'] = True
    
    # Create environment
    env = SonicEnvironment(config)
    
    # Wrap with monitor for logging
    env = Monitor(env)
    
    return env


def create_agent(agent_type: str, env, config: Dict[str, Any]) -> Any:
    """Create the RL agent based on configuration."""
    agent_config = config['agent']
    network_config = config['network']
    
    # Common parameters
    common_params = {
        'learning_rate': agent_config['learning_rate'],
        'gamma': agent_config['gamma'],
        'verbose': 1,
        'tensorboard_log': config['training']['log_dir']
    }
    
    # Network architecture
    if network_config['type'] == 'cnn':
        policy_kwargs = {
            'features_extractor_class': None,  # Use default CNN
            'features_extractor_kwargs': {
                'features_dim': network_config['cnn_features'][-1]
            }
        }
    elif network_config['type'] == 'mlp':
        policy_kwargs = {
            'net_arch': network_config['mlp_layers']
        }
    else:
        policy_kwargs = {}
    
    # Create agent
    if agent_type.lower() == 'ppo':
        agent = PPO(
            "CnnPolicy" if network_config['type'] == 'cnn' else "MlpPolicy",
            env,
            batch_size=agent_config['batch_size'],
            n_steps=agent_config.get('n_steps', 2048),
            clip_range=agent_config['clip_range'],
            ent_coef=agent_config['ent_coef'],
            vf_coef=agent_config['vf_coef'],
            max_grad_norm=agent_config['max_grad_norm'],
            policy_kwargs=policy_kwargs,
            **common_params
        )
    elif agent_type.lower() == 'a2c':
        agent = A2C(
            "CnnPolicy" if network_config['type'] == 'cnn' else "MlpPolicy",
            env,
            ent_coef=agent_config['ent_coef'],
            vf_coef=agent_config['vf_coef'],
            max_grad_norm=agent_config['max_grad_norm'],
            policy_kwargs=policy_kwargs,
            **common_params
        )
    elif agent_type.lower() == 'dqn':
        agent = DQN(
            "CnnPolicy" if network_config['type'] == 'cnn' else "MlpPolicy",
            env,
            buffer_size=agent_config['buffer_size'],
            learning_starts=agent_config.get('learning_starts', 1000),
            policy_kwargs=policy_kwargs,
            **common_params
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    return agent


def create_callbacks(config: Dict[str, Any], eval_env) -> list:
    """Create training callbacks."""
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_interval'],
        save_path=config['training']['checkpoint_dir'],
        name_prefix=f"sonic_{config['game']['name']}_{config['agent']['type']}"
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(config['training']['checkpoint_dir'], 'best'),
            log_path=config['training']['log_dir'],
            eval_freq=config['evaluation']['eval_freq'],
            n_eval_episodes=config['evaluation']['n_eval_episodes'],
            deterministic=config['evaluation']['deterministic'],
            render=config['evaluation']['render']
        )
        callbacks.append(eval_callback)
    
    return callbacks


def train_agent(agent, env, config: Dict[str, Any], callbacks: list, logger: logging.Logger):
    """Train the agent."""
    training_config = config['training']
    
    logger.info(f"Starting training for {training_config['total_timesteps']} timesteps")
    logger.info(f"Agent: {config['agent']['type']}")
    logger.info(f"Game: {config['game']['name']}")
    
    start_time = time.time()
    
    try:
        agent.learn(
            total_timesteps=training_config['total_timesteps'],
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    return agent


def save_final_model(agent, config: Dict[str, Any], logger: logging.Logger):
    """Save the final trained model."""
    model_path = os.path.join(
        config['training']['checkpoint_dir'],
        f"sonic_{config['game']['name']}_{config['agent']['type']}_final.pth"
    )
    
    agent.save(model_path)
    logger.info(f"Final model saved to {model_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Sonic AI agent")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--game", type=str, choices=["sonic1", "sonic2", "sonic3"],
                       help="Sonic game to train on")
    parser.add_argument("--agent", type=str, choices=["ppo", "a2c", "dqn"],
                       help="RL algorithm to use")
    parser.add_argument("--episodes", type=int, help="Number of episodes to train")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument("--eval", action="store_true", help="Enable evaluation")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.game:
        config['game']['name'] = args.game
    if args.agent:
        config['agent']['type'] = args.agent
    if args.episodes:
        # Convert episodes to timesteps (rough estimate)
        config['training']['total_timesteps'] = args.episodes * 1000
    
    # Create directories
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    
    # Setup logging
    logger = setup_logging(config['training']['log_dir'], args.log_level)
    
    # Check if ROM exists
    rom_path = config['game']['rom_path']
    if not os.path.exists(rom_path):
        logger.error(f"ROM file not found: {rom_path}")
        logger.error("Please place your Sonic ROM in the roms/ directory")
        return
    
    # Set device
    device = config['hardware']['device']
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        # Create environments
        logger.info("Creating training environment...")
        train_env = create_environment(config, render=args.render)
        
        eval_env = None
        if args.eval:
            logger.info("Creating evaluation environment...")
            eval_config = config.copy()
            eval_config['game']['render'] = True
            eval_env = create_environment(eval_config, render=True)
        
        # Create agent
        logger.info(f"Creating {config['agent']['type']} agent...")
        agent = create_agent(config['agent']['type'], train_env, config)
        
        # Create callbacks
        callbacks = create_callbacks(config, eval_env)
        
        # Train agent
        agent = train_agent(agent, train_env, config, callbacks, logger)
        
        # Save final model
        save_final_model(agent, config, logger)
        
        # Close environments
        train_env.close()
        if eval_env:
            eval_env.close()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 