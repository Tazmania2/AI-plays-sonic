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
import csv
import json
from pathlib import Path
from typing import Dict, Any, Optional
import multiprocessing
import psutil
import torch
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure as sb3_configure
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from environment.sonic_env import SonicEnvironment
from utils.observation_processor import SonicSpecificProcessor
from utils.reward_calculator import SonicSpecificRewardCalculator
from visualization.training_visualizer import TrainingVisualizer
from environment.hierarchical_shaping_wrapper import HierarchicalShapingWrapper


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


def log_episode_metrics(metrics, log_dir, mode):
    """Log per-episode metrics to CSV and TensorBoard."""
    csv_path = os.path.join(log_dir, f"{mode}.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = list(metrics.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)
    # TensorBoard logging
    tb_log_dir = os.path.join(log_dir, mode)
    tb_logger = sb3_configure(tb_log_dir, ['tensorboard'])
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            tb_logger.record(k, v)
    tb_logger.dump(metrics.get('step', 0))


def train_agent(agent, env, config, callbacks, logger, log_dir=None, mode='baseline'):
    """Train the agent with enhanced logging and objective detection."""
    print(f"Starting {mode} training...")
    
    # Setup logging files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"logs/{mode}_training_{timestamp}.csv"
    json_filename = f"logs/{mode}_training_{timestamp}.json"
    session_summary_filename = f"logs/{mode}_session_summary_{timestamp}.json"
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Session tracking
    session_data = {
        "mode": mode,
        "start_time": datetime.now().isoformat(),
        "config": config,
        "episodes": [],
        "objective_completed": False,
        "final_progress": 0.0,
        "total_reward": 0.0,
        "total_steps": 0
    }
    
    training_config = config['training']
    start_time = time.time()
    episode = 0
    obs = env.reset()
    # Handle tuple return from environment reset (obs, info)
    if isinstance(obs, tuple):
        obs, _ = obs
    done, truncated = False, False
    episode_metrics = None
    
    print(f"Training for {training_config['total_timesteps']} timesteps...")
    
    for step in range(training_config['total_timesteps']):
        action, _ = agent.predict(obs, deterministic=False)
        result = env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
            truncated = False

        # Robust check for episode termination
        done_flag = (np.any(done) if isinstance(done, (np.ndarray, list)) else bool(done))
        truncated_flag = (np.any(truncated) if isinstance(truncated, (np.ndarray, list)) else bool(truncated))
        
        # Check for objective completion
        objective_completed = False
        objective_progress = 0.0
        if hasattr(env, 'check_objective_completed'):
            objective_completed = env.check_objective_completed()
        if hasattr(env, 'get_objective_progress'):
            objective_progress = env.get_objective_progress()
        
        if done_flag or truncated_flag:
            episode += 1
            # Handle info from vectorized environments (list of dicts)
            if isinstance(info, list):
                info_dict = info[0] if info else {}
            else:
                info_dict = info
            # Collect metrics from info
            episode_metrics = {
                'episode': episode,
                'step': step,
                'total_reward': info_dict.get('episode', {}).get('r', 0),
                'episode_length': info_dict.get('episode', {}).get('l', 0),
                'objective_progress': objective_progress,
                'objective_completed': objective_completed,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log episode metrics
            log_metrics_to_csv(episode_metrics, csv_filename)
            log_metrics_to_json(episode_metrics, json_filename)
            
            # Update session data
            session_data["episodes"].append(episode_metrics)
            session_data["total_reward"] += episode_metrics['total_reward']
            session_data["total_steps"] += episode_metrics['episode_length']
            session_data["final_progress"] = max(session_data["final_progress"], objective_progress)
            
            if objective_completed:
                session_data["objective_completed"] = True
                print(f"🎉 OBJECTIVE COMPLETED! Episode {episode} - End of Green Hill Zone Act 3 reached!")
                break
            
            if log_dir:
                log_episode_metrics(episode_metrics, log_dir, mode)
            obs = env.reset()
            # Handle tuple return from environment reset (obs, info)
            if isinstance(obs, tuple):
                obs, _ = obs
    
    # Finalize session data
    session_data["end_time"] = datetime.now().isoformat()
    session_data["duration_seconds"] = time.time() - start_time
    session_data["total_episodes"] = episode
    
    # Log session summary
    log_training_session_summary(session_data, session_summary_filename)
    
    print(f"Training completed!")
    print(f"Session summary saved to: {session_summary_filename}")
    print(f"Episode data saved to: {csv_filename} and {json_filename}")
    print(f"Objective completed: {session_data['objective_completed']}")
    print(f"Final progress: {session_data['final_progress']:.2%}")
    print(f"Total episodes: {episode}")
    print(f"Total reward: {session_data['total_reward']:.2f}")
    
    return agent


def save_final_model(agent, config: Dict[str, Any], logger: logging.Logger):
    """Save the final trained model."""
    model_path = os.path.join(
        config['training']['checkpoint_dir'],
        f"sonic_{config['game']['name']}_{config['agent']['type']}_final.pth"
    )
    
    agent.save(model_path)
    logger.info(f"Final model saved to {model_path}")


class EnvThunk:
    def __init__(self, config, render, shaping_phase_steps, mode):
        self.config = config
        self.render = render
        self.shaping_phase_steps = shaping_phase_steps
        self.mode = mode
    def __call__(self):
        env = create_environment(self.config, render=self.render)
        return HierarchicalShapingWrapper(env, reward_mode=self.mode, shaping_phase_steps=self.shaping_phase_steps)


def ab_train_process(config, env_fns, agent_type, log_dir, mode, render, shaping_phase_steps, logger, cpu_cores=None, gpu_mem_fraction=None):
    # Set CPU affinity
    if cpu_cores is not None:
        p = psutil.Process()
        try:
            p.cpu_affinity(cpu_cores)
        except Exception as e:
            logger.warning(f"Could not set CPU affinity: {e}")
    # Set per-process GPU memory fraction (PyTorch)
    if torch.cuda.is_available() and gpu_mem_fraction is not None:
        try:
            torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction)
        except Exception as e:
            logger.warning(f"Could not set GPU memory fraction: {e}")
    env = DummyVecEnv(env_fns)
    agent = create_agent(agent_type, env, config)
    train_agent(agent, env, config, [], logger, log_dir=log_dir, mode=mode)
    save_final_model(agent, config, logger)
    env.close()


def log_metrics_to_csv(metrics, filename):
    """Log metrics to CSV file."""
    file_exists = os.path.exists(filename)
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = list(metrics.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def log_metrics_to_json(metrics, filename):
    """Log metrics to JSON file."""
    data = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    
    data.append(metrics)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def log_training_session_summary(session_data, filename):
    """Log complete training session summary to JSON."""
    with open(filename, 'w') as f:
        json.dump(session_data, f, indent=2)


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
    # New CLI flags
    parser.add_argument("--reward_mode", type=str, choices=["baseline", "shaping", "both"], default="baseline",
                        help="Reward mode: baseline, shaping, or both (A/B test)")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--shaping_phase_steps", type=int, default=500_000, help="Steps for shaping phase")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.game:
        config['game']['name'] = args.game
    if args.agent:
        config['agent']['type'] = args.agent
    if args.episodes:
        config['training']['total_timesteps'] = args.episodes * 1000
    # New CLI overrides
    config['reward_mode'] = args.reward_mode
    config['num_envs'] = args.num_envs
    config['shaping_phase_steps'] = args.shaping_phase_steps
    
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
        logger.info("Creating training environment(s)...")
        if args.reward_mode == "both":
            n_envs = args.num_envs
            n_envs_baseline = n_envs // 2
            n_envs_shaping = n_envs - n_envs_baseline
            # Use EnvThunk for Windows compatibility
            env_fns_baseline = [EnvThunk(config, args.render, args.shaping_phase_steps, 'baseline') for _ in range(n_envs_baseline)]
            env_fns_shaping = [EnvThunk(config, args.render, args.shaping_phase_steps, 'shaping') for _ in range(n_envs_shaping)]
            # CPU affinity: baseline (0,1,2,6,7,8), shaping (3,4,5,9,10,11)
            baseline_cores = [0,1,2,6,7,8]
            shaping_cores = [3,4,5,9,10,11]
            gpu_mem_fraction = 0.45  # Each process gets ~45% of GPU
            logger.info("Launching parallel A/B training processes with CPU affinity and GPU memory split...")
            baseline_proc = multiprocessing.Process(
                target=ab_train_process,
                args=(config, env_fns_baseline, 'ppo', config['training']['log_dir'], "baseline", args.render, args.shaping_phase_steps, logger, baseline_cores, gpu_mem_fraction)
            )
            shaping_proc = multiprocessing.Process(
                target=ab_train_process,
                args=(config, env_fns_shaping, 'ppo', config['training']['log_dir'], "shaping", args.render, args.shaping_phase_steps, logger, shaping_cores, gpu_mem_fraction)
            )
            baseline_proc.start()
            shaping_proc.start()
            baseline_proc.join()
            shaping_proc.join()
            logger.info("A/B training completed!")
            return
        else:
            # Single reward mode
            base_env = create_environment(config, render=args.render)
            env = HierarchicalShapingWrapper(base_env, reward_mode=args.reward_mode, shaping_phase_steps=args.shaping_phase_steps)
            # Create agent
            logger.info(f"Creating {config['agent']['type']} agent...")
            agent = create_agent(config['agent']['type'], env, config)
            # Create callbacks
            callbacks = create_callbacks(config, None)
            # Train agent
            agent = train_agent(agent, env, config, callbacks, logger, log_dir=config['training']['log_dir'], mode=args.reward_mode)
            # Save final model
            save_final_model(agent, config, logger)
            env.close()
            logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 