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
import signal
import atexit
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
from utils.input_isolator import get_input_manager, shutdown_input_manager

# Global variables for cleanup
current_agent = None
current_env = None
current_processes = []
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    global shutdown_requested
    print(f"\n🛑 Received signal {signum}. Initiating graceful shutdown...")
    shutdown_requested = True
    cleanup_resources()

def cleanup_resources():
    """Clean up all resources gracefully."""
    global current_agent, current_env, current_processes
    
    print("🧹 Cleaning up resources...")
    
    # Close environment
    if current_env is not None:
        try:
            current_env.close()
            print("✅ Environment closed")
        except Exception as e:
            print(f"⚠️  Error closing environment: {e}")
    
    # Shutdown input manager
    try:
        shutdown_input_manager()
        print("✅ Input manager shutdown")
    except Exception as e:
        print(f"⚠️  Error shutting down input manager: {e}")
    
    # Terminate child processes
    for proc in current_processes:
        if proc.is_alive():
            try:
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                print(f"✅ Process {proc.name} terminated")
            except Exception as e:
                print(f"⚠️  Error terminating process {proc.name}: {e}")
    
    # Clear lists
    current_processes.clear()
    current_agent = None
    current_env = None
    
    print("✅ Cleanup complete")

def register_cleanup():
    """Register cleanup function to run on exit."""
    atexit.register(cleanup_resources)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
register_cleanup()

def setup_logging(log_dir: str, level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("SonicAI")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create handlers
    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_environment(config: Dict[str, Any], render: bool = False, env_id: int = 0) -> SonicEnvironment:
    """Create and configure the Sonic environment."""
    # Update config for rendering if needed
    if render:
        config['game']['render'] = True
    
    # Create environment with specific env_id for input isolation
    env = SonicEnvironment(config, env_id=env_id)
    
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
    """Picklable environment thunk for multiprocessing."""
    def __init__(self, config, render, shaping_phase_steps, mode, env_id):
        # Store only the necessary data for pickling
        self.config = config
        self.render = render
        self.shaping_phase_steps = shaping_phase_steps
        self.mode = mode
        self.env_id = env_id
    
    def __call__(self):
        """Create and return the environment."""
        try:
            # Create base environment
            env = create_environment(self.config, render=self.render, env_id=self.env_id)
            
            # Wrap with hierarchical shaping
            wrapped_env = HierarchicalShapingWrapper(
                env, 
                reward_mode=self.mode, 
                shaping_phase_steps=self.shaping_phase_steps
            )
            
            return wrapped_env
        except Exception as e:
            print(f"Error creating environment {self.env_id} for {self.mode}: {e}")
            raise


def ab_train_process(config, env_fns, agent_type, log_dir, mode, render, shaping_phase_steps, cpu_cores=None, gpu_mem_fraction=None):
    """A/B training process that runs in a separate process."""
    # Create a new logger for this process
    logger = setup_logging(log_dir, "INFO")
    logger.info(f"Starting {mode} training process...")
    
    # Set CPU affinity
    if cpu_cores is not None:
        p = psutil.Process()
        try:
            p.cpu_affinity(cpu_cores)
            logger.info(f"Set CPU affinity to cores: {cpu_cores}")
        except Exception as e:
            logger.warning(f"Could not set CPU affinity: {e}")
    
    # Set per-process GPU memory fraction (PyTorch)
    if torch.cuda.is_available() and gpu_mem_fraction is not None:
        try:
            torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction)
            logger.info(f"Set GPU memory fraction to: {gpu_mem_fraction}")
        except Exception as e:
            logger.warning(f"Could not set GPU memory fraction: {e}")
    
    try:
        # Create input manager for this process (each process gets its own)
        from utils.input_isolator import get_input_manager
        input_manager = get_input_manager(num_instances=4)
        logger.info(f"Created input manager for {mode} process")
        
        # Create environments - each process manages its own environments
        logger.info(f"Creating {len(env_fns)} environments for {mode} training...")
        
        # Create vectorized environment directly from environment functions
        env = DummyVecEnv(env_fns)
        
        # Create agent
        logger.info(f"Creating {agent_type} agent for {mode} training...")
        agent = create_agent(agent_type, env, config)
        
        # Train agent
        logger.info(f"Starting training for {mode} mode...")
        train_agent(agent, env, config, [], logger, log_dir=log_dir, mode=mode)
        
        # Save final model
        save_final_model(agent, config, logger)
        
        # Cleanup
        env.close()
        input_manager.shutdown()
        logger.info(f"{mode} training process completed successfully!")
        
    except Exception as e:
        logger.error(f"{mode} training process failed: {e}")
        raise


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
    global current_agent, current_env, current_processes, shutdown_requested
    
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
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
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
        # Initialize input manager for multiple environments (only for single process)
        if args.num_envs > 1 and args.reward_mode != "both":
            input_manager = get_input_manager(args.num_envs)
            logger.info(f"Initialized input manager for {args.num_envs} environments")
        
        # Create environments
        logger.info("Creating training environment(s)...")
        if args.reward_mode == "both":
            n_envs = args.num_envs
            n_envs_baseline = n_envs // 2
            n_envs_shaping = n_envs - n_envs_baseline
            
            # Create environment functions with proper env_ids
            env_fns_baseline = []
            env_fns_shaping = []
            
            for i in range(n_envs_baseline):
                env_fns_baseline.append(EnvThunk(config, args.render, args.shaping_phase_steps, 'baseline', env_id=i))
            
            for i in range(n_envs_shaping):
                env_fns_shaping.append(EnvThunk(config, args.render, args.shaping_phase_steps, 'shaping', env_id=i+n_envs_baseline))
            
            # CPU affinity: baseline (0,1,2,6,7,8), shaping (3,4,5,9,10,11)
            baseline_cores = [0,1,2,6,7,8]
            shaping_cores = [3,4,5,9,10,11]
            gpu_mem_fraction = 0.45  # Each process gets ~45% of GPU
            logger.info("Launching parallel A/B training processes with CPU affinity and GPU memory split...")
            baseline_proc = multiprocessing.Process(
                target=ab_train_process,
                args=(config, env_fns_baseline, 'ppo', config['training']['log_dir'], "baseline", args.render, args.shaping_phase_steps, baseline_cores, gpu_mem_fraction)
            )
            shaping_proc = multiprocessing.Process(
                target=ab_train_process,
                args=(config, env_fns_shaping, 'ppo', config['training']['log_dir'], "shaping", args.render, args.shaping_phase_steps, shaping_cores, gpu_mem_fraction)
            )
            
            # Add processes to global list for cleanup
            current_processes.extend([baseline_proc, shaping_proc])
            
            baseline_proc.start()
            shaping_proc.start()
            
            # Monitor processes and check for shutdown
            while baseline_proc.is_alive() or shaping_proc.is_alive():
                if shutdown_requested:
                    logger.info("Shutdown requested, terminating processes...")
                    break
                time.sleep(1)
            
            if not shutdown_requested:
                baseline_proc.join()
                shaping_proc.join()
                logger.info("A/B training completed!")
            return
        else:
            # Single reward mode with multiple environments
            if args.num_envs > 1:
                # Create multiple environments with input isolation
                env_fns = []
                for i in range(args.num_envs):
                    env_fns.append(EnvThunk(config, args.render, args.shaping_phase_steps, args.reward_mode, env_id=i))
                
                env = DummyVecEnv(env_fns)
                current_env = env  # Store for cleanup
                logger.info(f"Created {args.num_envs} environments with input isolation")
            else:
                # Single environment
                base_env = create_environment(config, render=args.render, env_id=0)
                env = HierarchicalShapingWrapper(base_env, reward_mode=args.reward_mode, shaping_phase_steps=args.shaping_phase_steps)
                current_env = env  # Store for cleanup
            
            # Create agent
            logger.info(f"Creating {config['agent']['type']} agent...")
            agent = create_agent(config['agent']['type'], env, config)
            current_agent = agent  # Store for cleanup
            
            # Create callbacks
            callbacks = create_callbacks(config, None)
            
            # Check for shutdown before training
            if shutdown_requested:
                logger.info("Shutdown requested before training started")
                return
            
            # Train agent
            agent = train_agent(agent, env, config, callbacks, logger, log_dir=config['training']['log_dir'], mode=args.reward_mode)
            
            # Check for shutdown after training
            if not shutdown_requested:
                # Save final model
                save_final_model(agent, config, logger)
                logger.info("Training completed successfully!")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup input manager (only if created in main process)
        if args.num_envs > 1 and args.reward_mode != "both":
            shutdown_input_manager()
            logger.info("Input manager shutdown complete")
        
        # Clear global variables
        current_agent = None
        current_env = None
        current_processes.clear()


if __name__ == "__main__":
    main() 