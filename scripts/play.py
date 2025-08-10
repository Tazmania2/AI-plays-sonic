#!/usr/bin/env python3
"""
Sonic AI Play Script

Script for playing Sonic the Hedgehog using trained AI models.
"""

import argparse
import os
import sys
import yaml
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from stable_baselines3 import PPO, A2C, DQN

# Add project root to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from environment.sonic_env import SonicEnvironment
from environment.file_based_env import FileBasedSonicEnvironment
from environment.direct_input_env import DirectInputSonicEnvironment
from visualization.training_visualizer import TrainingVisualizer, SonicGameVisualizer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path: str, agent_type: str, env) -> Any:
    """Load a trained model."""
    if agent_type.lower() == 'ppo':
        model = PPO.load(model_path, env=env)
    elif agent_type.lower() == 'a2c':
        model = A2C.load(model_path, env=env)
    elif agent_type.lower() == 'dqn':
        model = DQN.load(model_path, env=env)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    return model


def play_episode(model, env, render: bool = True, max_steps: int = 10000) -> Dict[str, Any]:
    """Play a single episode with the trained model."""
    obs, info = env.reset()
    total_reward = 0
    step_count = 0
    episode_data = {
        'rewards': [],
        'actions': [],
        'positions': [],
        'scores': [],
        'rings': []
    }
    
    print("Starting episode...")
    print("Controls: Press 'q' to quit, 'p' to pause/resume")
    
    paused = False
    
    while step_count < max_steps:
        if render:
            # Handle user input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Episode interrupted by user")
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
        
        if paused:
            time.sleep(0.1)
            continue
        
        # Get action from model
        action, _states = model.predict(obs, deterministic=True)
        
        # Take step in environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Record data
        total_reward += reward
        step_count += 1
        episode_data['rewards'].append(reward)
        episode_data['actions'].append(action)
        
        if 'position' in info:
            episode_data['positions'].append(info['position'])
        if 'score' in info:
            episode_data['scores'].append(info['score'])
        if 'rings' in info:
            episode_data['rings'].append(info['rings'])
        
        # Display info
        if step_count % 100 == 0:
            print(f"Step: {step_count}, Reward: {total_reward:.2f}, "
                  f"Score: {info.get('score', 0)}, Rings: {info.get('rings', 0)}")
        
        if render:
            env.render()
        
        if done or truncated:
            break
    
    print(f"Episode finished! Total reward: {total_reward:.2f}, Steps: {step_count}")
    
    return {
        'total_reward': total_reward,
        'steps': step_count,
        'final_score': info.get('score', 0),
        'final_rings': info.get('rings', 0),
        'episode_data': episode_data
    }


def play_multiple_episodes(model, env, num_episodes: int, render: bool = True) -> Dict[str, Any]:
    """Play multiple episodes and collect statistics."""
    episode_results = []
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        
        result = play_episode(model, env, render=render)
        episode_results.append(result)
        
        # Small delay between episodes
        time.sleep(1)
    
    # Calculate statistics
    total_rewards = [r['total_reward'] for r in episode_results]
    steps = [r['steps'] for r in episode_results]
    scores = [r['final_score'] for r in episode_results]
    rings = [r['final_rings'] for r in episode_results]
    
    stats = {
        'num_episodes': num_episodes,
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'max_reward': np.max(total_rewards),
        'min_reward': np.min(total_rewards),
        'avg_steps': np.mean(steps),
        'avg_score': np.mean(scores),
        'avg_rings': np.mean(rings),
        'episode_results': episode_results
    }
    
    return stats


def display_statistics(stats: Dict[str, Any]):
    """Display episode statistics."""
    print("\n" + "="*50)
    print("EPISODE STATISTICS")
    print("="*50)
    print(f"Number of episodes: {stats['num_episodes']}")
    print(f"Average reward: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Reward range: {stats['min_reward']:.2f} - {stats['max_reward']:.2f}")
    print(f"Average steps: {stats['avg_steps']:.1f}")
    print(f"Average score: {stats['avg_score']:.1f}")
    print(f"Average rings: {stats['avg_rings']:.1f}")
    print("="*50)


def save_episode_data(stats: Dict[str, Any], output_dir: str):
    """Save episode data for later analysis."""
    import json
    import pandas as pd
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary statistics
    summary = {k: v for k, v in stats.items() if k != 'episode_results'}
    with open(os.path.join(output_dir, 'episode_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed episode data
    episode_data = []
    for i, result in enumerate(stats['episode_results']):
        episode_info = {
            'episode': i + 1,
            'total_reward': result['total_reward'],
            'steps': result['steps'],
            'final_score': result['final_score'],
            'final_rings': result['final_rings']
        }
        episode_data.append(episode_info)
    
    df = pd.DataFrame(episode_data)
    df.to_csv(os.path.join(output_dir, 'episode_data.csv'), index=False)
    
    print(f"Episode data saved to {output_dir}")


def create_visualization(episode_results: list, output_dir: str):
    """Create visualization of episode results."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Sonic AI - Episode Analysis', fontsize=16, fontweight='bold')
    
    # Extract data
    rewards = [r['total_reward'] for r in episode_results]
    steps = [r['steps'] for r in episode_results]
    scores = [r['final_score'] for r in episode_results]
    rings = [r['final_rings'] for r in episode_results]
    episodes = range(1, len(episode_results) + 1)
    
    # Plot rewards
    axes[0, 0].plot(episodes, rewards, 'b-o', alpha=0.7)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot steps
    axes[0, 1].plot(episodes, steps, 'g-o', alpha=0.7)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot scores
    axes[1, 0].plot(episodes, scores, 'r-o', alpha=0.7)
    axes[1, 0].set_title('Final Scores')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot rings
    axes[1, 1].plot(episodes, rings, 'm-o', alpha=0.7)
    axes[1, 1].set_title('Final Rings')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Rings')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'episode_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function for playing Sonic with trained models."""
    parser = argparse.ArgumentParser(description="Play Sonic with trained AI model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model file")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--episodes", type=int, default=1,
                       help="Number of episodes to play")
    parser.add_argument("--render", action="store_true", default=True,
                       help="Render the game")
    parser.add_argument("--max-steps", type=int, default=10000,
                       help="Maximum steps per episode")
    parser.add_argument("--output-dir", type=str, default="play_results",
                       help="Directory to save results")
    parser.add_argument("--agent-type", type=str, choices=["ppo", "a2c", "dqn"],
                       help="Agent type (auto-detected from model if not specified)")
    parser.add_argument("--file-based", action="store_true",
                       help="Use file-based environment instead of direct memory access")
    parser.add_argument("--direct-input", action="store_true",
                       help="Use direct input environment (Windows API-based input injection)")
    parser.add_argument("--instance-id", type=int, default=0,
                       help="Instance ID for environment (default: 0)")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine agent type
    if args.agent_type:
        agent_type = args.agent_type
    else:
        # Try to infer from model filename
        model_name = os.path.basename(args.model).lower()
        if 'ppo' in model_name:
            agent_type = 'ppo'
        elif 'a2c' in model_name:
            agent_type = 'a2c'
        elif 'dqn' in model_name:
            agent_type = 'dqn'
        else:
            print("Warning: Could not determine agent type from model name.")
            print("Please specify --agent-type")
            return
    
    print(f"Loading {agent_type.upper()} model from: {args.model}")
    print(f"Playing {args.episodes} episode(s)")
    print(f"Rendering: {args.render}")
    if args.direct_input:
        env_type = "Direct Input"
    elif args.file_based:
        env_type = "File-based"
    else:
        env_type = "Standard"
    print(f"Environment type: {env_type}")
    
    try:
        # Create environment
        print("Creating environment...")
        if args.direct_input:
            env = DirectInputSonicEnvironment(config, env_id=args.instance_id)
            print(f"Using direct input environment with instance ID: {args.instance_id}")
        elif args.file_based:
            env = FileBasedSonicEnvironment(config, instance_id=args.instance_id)
            print(f"Using file-based environment with instance ID: {args.instance_id}")
        else:
            env = SonicEnvironment(config)
            print("Using standard environment with direct memory access")
        
        # Load model
        print("Loading model...")
        model = load_model(args.model, agent_type, env)
        
        # Play episodes
        if args.episodes == 1:
            result = play_episode(model, env, render=args.render, max_steps=args.max_steps)
            stats = {
                'num_episodes': 1,
                'avg_reward': result['total_reward'],
                'std_reward': 0,
                'max_reward': result['total_reward'],
                'min_reward': result['total_reward'],
                'avg_steps': result['steps'],
                'avg_score': result['final_score'],
                'avg_rings': result['final_rings'],
                'episode_results': [result]
            }
        else:
            stats = play_multiple_episodes(model, env, args.episodes, render=args.render)
        
        # Display statistics
        display_statistics(stats)
        
        # Save results
        if args.output_dir:
            save_episode_data(stats, args.output_dir)
            create_visualization(stats['episode_results'], args.output_dir)
        
        # Close environment
        env.close()
        
        print("Play session completed!")
        
    except Exception as e:
        print(f"Error during play session: {e}")
        raise


if __name__ == "__main__":
    main() 