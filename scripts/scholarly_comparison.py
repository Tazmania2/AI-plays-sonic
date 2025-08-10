#!/usr/bin/env python3
"""
Scholarly Comparison: Traditional RL vs. Shaping-Based RL
Research Objective: Compare learning efficiency between standard RL and dog training-inspired shaping methods

This script implements a rigorous scientific comparison between:
- Method A: Traditional Reinforcement Learning (Control Group)
- Method B: Shaping-Based Reinforcement Learning (Experimental Group)

Based on Susan Garrett's dog training principles and behavioral psychology.
"""

import argparse
import os
import sys
import yaml
import time
import logging
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
import multiprocessing as mp
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from environment.sonic_env import SonicEnvironment
from environment.hierarchical_shaping_wrapper import HierarchicalShapingWrapper
from utils.reward_calculator import SonicSpecificRewardCalculator
from utils.simplified_reward_calculator import SimplifiedRewardCalculator

class ScholarlyComparison:
    """
    Scholarly comparison framework for Traditional RL vs. Shaping-Based RL.
    
    Implements proper experimental methodology with:
    - Controlled variables
    - Statistical analysis
    - Comprehensive metrics
    - Reproducible results
    """
    
    def __init__(self, config_path: str):
        """Initialize the scholarly comparison framework."""
        self.config = self.load_config(config_path)
        self.experiment_name = self.config['experiment']['name']
        self.results_dir = Path(f"results/{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self.setup_logging()
        
        # Initialize results storage
        self.traditional_rl_results = []
        self.shaping_rl_results = []
        
        # Statistical analysis parameters
        self.confidence_level = self.config['statistical_analysis']['confidence_level']
        self.significance_threshold = self.config['statistical_analysis']['significance_threshold']
        self.effect_size_threshold = self.config['statistical_analysis']['effect_size_threshold']
        self.sample_size = self.config['statistical_analysis']['sample_size']
        self.random_seeds = self.config['statistical_analysis']['random_seeds']
        
        self.logger.info(f"Initialized scholarly comparison: {self.experiment_name}")
        self.logger.info(f"Results directory: {self.results_dir}")
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for the experiment."""
        log_dir = self.results_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"scholarly_comparison_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger("ScholarlyComparison")
    
    def create_traditional_rl_environment(self, env_id: int = 0, seed: int = None) -> SonicEnvironment:
        """Create environment for Traditional RL (Method A - Control Group)."""
        # Use standard Sonic environment with traditional rewards
        env_config = self.config.copy()
        env_config['rewards'] = self.config['traditional_rl_rewards']
        
        env = SonicEnvironment(env_config, env_id=env_id)
        if seed is not None:
            env.seed(seed)
        
        return Monitor(env)
    
    def create_shaping_rl_environment(self, env_id: int = 0, seed: int = None) -> HierarchicalShapingWrapper:
        """Create environment for Shaping-Based RL (Method B - Experimental Group)."""
        # Use base environment wrapped with hierarchical shaping
        env_config = self.config.copy()
        env_config['rewards'] = self.config['shaping_rl_rewards']
        
        base_env = SonicEnvironment(env_config, env_id=env_id)
        if seed is not None:
            base_env.seed(seed)
        
        # Wrap with hierarchical shaping (dog training inspired)
        shaped_env = HierarchicalShapingWrapper(
            base_env,
            reward_mode='shaping',
            shaping_phase_steps=self.config['training']['total_timesteps'] // 2
        )
        
        return Monitor(shaped_env)
    
    def create_agent(self, env, method_name: str) -> PPO:
        """Create PPO agent with identical architecture for both methods."""
        agent_config = self.config['agent']
        network_config = self.config['network']
        
        policy_kwargs = {
            'net_arch': [dict(pi=network_config['mlp_layers'], 
                              vf=network_config['mlp_layers'])]
        }
        
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=agent_config['learning_rate'],
            n_steps=agent_config['n_steps'],
            batch_size=agent_config['batch_size'],
            gamma=agent_config['gamma'],
            gae_lambda=agent_config['gae_lambda'],
            clip_range=agent_config['clip_range'],
            ent_coef=agent_config['ent_coef'],
            vf_coef=agent_config['vf_coef'],
            max_grad_norm=agent_config['max_grad_norm'],
            policy_kwargs=policy_kwargs,
            verbose=0,  # Reduce output for cleaner logs
            tensorboard_log=str(self.results_dir / "tensorboard" / method_name)
        )
    
    def train_method(self, method_name: str, seed: int, run_id: int) -> Dict[str, Any]:
        """Train a single method with comprehensive metrics collection."""
        self.logger.info(f"Training {method_name} - Seed: {seed}, Run: {run_id}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create environment based on method
        if method_name == "traditional_rl":
            env = self.create_traditional_rl_environment(env_id=run_id, seed=seed)
        elif method_name == "shaping_rl":
            env = self.create_shaping_rl_environment(env_id=run_id, seed=seed)
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        # Create agent
        agent = self.create_agent(env, method_name)
        
        # Training metrics collection
        training_metrics = {
            'method': method_name,
            'seed': seed,
            'run_id': run_id,
            'episodes': [],
            'rewards': [],
            'scores': [],
            'completion_times': [],
            'success_indicators': [],
            'behavioral_metrics': [],
            'learning_curve': []
        }
        
        # Custom callback for comprehensive metrics
        class MetricsCallback:
            def __init__(self, metrics_dict):
                self.metrics = metrics_dict
                self.episode_count = 0
                self.start_time = time.time()
                self.first_success_episode = None
                self.first_success_step = None
            
            def __call__(self, locals, globals):
                info = locals.get('infos', [{}])[0] if isinstance(locals.get('infos'), list) else locals.get('infos', {})
                
                if 'episode' in info:
                    episode = info['episode']['r'] if isinstance(info['episode'], dict) else info['episode']
                    reward = info['episode']['r'] if isinstance(info['episode'], dict) else 0
                    length = info['episode']['l'] if isinstance(info['episode'], dict) else 0
                    
                    self.episode_count += 1
                    
                    # Record episode metrics
                    self.metrics['episodes'].append(self.episode_count)
                    self.metrics['rewards'].append(reward)
                    self.metrics['learning_curve'].append({
                        'episode': self.episode_count,
                        'reward': reward,
                        'length': length,
                        'timestamp': time.time() - self.start_time
                    })
                    
                    # Check for first success
                    if self.first_success_episode is None and reward > 1000:  # Level completion threshold
                        self.first_success_episode = self.episode_count
                        self.first_success_step = locals.get('num_timesteps', 0)
        
        metrics_callback = MetricsCallback(training_metrics)
        
        # Train the agent
        start_time = time.time()
        agent.learn(
            total_timesteps=self.config['training']['total_timesteps'],
            callback=metrics_callback,
            progress_bar=False
        )
        training_time = time.time() - start_time
        
        # Final evaluation
        final_metrics = self.evaluate_agent(agent, env, method_name, run_id)
        
        # Compile results
        results = {
            'method': method_name,
            'seed': seed,
            'run_id': run_id,
            'training_time': training_time,
            'first_success_episode': metrics_callback.first_success_episode,
            'first_success_step': metrics_callback.first_success_step,
            'final_metrics': final_metrics,
            'training_metrics': training_metrics,
            'learning_curve': training_metrics['learning_curve']
        }
        
        # Save individual run results
        run_file = self.results_dir / f"{method_name}_run_{run_id}_seed_{seed}.json"
        with open(run_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        env.close()
        return results
    
    def evaluate_agent(self, agent, env, method_name: str, run_id: int) -> Dict[str, Any]:
        """Comprehensive evaluation of trained agent."""
        self.logger.info(f"Evaluating {method_name} - Run: {run_id}")
        
        eval_episodes = 10
        eval_results = {
            'scores': [],
            'completion_times': [],
            'success_rate': 0,
            'average_reward': 0,
            'behavioral_diversity': 0,
            'exploration_coverage': 0
        }
        
        for episode in range(eval_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs, _ = obs
            
            episode_reward = 0
            episode_length = 0
            actions_taken = set()
            positions_visited = set()
            
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                actions_taken.add(action)
                
                result = env.step(action)
                if len(result) == 5:
                    obs, reward, done, truncated, info = result
                    done = done or truncated
                else:
                    obs, reward, done, info = result
                
                episode_reward += reward
                episode_length += 1
                
                # Track exploration
                if 'position' in info:
                    pos = info['position']
                    if isinstance(pos, (list, tuple)):
                        positions_visited.add((pos[0] // 50, pos[1] // 50))
            
            eval_results['scores'].append(episode_reward)
            eval_results['completion_times'].append(episode_length)
            
            # Success indicator (reward > 1000 indicates level completion)
            if episode_reward > 1000:
                eval_results['success_rate'] += 1
        
        # Calculate final metrics
        eval_results['success_rate'] /= eval_episodes
        eval_results['average_reward'] = np.mean(eval_results['scores'])
        eval_results['behavioral_diversity'] = len(actions_taken) / 14  # Normalize by action space size
        eval_results['exploration_coverage'] = len(positions_visited) / 100  # Normalize by expected coverage
        
        return eval_results
    
    def run_comparison(self) -> Dict[str, Any]:
        """Run the complete scholarly comparison."""
        self.logger.info("Starting scholarly comparison experiment")
        
        # Run multiple trials for statistical significance
        for run_id in range(self.sample_size):
            seed = self.random_seeds[run_id]
            
            # Train Traditional RL
            traditional_result = self.train_method("traditional_rl", seed, run_id)
            self.traditional_rl_results.append(traditional_result)
            
            # Train Shaping RL
            shaping_result = self.train_method("shaping_rl", seed, run_id)
            self.shaping_rl_results.append(shaping_result)
            
            self.logger.info(f"Completed run {run_id + 1}/{self.sample_size}")
        
        # Perform statistical analysis
        analysis_results = self.perform_statistical_analysis()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Save comprehensive results
        self.save_results(analysis_results)
        
        return analysis_results
    
    def perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        self.logger.info("Performing statistical analysis")
        
        # Extract key metrics for comparison
        traditional_metrics = {
            'first_success_episodes': [r['first_success_episode'] for r in self.traditional_rl_results if r['first_success_episode'] is not None],
            'final_scores': [r['final_metrics']['average_reward'] for r in self.traditional_rl_results],
            'success_rates': [r['final_metrics']['success_rate'] for r in self.traditional_rl_results],
            'behavioral_diversity': [r['final_metrics']['behavioral_diversity'] for r in self.traditional_rl_results]
        }
        
        shaping_metrics = {
            'first_success_episodes': [r['first_success_episode'] for r in self.shaping_rl_results if r['first_success_episode'] is not None],
            'final_scores': [r['final_metrics']['average_reward'] for r in self.shaping_rl_results],
            'success_rates': [r['final_metrics']['success_rate'] for r in self.shaping_rl_results],
            'behavioral_diversity': [r['final_metrics']['behavioral_diversity'] for r in self.shaping_rl_results]
        }
        
        # Statistical tests
        analysis = {}
        
        # Learning Speed (episodes to first success)
        if traditional_metrics['first_success_episodes'] and shaping_metrics['first_success_episodes']:
            t_stat, p_value = stats.ttest_ind(
                traditional_metrics['first_success_episodes'],
                shaping_metrics['first_success_episodes']
            )
            analysis['learning_speed'] = {
                'traditional_mean': np.mean(traditional_metrics['first_success_episodes']),
                'shaping_mean': np.mean(shaping_metrics['first_success_episodes']),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < self.significance_threshold,
                'effect_size': self.calculate_cohens_d(
                    traditional_metrics['first_success_episodes'],
                    shaping_metrics['first_success_episodes']
                )
            }
        
        # Final Performance (scores)
        t_stat, p_value = stats.ttest_ind(
            traditional_metrics['final_scores'],
            shaping_metrics['final_scores']
        )
        analysis['final_performance'] = {
            'traditional_mean': np.mean(traditional_metrics['final_scores']),
            'shaping_mean': np.mean(shaping_metrics['final_scores']),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.significance_threshold,
            'effect_size': self.calculate_cohens_d(
                traditional_metrics['final_scores'],
                shaping_metrics['final_scores']
            )
        }
        
        # Success Rate
        t_stat, p_value = stats.ttest_ind(
            traditional_metrics['success_rates'],
            shaping_metrics['success_rates']
        )
        analysis['success_rate'] = {
            'traditional_mean': np.mean(traditional_metrics['success_rates']),
            'shaping_mean': np.mean(shaping_metrics['success_rates']),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.significance_threshold,
            'effect_size': self.calculate_cohens_d(
                traditional_metrics['success_rates'],
                shaping_metrics['success_rates']
            )
        }
        
        # Behavioral Diversity
        t_stat, p_value = stats.ttest_ind(
            traditional_metrics['behavioral_diversity'],
            shaping_metrics['behavioral_diversity']
        )
        analysis['behavioral_diversity'] = {
            'traditional_mean': np.mean(traditional_metrics['behavioral_diversity']),
            'shaping_mean': np.mean(shaping_metrics['behavioral_diversity']),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.significance_threshold,
            'effect_size': self.calculate_cohens_d(
                traditional_metrics['behavioral_diversity'],
                shaping_metrics['behavioral_diversity']
            )
        }
        
        return analysis
    
    def calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        return (np.mean(group2) - np.mean(group1)) / pooled_std
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations for the comparison."""
        self.logger.info("Generating visualizations")
        
        # Create visualization directory
        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Learning Curves Comparison
        self.plot_learning_curves(viz_dir)
        
        # 2. Performance Comparison
        self.plot_performance_comparison(viz_dir)
        
        # 3. Statistical Analysis Results
        self.plot_statistical_results(viz_dir)
    
    def plot_learning_curves(self, viz_dir: Path):
        """Plot learning curves for both methods."""
        plt.figure(figsize=(12, 8))
        
        # Traditional RL learning curves
        for i, result in enumerate(self.traditional_rl_results):
            episodes = [m['episode'] for m in result['learning_curve']]
            rewards = [m['reward'] for m in result['learning_curve']]
            plt.plot(episodes, rewards, 'b-', alpha=0.3, label='Traditional RL' if i == 0 else "")
        
        # Shaping RL learning curves
        for i, result in enumerate(self.shaping_rl_results):
            episodes = [m['episode'] for m in result['learning_curve']]
            rewards = [m['reward'] for m in result['learning_curve']]
            plt.plot(episodes, rewards, 'r-', alpha=0.3, label='Shaping RL' if i == 0 else "")
        
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.title('Learning Curves: Traditional RL vs Shaping RL')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(viz_dir / "learning_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_comparison(self, viz_dir: Path):
        """Plot performance comparison between methods."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract metrics
        traditional_scores = [r['final_metrics']['average_reward'] for r in self.traditional_rl_results]
        shaping_scores = [r['final_metrics']['average_reward'] for r in self.shaping_rl_results]
        
        traditional_success = [r['final_metrics']['success_rate'] for r in self.traditional_rl_results]
        shaping_success = [r['final_metrics']['success_rate'] for r in self.shaping_rl_results]
        
        # Final Scores
        axes[0, 0].boxplot([traditional_scores, shaping_scores], labels=['Traditional RL', 'Shaping RL'])
        axes[0, 0].set_title('Final Performance (Average Reward)')
        axes[0, 0].set_ylabel('Average Reward')
        
        # Success Rates
        axes[0, 1].boxplot([traditional_success, shaping_success], labels=['Traditional RL', 'Shaping RL'])
        axes[0, 1].set_title('Success Rate')
        axes[0, 1].set_ylabel('Success Rate')
        
        # Learning Speed
        traditional_speed = [r['first_success_episode'] for r in self.traditional_rl_results if r['first_success_episode'] is not None]
        shaping_speed = [r['first_success_episode'] for r in self.shaping_rl_results if r['first_success_episode'] is not None]
        
        if traditional_speed and shaping_speed:
            axes[1, 0].boxplot([traditional_speed, shaping_speed], labels=['Traditional RL', 'Shaping RL'])
            axes[1, 0].set_title('Learning Speed (Episodes to First Success)')
            axes[1, 0].set_ylabel('Episodes')
        
        # Behavioral Diversity
        traditional_diversity = [r['final_metrics']['behavioral_diversity'] for r in self.traditional_rl_results]
        shaping_diversity = [r['final_metrics']['behavioral_diversity'] for r in self.shaping_rl_results]
        
        axes[1, 1].boxplot([traditional_diversity, shaping_diversity], labels=['Traditional RL', 'Shaping RL'])
        axes[1, 1].set_title('Behavioral Diversity')
        axes[1, 1].set_ylabel('Diversity Score')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_statistical_results(self, viz_dir: Path):
        """Plot statistical analysis results."""
        # This would be implemented based on the statistical analysis results
        pass
    
    def save_results(self, analysis_results: Dict[str, Any]):
        """Save comprehensive results to files."""
        self.logger.info("Saving comprehensive results")
        
        # Save analysis results
        analysis_file = self.results_dir / "statistical_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Save summary report
        self.generate_summary_report(analysis_results)
        
        # Save raw data
        raw_data_file = self.results_dir / "raw_data.json"
        with open(raw_data_file, 'w') as f:
            json.dump({
                'traditional_rl_results': self.traditional_rl_results,
                'shaping_rl_results': self.shaping_rl_results
            }, f, indent=2, default=str)
    
    def generate_summary_report(self, analysis_results: Dict[str, Any]):
        """Generate a comprehensive summary report."""
        report_file = self.results_dir / "summary_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Scholarly Comparison: Traditional RL vs. Shaping-Based RL\n\n")
            f.write(f"**Experiment Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Sample Size**: {self.sample_size} runs per method\n")
            f.write(f"**Confidence Level**: {self.confidence_level}\n")
            f.write(f"**Significance Threshold**: {self.significance_threshold}\n\n")
            
            f.write("## Research Hypothesis\n\n")
            f.write(f"**Hypothesis**: {self.config['experiment']['hypothesis']}\n\n")
            
            f.write("## Results Summary\n\n")
            
            for metric, results in analysis_results.items():
                f.write(f"### {metric.replace('_', ' ').title()}\n\n")
                f.write(f"- **Traditional RL Mean**: {results['traditional_mean']:.3f}\n")
                f.write(f"- **Shaping RL Mean**: {results['shaping_mean']:.3f}\n")
                f.write(f"- **T-Statistic**: {results['t_statistic']:.3f}\n")
                f.write(f"- **P-Value**: {results['p_value']:.4f}\n")
                f.write(f"- **Statistically Significant**: {'Yes' if results['significant'] else 'No'}\n")
                f.write(f"- **Effect Size (Cohen's d)**: {results['effect_size']:.3f}\n\n")
            
            f.write("## Conclusion\n\n")
            # Add conclusion based on results
            f.write("Detailed conclusions will be added based on the statistical analysis results.\n")

def main():
    """Main function for scholarly comparison."""
    parser = argparse.ArgumentParser(description="Scholarly Comparison: Traditional RL vs. Shaping-Based RL")
    parser.add_argument("--config", type=str, default="configs/scholarly_comparison_config.yaml",
                       help="Path to scholarly comparison configuration file")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Override sample size from config")
    
    args = parser.parse_args()
    
    # Initialize comparison framework
    comparison = ScholarlyComparison(args.config)
    
    # Override sample size if specified
    if args.sample_size:
        comparison.sample_size = args.sample_size
        comparison.random_seeds = comparison.random_seeds[:args.sample_size]
    
    # Run the comparison
    results = comparison.run_comparison()
    
    print(f"\n🎓 Scholarly Comparison Complete!")
    print(f"📊 Results saved to: {comparison.results_dir}")
    print(f"📈 Visualizations: {comparison.results_dir}/visualizations/")
    print(f"📋 Summary Report: {comparison.results_dir}/summary_report.md")
    
    # Print key findings
    print(f"\n🔬 Key Findings:")
    for metric, analysis in results.items():
        if analysis['significant']:
            print(f"✅ {metric.replace('_', ' ').title()}: Statistically significant difference")
            print(f"   Traditional RL: {analysis['traditional_mean']:.2f}")
            print(f"   Shaping RL: {analysis['shaping_mean']:.2f}")
            print(f"   Effect Size: {analysis['effect_size']:.2f}")

if __name__ == "__main__":
    main()
