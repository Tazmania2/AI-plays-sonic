#!/usr/bin/env python3
"""
Setup script for Sonic AI project.
"""

import os
import sys
from pathlib import Path
import subprocess


def create_directories():
    """Create necessary project directories."""
    directories = [
        "roms",
        "models",
        "logs",
        "configs",
        "environment",
        "emulator",
        "utils",
        "visualization",
        "agents",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")


def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("Please install dependencies manually: pip install -r requirements.txt")


def check_rom_file():
    """Check if ROM file exists and provide instructions."""
    rom_files = [
        "roms/sonic1.bin",
        "roms/sonic1.md",
        "roms/sonic1.gen",
        "roms/Sonic the Hedgehog (USA, Europe).bin",
        "roms/Sonic the Hedgehog (USA, Europe).md",
        "roms/Sonic the Hedgehog (USA, Europe).gen"
    ]
    
    rom_found = False
    for rom_file in rom_files:
        if os.path.exists(rom_file):
            print(f"Found ROM file: {rom_file}")
            rom_found = True
            break
    
    if not rom_found:
        print("\n" + "="*60)
        print("ROM FILE NOT FOUND")
        print("="*60)
        print("Please place your legally obtained Sonic ROM file in the 'roms/' directory.")
        print("Supported formats: .bin, .md, .gen")
        print("Common filenames:")
        print("- sonic1.bin")
        print("- Sonic the Hedgehog (USA, Europe).bin")
        print("\nMake sure you own the game before using it for training!")
        print("="*60)


def check_emulator():
    """Check for available emulators."""
    print("\nChecking for available emulators...")
    
    emulators = {
        'BizHawk': [
            "C:\\Program Files\\BizHawk\\EmuHawk.exe",
            "C:\\Program Files (x86)\\BizHawk\\EmuHawk.exe"
        ],
        'RetroArch': [
            "C:\\Program Files\\RetroArch\\retroarch.exe",
            "C:\\Program Files (x86)\\RetroArch\\retroarch.exe"
        ]
    }
    
    found_emulators = []
    for name, paths in emulators.items():
        for path in paths:
            if os.path.exists(path):
                found_emulators.append((name, path))
                break
    
    if found_emulators:
        print("Found emulators:")
        for name, path in found_emulators:
            print(f"  - {name}: {path}")
    else:
        print("No emulators found!")
        print("Please install one of the following:")
        print("  - BizHawk: https://tasvideos.org/BizHawk")
        print("  - RetroArch: https://retroarch.com/")


def create_example_config():
    """Create an example configuration file if it doesn't exist."""
    config_path = "configs/training_config.yaml"
    
    if not os.path.exists(config_path):
        print("Creating example configuration file...")
        
        example_config = """# Sonic AI Training Configuration

# Game Settings
game:
  name: "sonic1"  # sonic1, sonic2, sonic3
  rom_path: "roms/sonic1.bin"
  frame_skip: 4  # Skip frames for faster training
  max_steps: 10000  # Maximum steps per episode
  render: false  # Render during training

# Environment Settings
environment:
  screen_width: 224
  screen_height: 256
  grayscale: true
  stack_frames: 4  # Number of frames to stack for temporal information
  normalize_obs: true
  reward_scale: 1.0

# Agent Settings
agent:
  type: "ppo"  # ppo, a2c, dqn
  learning_rate: 0.0003
  batch_size: 64
  buffer_size: 10000
  gamma: 0.99  # Discount factor
  gae_lambda: 0.95  # GAE lambda parameter
  clip_range: 0.2
  ent_coef: 0.01  # Entropy coefficient for exploration
  vf_coef: 0.5  # Value function coefficient
  max_grad_norm: 0.5

# Network Architecture
network:
  type: "cnn"  # cnn, mlp, cnn_lstm
  cnn_features: [32, 64, 64, 512]
  mlp_layers: [512, 256]
  lstm_hidden_size: 256
  activation: "relu"  # relu, tanh, leaky_relu

# Training Settings
training:
  total_timesteps: 1000000
  save_interval: 10000
  eval_interval: 5000
  log_interval: 100
  checkpoint_dir: "models/"
  log_dir: "logs/"
  
# Reward Function
rewards:
  # Basic rewards
  ring_collected: 10.0
  enemy_defeated: 5.0
  power_up_collected: 15.0
  level_completed: 1000.0
  game_over: -100.0
  
  # Movement rewards
  forward_progress: 1.0
  speed_bonus: 2.0
  height_bonus: 0.5
  
  # Penalties
  time_penalty: -0.1
  stuck_penalty: -1.0
  fall_penalty: -10.0

# Observation Processing
observation:
  resize: [84, 84]  # Resize observation to this size
  crop: [0, 0, 224, 256]  # Crop region [x, y, width, height]
  normalize: true
  frame_stack: 4

# Action Space
actions:
  # Available actions (based on Sonic controls)
  - "NOOP"
  - "LEFT"
  - "RIGHT"
  - "UP"
  - "DOWN"
  - "A"  # Jump
  - "B"  # Spin dash
  - "START"
  - "SELECT"
  
  # Action combinations
  combinations:
    - ["LEFT", "A"]  # Jump left
    - ["RIGHT", "A"]  # Jump right
    - ["DOWN", "B"]  # Spin dash
    - ["LEFT", "DOWN", "B"]  # Spin dash left
    - ["RIGHT", "DOWN", "B"]  # Spin dash right

# Logging and Visualization
logging:
  tensorboard: true
  wandb: false  # Set to true to use Weights & Biases
  save_videos: true
  video_fps: 30
  log_episode_rewards: true
  log_episode_lengths: true
  log_episode_scores: true

# Hardware Settings
hardware:
  device: "auto"  # auto, cpu, cuda
  num_envs: 1  # Number of parallel environments
  num_threads: 4  # Number of threads for data loading

# Evaluation Settings
evaluation:
  eval_freq: 10000
  n_eval_episodes: 5
  deterministic: true
  render: true
"""
        
        with open(config_path, 'w') as f:
            f.write(example_config)
        
        print(f"Created example configuration: {config_path}")


def main():
    """Main setup function."""
    print("Sonic AI Setup")
    print("="*50)
    
    # Create directories
    print("\n1. Creating project directories...")
    create_directories()
    
    # Install dependencies
    print("\n2. Installing dependencies...")
    install_dependencies()
    
    # Check ROM file
    print("\n3. Checking ROM file...")
    check_rom_file()
    
    # Check emulator
    print("\n4. Checking emulator...")
    check_emulator()
    
    # Create example config
    print("\n5. Creating example configuration...")
    create_example_config()
    
    print("\n" + "="*50)
    print("SETUP COMPLETED!")
    print("="*50)
    print("\nNext steps:")
    print("1. Add your Sonic ROM to the 'roms/' directory")
    print("2. Install an emulator (BizHawk or RetroArch)")
    print("3. Configure training parameters in 'configs/training_config.yaml'")
    print("4. Start training: python train_sonic.py")
    print("5. Play with trained model: python play_sonic.py --model models/your_model.pth")
    print("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main() 