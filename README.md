# **BUILT BY CURSOR.AI**

# Sonic AI - Reinforcement Learning Agent for Sonic the Hedgehog

An AI agent that learns to play Sonic the Hedgehog using reinforcement learning, inspired by projects like [Pokemon Red Experiments](https://github.com/PWhiddy/PokemonRedExperiments). This implementation uses comprehensive behavioral psychology principles and is optimized for RTX 2060 + i7-9750H + 40GB RAM systems.

## 🎯 **Primary Objective: Green Hill Zone Act 3 Completion**

The AI is specifically trained to:
1. **Complete Green Hill Zone Act 3** - This is the main win condition
2. **Maximize Score** - Each episode ends at Act 3 completion, then restarts to improve scores
3. **Track High Scores** - The system logs first completion and highlights each new high score

### 🏆 High Score Tracking
- **First Completion**: Special celebration when AI first reaches Green Hill Zone Act 3
- **New High Scores**: Highlighted logs when AI beats previous best scores
- **Progress Tracking**: Continuous monitoring of completion count and score improvements

## 🎮 Features

- **Local Training**: No cloud costs, runs entirely on your machine
- **Real-time Visualization**: Watch the AI learn and play in real-time
- **Multiple Sonic Games**: Support for Sonic 1, 2, and 3
- **Advanced RL Algorithms**: PPO, A2C, and DQN implementations
- **Progress Tracking**: TensorBoard integration for monitoring training
- **Save States**: Resume training from any point
- **Behavioral Psychology**: Comprehensive reward system using all 4 operant conditioning techniques
- **GPU Acceleration**: Optimized for NVIDIA RTX 2060 with CUDA support
- **🚀 Advanced Input Isolation**: Multi-instance BizHawk support with isolated inputs for 4x faster training
- **🧠 A/B Testing Framework**: Compare baseline vs shaping reward methods in parallel
- **🧹 Automatic Cleanup**: Graceful process termination and resource management
- **🎯 Focused Objective**: Complete Green Hill Zone Act 3 and maximize scores

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- A legally obtained Sonic ROM file (already present: `roms/sonic1.md`)
- BizHawk emulator (installed: `C:\Program Files (x86)\BizHawk-2.10-win-x64`)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd sonic-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the input isolation system:**
   ```bash
   python test_input_isolation.py
   ```

4. **Run the AI:**
   ```bash
   python train_sonic.py --num_envs 4 --reward_mode baseline
   ```

## 🧹 Process Management & Cleanup

### Automatic Cleanup
The system includes comprehensive process management to prevent hanging processes and resource conflicts:

- **Signal Handlers**: Graceful termination on Ctrl+C
- **Resource Cleanup**: Automatic environment and input manager shutdown
- **Process Monitoring**: Tracks and terminates child processes
- **File Bridge Cleanup**: Removes temporary communication files

### Manual Cleanup
If you need to manually clean up processes:

```bash
# Run the cleanup script
python cleanup_processes.py

# Or use the training launcher (recommended)
start_training.bat
```

### Cleanup Script Features
- **Process Termination**: Kills all Python, BizHawk, and training processes
- **Resource Monitoring**: Shows CPU, memory, and GPU usage
- **File Cleanup**: Removes temporary communication files
- **System Check**: Verifies system is ready for new training sessions

### Best Practices
1. **Always run cleanup** before starting new training sessions
2. **Use Ctrl+C** to stop training gracefully
3. **Check Task Manager** if you suspect hanging processes
4. **Use the training launcher** (`start_training.bat`) for automatic cleanup

## 🎯 Advanced Features

### Multi-Instance Training (4x Faster)
The system supports running multiple BizHawk instances simultaneously with isolated inputs:

```bash
# Train with 4 parallel environments
python train_sonic.py --num_envs 4 --reward_mode baseline

# A/B testing with 4 environments (2 baseline + 2 shaping)
python train_sonic.py --num_envs 4 --reward_mode both
```

### Input Isolation System
- **No Input Conflicts**: Each BizHawk window receives inputs independently
- **Windows API Integration**: Direct window messaging for precise control
- **Background Processing**: Smooth input delivery without blocking
- **Automatic Window Management**: Focus and visibility handling

### A/B Testing Framework
Compare different reward methods:
- **Baseline**: Standard reinforcement learning rewards
- **Shaping**: Susan Garrett's shaping method-inspired rewards
- **Parallel Testing**: Run both methods simultaneously

## 📁 Project Structure

```
sonic-ai/
├── agents/           # RL agent implementations
├── emulator/         # Game emulation wrapper
├── environment/      # Sonic environment definitions
├── models/          # Pre-trained models
├── roms/            # ROM files (sonic1.md already present)
├── utils/           # Utility functions (including input isolation)
├── visualization/   # Training visualization tools
├── configs/         # Configuration files (optimized for RTX 2060)
├── logs/            # Training logs
├── test_input_isolation.py  # Input isolation system test
├── INPUT_ISOLATION_GUIDE.md # Detailed input isolation guide
└── requirements.txt # Python dependencies
```

## 🎯 Training

### Basic Training (Single Environment)
```bash
python train_sonic.py --game sonic1 --agent ppo --episodes 1000
```

### Advanced Training (4 Environments - Recommended)
```bash
python train_sonic.py \
    --game sonic1 \
    --agent ppo \
    --num_envs 4 \
    --reward_mode baseline \
    --render
```

### A/B Testing
```bash
# Sequential testing
python train_sonic.py --num_envs 1 --reward_mode baseline
python train_sonic.py --num_envs 1 --reward_mode shaping

# Parallel testing (4 environments)
python train_sonic.py --num_envs 4 --reward_mode both
```

### Monitor Training
```bash
tensorboard --logdir logs/
```

## 🎮 Playing with Pre-trained Models

```bash
python play_sonic.py --model models/sonic1_ppo_final.pth --render
```

## 🔧 Configuration

The system is pre-configured for your RTX 2060 + i7-9750H + 40GB RAM setup:

- **GPU**: CUDA acceleration with 80% memory usage (4.8GB of 6GB)
- **CPU**: 8 threads (6 physical cores + 2 hyperthreads)
- **RAM**: 50,000 buffer size for experience replay
- **Batch Size**: 128 (optimized for RTX 2060 VRAM)
- **Network**: Deeper CNN architecture for better learning
- **Input Isolation**: 4 BizHawk instances with isolated inputs

## 🧠 Behavioral Psychology Implementation

This AI uses all 4 operant conditioning techniques from behavioral psychology:

### 1. Positive Reinforcement (Adding Good Stimulus)
- **Ring Collection**: +15.0 reward for collecting rings
- **Enemy Defeat**: +8.0 reward for defeating enemies  
- **Power-ups**: +20.0 reward for collecting power-ups
- **Level Completion**: +1500.0 reward for finishing level
- **Checkpoints**: +100.0 reward for reaching checkpoints

### 2. Positive Punishment (Adding Bad Stimulus)
- **Game Over**: -200.0 penalty for losing all lives
- **Falling**: -15.0 penalty for falling into pits
- **Getting Stuck**: -2.0 penalty for being stuck in one place

### 3. Negative Reinforcement (Removing Bad Stimulus)
- **Forward Progress**: +2.0 reward for moving right (removes time pressure)
- **Speed Bonus**: +3.0 reward for high speed (removes slow movement penalty)
- **Height Bonus**: +1.0 reward for jumping (removes ground obstacle penalty)

### 4. Negative Punishment (Removing Good Stimulus)
- **Time Penalty**: -0.2 penalty for time passing (removes time bonus)
- **Ring Loss**: -5.0 penalty for losing rings (removes ring bonus)

### Advanced Behavioral Incentives
- **Exploration**: +1.0 reward for visiting new areas
- **Efficiency**: +2.0 reward for completing level quickly
- **Skill Bonus**: +5.0 reward for advanced moves (spin dash, homing attack)

## 🛑 Session Termination Conditions

The AI stops a training session when:

1. **Episode Completion**: Level is completed (100% progress)
2. **Game Over**: All lives are lost
3. **Time Limit**: Maximum steps reached (15,000 steps)
4. **Stuck Detection**: Agent remains in same position for too long
5. **Manual Interruption**: User presses Ctrl+C

## 🚫 Preventing Stuck Behavior

To ensure the AI doesn't get stuck and continues progressing:

1. **Stuck Penalty**: -2.0 reward for staying in same position
2. **Exploration Bonus**: +1.0 reward for visiting new areas
3. **Progress Tracking**: Monitors X-coordinate movement
4. **Time Pressure**: Small time penalty encourages forward movement
5. **Jump Encouragement**: Height bonus for jumping over obstacles
6. **Speed Incentives**: Speed bonus for maintaining momentum

## 📊 TensorBoard Monitoring

TensorBoard provides comprehensive training insights:

### How to Access
```bash
tensorboard --logdir logs/
```
Then open http://localhost:6006 in your browser.

### Available Metrics

#### Training Metrics
- **Episode Reward**: Total reward per episode
- **Episode Length**: Number of steps per episode
- **Training Loss**: Neural network loss during training
- **Learning Rate**: Current learning rate value
- **Policy Loss**: Policy network loss
- **Value Loss**: Value function loss
- **Entropy**: Exploration vs exploitation balance

#### Game Performance
- **Score**: In-game score progression
- **Rings Collected**: Ring collection over time
- **Lives Remaining**: Life count during episodes
- **Level Progress**: Percentage of level completed
- **Position Tracking**: X/Y coordinates over time

#### System Performance
- **FPS**: Training speed (frames per second)
- **GPU Memory**: CUDA memory usage
- **CPU Usage**: Thread utilization
- **Training Time**: Time per episode/step

### How to Extract Data

#### From TensorBoard Logs
```python
import tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator

# Load TensorBoard data
ea = event_accumulator.EventAccumulator('logs/')
ea.Reload()

# Extract specific metrics
episode_rewards = ea.Scalars('train/episode_reward')
training_loss = ea.Scalars('train/loss')
```

#### From Training Scripts
```python
# Training data is automatically saved to:
# - logs/baseline_training_YYYYMMDD_HHMMSS.csv
# - logs/shaping_training_YYYYMMDD_HHMMSS.csv
# - logs/baseline_session_summary_YYYYMMDD_HHMMSS.json
# - logs/shaping_session_summary_YYYYMMDD_HHMMSS.json
```

## 🔬 Development & Research

### Project Evolution
This project evolved from a basic RL implementation to a sophisticated multi-instance training system:

1. **Initial Implementation**: Basic PPO agent with RetroArch integration
2. **BizHawk Migration**: Switched to BizHawk for better Windows compatibility
3. **Input Isolation System**: Developed advanced multi-instance input isolation
4. **A/B Testing Framework**: Added parallel comparison capabilities
5. **Performance Optimization**: GPU/CPU utilization improvements

### Key Technical Achievements
- **Input Isolation**: Solved the critical problem of multiple BizHawk instances competing for keyboard input
- **Windows API Integration**: Direct window messaging for precise control
- **Multi-Instance Coordination**: Seamless management of multiple emulator instances
- **Performance Scaling**: 4x training speed improvement through parallelization

### Research Contributions
- **Behavioral Psychology in RL**: Comprehensive implementation of operant conditioning
- **Multi-Instance RL Training**: Novel approach to parallel environment training
- **Input Isolation Techniques**: Advanced Windows API usage for emulator control

## 🤝 Contributing

This project demonstrates advanced techniques in:
- Reinforcement Learning
- Game AI development
- Multi-instance training systems
- Windows API integration
- Behavioral psychology in AI

Feel free to explore the codebase and adapt these techniques for your own projects!

## 📄 License

This project is for educational and research purposes. Please ensure you have legal access to any ROM files used.

## 🙏 Acknowledgments

- **BizHawk Team**: For the excellent emulator that made this project possible
- **Stable Baselines3**: For the robust RL framework
- **Susan Garrett**: For inspiring the shaping method approach
- **OpenAI Gym**: For the environment interface standards

---

*This project showcases advanced techniques in reinforcement learning, multi-instance training, and behavioral psychology-inspired AI development.* 