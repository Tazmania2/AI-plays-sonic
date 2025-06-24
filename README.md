# Sonic AI - Reinforcement Learning Agent for Sonic the Hedgehog

An AI agent that learns to play Sonic the Hedgehog using reinforcement learning, inspired by projects like [Pokemon Red Experiments](https://github.com/PWhiddy/PokemonRedExperiments). This implementation uses comprehensive behavioral psychology principles and is optimized for RTX 2060 + i7-9750H + 40GB RAM systems.

## 🎮 Features

- **Local Training**: No cloud costs, runs entirely on your machine
- **Real-time Visualization**: Watch the AI learn and play in real-time
- **Multiple Sonic Games**: Support for Sonic 1, 2, and 3
- **Advanced RL Algorithms**: PPO, A2C, and DQN implementations
- **Progress Tracking**: TensorBoard integration for monitoring training
- **Save States**: Resume training from any point
- **Behavioral Psychology**: Comprehensive reward system using all 4 operant conditioning techniques
- **GPU Acceleration**: Optimized for NVIDIA RTX 2060 with CUDA support

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- A legally obtained Sonic ROM file (already present: `roms/sonic1.md`)
- FFmpeg (configured: `C:\Users\guard\Downloads\ffmpeg-2025-06-02-git-688f3944ce-full_build\ffmpeg-2025-06-02-git-688f3944ce-full_build\bin`)
- RetroArch (installed: `C:\RetroArch-Win64`)
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

3. **Run the AI:**
   ```bash
   python train_sonic.py
   ```

## 📁 Project Structure

```
sonic-ai/
├── agents/           # RL agent implementations
├── emulator/         # Game emulation wrapper
├── environment/      # Sonic environment definitions
├── models/          # Pre-trained models
├── roms/            # ROM files (sonic1.md already present)
├── utils/           # Utility functions
├── visualization/   # Training visualization tools
├── configs/         # Configuration files (optimized for RTX 2060)
├── logs/            # Training logs
└── requirements.txt # Python dependencies
```

## 🎯 Training

### Basic Training
```bash
python train_sonic.py --game sonic1 --agent ppo --episodes 1000
```

### Advanced Training (Recommended)
```bash
python train_sonic.py \
    --game sonic1 \
    --agent ppo \
    --episodes 10000 \
    --render \
    --eval
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
# During training, metrics are automatically logged
# Access them programmatically:
from stable_baselines3.common.logger import configure

# Configure custom logging
configure("logs/", ["stdout", "csv", "tensorboard"])
```

#### Export to CSV
```bash
# TensorBoard data can be exported to CSV for analysis
tensorboard --logdir logs/ --outdir exported_data/
```

## 🔍 Training Analysis

### Key Performance Indicators

1. **Episode Reward Trend**: Should increase over time
2. **Episode Length**: Should stabilize or decrease (efficiency)
3. **Ring Collection Rate**: Should improve over time
4. **Level Completion Rate**: Should increase with training
5. **Loss Convergence**: Training loss should decrease and stabilize

### Troubleshooting Training Issues

#### If AI Gets Stuck
- Increase `stuck_penalty` in config
- Add more exploration incentives
- Reduce `time_penalty` to encourage movement

#### If Training is Slow
- Increase `batch_size` (if GPU memory allows)
- Reduce `frame_skip` for more frequent updates
- Use multiple environments (`num_envs`)

#### If AI Doesn't Learn
- Check reward scaling
- Adjust learning rate
- Verify observation preprocessing
- Monitor entropy for exploration

## 🎯 Results

The AI learns to:
- Navigate levels efficiently
- Collect rings and power-ups
- Avoid enemies and obstacles
- Complete levels with high scores
- Use advanced moves (spin dash, homing attack)
- Explore efficiently without getting stuck

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Inspired by [Pokemon Red Experiments](https://github.com/PWhiddy/PokemonRedExperiments)
- Built with Stable Baselines 3
- Uses RetroArch for game emulation
- Behavioral psychology principles from B.F. Skinner's operant conditioning

## 🐛 Troubleshooting

### Common Issues

1. **ROM not found**: Ensure `sonic1.md` is in the `roms/` directory
2. **RetroArch errors**: Check that RetroArch is installed at `C:\RetroArch-Win64`
3. **CUDA errors**: Ensure NVIDIA drivers are up to date
4. **Memory issues**: Reduce batch size or buffer size
5. **Training not converging**: Adjust learning rate or reward function

### Performance Optimization

For your RTX 2060 system:
- **GPU Memory**: Monitor with `nvidia-smi`
- **CPU Usage**: Use Task Manager to monitor thread utilization
- **RAM Usage**: 40GB allows for large buffer sizes
- **Temperature**: Monitor GPU temperature during long training sessions

### Getting Help

- Check the [Issues](../../issues) page
- Monitor TensorBoard for training insights
- Use the test script: `python test_sonic.py`
- Try the demo: `python demo_sonic.py --mode random`

---

**Note**: This project requires legally obtained ROM files. The included `sonic1.md` should be from a game you own. 