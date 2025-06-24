# Sonic AI Training Guide - Complete Behavioral Psychology Implementation

## 🧠 Behavioral Psychology in AI Training

This Sonic AI implementation uses **all 4 operant conditioning techniques** from B.F. Skinner's behavioral psychology to create a comprehensive learning system.

### 1. Positive Reinforcement (Adding Good Stimulus)
**Definition**: Adding a pleasant stimulus to increase desired behavior.

**Implementation**:
- **Ring Collection**: +15.0 reward
  - *Why*: Encourages the AI to seek out and collect rings
  - *Behavior*: AI learns to navigate toward ring clusters
- **Enemy Defeat**: +8.0 reward  
  - *Why*: Rewards combat skills and obstacle clearing
  - *Behavior*: AI learns to attack enemies strategically
- **Power-ups**: +20.0 reward
  - *Why*: Encourages exploration and item collection
  - *Behavior*: AI seeks out power-up boxes and monitors
- **Level Completion**: +1500.0 reward
  - *Why*: Major goal achievement reward
  - *Behavior*: Drives overall level progression
- **Checkpoints**: +100.0 reward
  - *Why*: Intermediate goal achievement
  - *Behavior*: Encourages steady progress through level

### 2. Positive Punishment (Adding Bad Stimulus)
**Definition**: Adding an unpleasant stimulus to decrease undesired behavior.

**Implementation**:
- **Game Over**: -200.0 penalty
  - *Why*: Severe penalty for complete failure
  - *Behavior*: AI learns to avoid life-threatening situations
- **Falling**: -15.0 penalty
  - *Why*: Penalty for poor platforming
  - *Behavior*: AI learns to navigate carefully and jump properly
- **Getting Stuck**: -2.0 penalty
  - *Why*: Prevents AI from staying in one place
  - *Behavior*: Encourages continuous movement and exploration

### 3. Negative Reinforcement (Removing Bad Stimulus)
**Definition**: Removing an unpleasant stimulus to increase desired behavior.

**Implementation**:
- **Forward Progress**: +2.0 reward
  - *Why*: Removes the "pressure" of time constraints
  - *Behavior*: AI learns to move rightward through the level
- **Speed Bonus**: +3.0 reward
  - *Why*: Removes the "penalty" of slow movement
  - *Behavior*: AI learns to maintain momentum and use speed
- **Height Bonus**: +1.0 reward
  - *Why*: Removes the "obstacle" of ground-level hazards
  - *Behavior*: AI learns to jump over enemies and obstacles

### 4. Negative Punishment (Removing Good Stimulus)
**Definition**: Removing a pleasant stimulus to decrease undesired behavior.

**Implementation**:
- **Time Penalty**: -0.2 penalty
  - *Why*: Removes the "bonus" of unlimited time
  - *Behavior*: Encourages efficiency and prevents dawdling
- **Ring Loss**: -5.0 penalty
  - *Why*: Removes the "bonus" of collected rings
  - *Behavior*: AI learns to protect rings and avoid damage

## 🎯 Advanced Behavioral Incentives

### Exploration Bonus (+1.0)
- **Purpose**: Encourages visiting new areas of the level
- **Implementation**: Tracks unique positions visited
- **Behavior**: AI learns to explore rather than take the same path

### Efficiency Bonus (+2.0)
- **Purpose**: Rewards completing levels quickly
- **Implementation**: Based on time-to-completion ratio
- **Behavior**: AI learns to optimize routes and minimize backtracking

### Skill Bonus (+5.0)
- **Purpose**: Rewards advanced Sonic moves
- **Implementation**: Detects spin dash, homing attack, etc.
- **Behavior**: AI learns to use Sonic's special abilities

## 🛑 Session Termination Conditions

The AI training session stops when any of these conditions are met:

### 1. Episode Completion (100% Progress)
- **Trigger**: Level is fully completed
- **Reward**: +1500.0 (major success)
- **Learning**: Reinforces successful strategies

### 2. Game Over (All Lives Lost)
- **Trigger**: Sonic loses all lives
- **Penalty**: -200.0 (major failure)
- **Learning**: Teaches what to avoid

### 3. Time Limit (15,000 Steps)
- **Trigger**: Maximum steps reached
- **Penalty**: -0.2 per step (cumulative)
- **Learning**: Encourages efficiency

### 4. Stuck Detection
- **Trigger**: AI remains in same position for too long
- **Penalty**: -2.0 per stuck frame
- **Learning**: Prevents infinite loops

### 5. Manual Interruption
- **Trigger**: User presses Ctrl+C
- **Action**: Saves current model and exits
- **Learning**: Allows user control over training

## 🚫 Preventing Stuck Behavior

The system uses multiple mechanisms to prevent the AI from getting stuck:

### 1. Stuck Penalty (-2.0)
- **Detection**: Monitors position changes over time
- **Threshold**: 100 frames in same position
- **Effect**: Immediate negative feedback

### 2. Exploration Bonus (+1.0)
- **Detection**: Tracks unique positions visited
- **Effect**: Positive reinforcement for movement
- **Implementation**: Grid-based position tracking

### 3. Progress Tracking
- **Detection**: Monitors X-coordinate movement
- **Effect**: Rewards forward progress
- **Implementation**: Compares current vs previous X position

### 4. Time Pressure (-0.2 per step)
- **Detection**: Small penalty for each time step
- **Effect**: Creates urgency to move forward
- **Implementation**: Cumulative time penalty

### 5. Jump Encouragement (+1.0 height bonus)
- **Detection**: Rewards vertical movement
- **Effect**: Encourages overcoming obstacles
- **Implementation**: Y-coordinate monitoring

### 6. Speed Incentives (+3.0 speed bonus)
- **Detection**: Rewards high horizontal velocity
- **Effect**: Encourages momentum and forward movement
- **Implementation**: Velocity tracking

## 📊 TensorBoard Data Extraction

### Available Metrics

#### Training Metrics
```python
# Episode-level metrics
episode_reward = ea.Scalars('train/episode_reward')
episode_length = ea.Scalars('train/episode_length')
episode_score = ea.Scalars('train/episode_score')
episode_rings = ea.Scalars('train/episode_rings')
episode_lives = ea.Scalars('train/episode_lives')

# Training metrics
training_loss = ea.Scalars('train/loss')
policy_loss = ea.Scalars('train/policy_loss')
value_loss = ea.Scalars('train/value_loss')
entropy = ea.Scalars('train/entropy')
learning_rate = ea.Scalars('train/learning_rate')
```

#### Game Performance Metrics
```python
# Position tracking
x_position = ea.Scalars('game/x_position')
y_position = ea.Scalars('game/y_position')
level_progress = ea.Scalars('game/level_progress')

# Game state
score = ea.Scalars('game/score')
rings = ea.Scalars('game/rings')
lives = ea.Scalars('game/lives')
time = ea.Scalars('game/time')
```

#### System Performance Metrics
```python
# Performance tracking
fps = ea.Scalars('system/fps')
gpu_memory = ea.Scalars('system/gpu_memory')
cpu_usage = ea.Scalars('system/cpu_usage')
training_time = ea.Scalars('system/training_time')
```

### Data Extraction Methods

#### Method 1: Direct TensorBoard Access
```python
import tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator

# Load TensorBoard data
ea = event_accumulator.EventAccumulator('logs/')
ea.Reload()

# Extract specific metrics
episode_rewards = ea.Scalars('train/episode_reward')
for event in episode_rewards:
    print(f"Step {event.step}: Reward {event.value}")
```

#### Method 2: CSV Export
```bash
# Export TensorBoard data to CSV
tensorboard --logdir logs/ --outdir exported_data/
```

#### Method 3: Programmatic Logging
```python
from stable_baselines3.common.logger import configure
import numpy as np

# Configure logging
configure("logs/", ["stdout", "csv", "tensorboard"])

# Log custom metrics
logger = configure("logs/")
logger.record("custom/ring_efficiency", rings_collected / total_rings)
logger.record("custom/speed_consistency", np.std(speeds))
logger.dump(step=current_step)
```

### Key Performance Indicators (KPIs)

#### 1. Episode Reward Trend
- **Good**: Steadily increasing over time
- **Warning**: Plateauing or decreasing
- **Action**: Adjust learning rate or reward function

#### 2. Episode Length
- **Good**: Stabilizing or decreasing (efficiency)
- **Warning**: Increasing (getting stuck)
- **Action**: Increase stuck penalties

#### 3. Ring Collection Rate
- **Good**: Improving over time
- **Warning**: Stagnant or decreasing
- **Action**: Adjust ring collection rewards

#### 4. Level Completion Rate
- **Good**: Increasing with training
- **Warning**: Low or zero completion
- **Action**: Check reward scaling and difficulty

#### 5. Loss Convergence
- **Good**: Decreasing and stabilizing
- **Warning**: Oscillating or increasing
- **Action**: Adjust learning rate or network architecture

## 🔍 Training Analysis Techniques

### 1. Learning Curve Analysis
```python
import matplotlib.pyplot as plt

# Plot learning curve
episode_rewards = [event.value for event in ea.Scalars('train/episode_reward')]
episodes = [event.step for event in ea.Scalars('train/episode_reward')]

plt.figure(figsize=(12, 6))
plt.plot(episodes, episode_rewards)
plt.title('Learning Curve - Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)
plt.show()
```

### 2. Behavior Analysis
```python
# Analyze movement patterns
x_positions = [event.value for event in ea.Scalars('game/x_position')]
y_positions = [event.value for event in ea.Scalars('game/y_position')]

# Calculate movement efficiency
movement_efficiency = []
for i in range(1, len(x_positions)):
    dx = x_positions[i] - x_positions[i-1]
    dy = y_positions[i] - y_positions[i-1]
    efficiency = dx / (abs(dx) + abs(dy)) if (abs(dx) + abs(dy)) > 0 else 0
    movement_efficiency.append(efficiency)
```

### 3. Reward Component Analysis
```python
# Analyze which rewards contribute most to learning
ring_rewards = [event.value for event in ea.Scalars('rewards/rings')]
enemy_rewards = [event.value for event in ea.Scalars('rewards/enemies')]
progress_rewards = [event.value for event in ea.Scalars('rewards/progress')]

# Calculate contribution percentages
total_rewards = [r + e + p for r, e, p in zip(ring_rewards, enemy_rewards, progress_rewards)]
ring_contribution = [r/t if t > 0 else 0 for r, t in zip(ring_rewards, total_rewards)]
```

## 🎯 Optimization Strategies

### For RTX 2060 Systems

#### GPU Optimization
- **Batch Size**: 128 (optimized for 6GB VRAM)
- **Memory Usage**: 80% (4.8GB of 6GB)
- **Network Depth**: Deeper CNN for better learning
- **Parallel Environments**: 4 environments for efficiency

#### CPU Optimization
- **Threads**: 8 (6 physical + 2 hyperthreads)
- **Frame Skip**: 4 for faster training
- **Buffer Size**: 50,000 for experience replay

#### Memory Optimization
- **RAM Usage**: 40GB allows large buffers
- **Observation Stacking**: 4 frames for temporal info
- **Normalization**: Reduces memory requirements

### Training Parameter Tuning

#### If AI Gets Stuck
```yaml
rewards:
  stuck_penalty: -5.0  # Increase penalty
  exploration_bonus: 2.0  # Increase exploration
  time_penalty: -0.1  # Reduce time pressure
```

#### If Training is Slow
```yaml
hardware:
  num_envs: 8  # Increase parallel environments
  batch_size: 256  # Increase batch size
game:
  frame_skip: 2  # Reduce frame skip
```

#### If AI Doesn't Learn
```yaml
agent:
  learning_rate: 0.0001  # Reduce learning rate
  ent_coef: 0.05  # Increase exploration
rewards:
  reward_scale: 0.5  # Scale down rewards
```

## 🎮 Expected Learning Progression

### Phase 1: Basic Movement (Episodes 1-100)
- **Goal**: Learn to move and jump
- **Metrics**: Basic forward progress, simple jumps
- **Rewards**: Forward progress, height bonus

### Phase 2: Ring Collection (Episodes 100-500)
- **Goal**: Learn to collect rings
- **Metrics**: Ring collection rate, score
- **Rewards**: Ring collection, enemy defeat

### Phase 3: Level Navigation (Episodes 500-1000)
- **Goal**: Navigate through levels efficiently
- **Metrics**: Level completion rate, time efficiency
- **Rewards**: Checkpoints, level completion

### Phase 4: Advanced Skills (Episodes 1000+)
- **Goal**: Use advanced Sonic moves
- **Metrics**: Skill usage, speed consistency
- **Rewards**: Skill bonus, efficiency bonus

## 🔧 Troubleshooting Guide

### Common Training Issues

#### Issue: AI Stays in One Place
**Symptoms**: Episode length increases, stuck penalties high
**Solutions**:
1. Increase `stuck_penalty` to -5.0
2. Add more exploration incentives
3. Reduce `time_penalty` to encourage movement
4. Check observation preprocessing

#### Issue: Training Not Converging
**Symptoms**: Episode rewards oscillate, loss doesn't decrease
**Solutions**:
1. Reduce learning rate to 0.0001
2. Increase entropy coefficient for exploration
3. Scale down rewards with `reward_scale: 0.5`
4. Check for reward function issues

#### Issue: Slow Training
**Symptoms**: Low FPS, long episode times
**Solutions**:
1. Increase `frame_skip` to 6
2. Reduce `num_envs` to 2
3. Use smaller network architecture
4. Check GPU memory usage

#### Issue: Memory Errors
**Symptoms**: CUDA out of memory errors
**Solutions**:
1. Reduce `batch_size` to 64
2. Reduce `buffer_size` to 25000
3. Use `gpu_memory_fraction: 0.6`
4. Check for memory leaks

### Performance Monitoring

#### GPU Monitoring
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check for memory leaks
watch -n 1 nvidia-smi
```

#### CPU Monitoring
```bash
# Monitor CPU usage
htop

# Check thread utilization
top -H -p <python_pid>
```

#### Memory Monitoring
```bash
# Monitor RAM usage
free -h

# Check for memory leaks
ps aux | grep python
```

## 🎯 Success Metrics

### Primary Metrics
- **Level Completion Rate**: >80% after 2000 episodes
- **Average Episode Reward**: >1000 after 1000 episodes
- **Ring Collection Efficiency**: >70% of available rings
- **Training Stability**: Loss convergence within 500 episodes

### Secondary Metrics
- **Speed Consistency**: <20% variance in movement speed
- **Exploration Coverage**: >60% of level area visited
- **Skill Usage**: >10% of episodes use advanced moves
- **Efficiency**: <300 seconds average level completion

### Long-term Goals
- **Perfect Runs**: 100% ring collection, no damage
- **Speed Runs**: Sub-2-minute level completions
- **Skill Mastery**: Consistent use of all Sonic abilities
- **Generalization**: Performance across multiple levels

---

This comprehensive training system ensures the AI learns efficiently while avoiding common pitfalls like getting stuck or developing suboptimal behaviors. The behavioral psychology approach creates a robust learning foundation that can be applied to other games and environments. 