# File-Based Sonic AI Approach

This document explains the file-based approach for training AI to play Sonic the Hedgehog, which serves as an alternative to the standard memory-based approach when input isolation issues occur.

## Overview

The file-based approach uses a different communication method between the AI and the emulator:

1. **AI writes commands to a text file**
2. **Emulator reads the file and executes commands**
3. **Emulator writes game state logs to another file**
4. **AI reads the logs and generates new commands**

This approach is more robust and can work around input isolation issues that sometimes occur with direct memory access and input injection.

## How It Works

### File-Based Communication

The system uses several files for communication:

- `bizhawk_comm_X/ai_inputs.txt` - AI writes commands here
- `bizhawk_comm_X/game_log.txt` - Emulator writes game state here
- `bizhawk_comm_X/status.txt` - Emulator status indicator
- `bizhawk_comm_X/execution_complete.txt` - Execution completion marker

Where `X` is the instance ID (0, 1, 2, etc.) for multiple parallel environments.

### Input Format

The AI writes inputs in the format:
```
FRAME:ACTION
```

Examples:
```
10:RIGHT
15:A
20:LEFT+A
25:NOOP
```

### Game State Log Format

The emulator writes game state as JSON:
```json
{"frame":10,"x":100,"y":200,"rings":5,"lives":3,"score":1000,"zone":1,"act":1,"timer":60,"invincibility":0,"status":0}
```

## Usage

### 1. Testing the File-Based Environment

First, test if the file-based approach works on your system:

```bash
python test_file_based_play.py --compare
```

This will test both the file-based and standard environments and tell you which one works.

### 2. Playing with File-Based Approach

To play Sonic using the file-based approach:

```bash
python play_sonic.py --model models/your_model.zip --file-based --instance-id 0
```

### 3. Training with File-Based Approach

To train a new model using the file-based approach:

```bash
python train_sonic_file_based.py --model-type ppo --timesteps 100000 --num-envs 1
```

### 4. Testing Existing Models

To test an existing model with the file-based approach:

```bash
python train_sonic_file_based.py --test-only --model-path models/your_model.zip --model-type ppo
```

## Command Line Options

### play_sonic.py
- `--file-based` - Use file-based environment instead of standard
- `--instance-id N` - Use instance ID N for file-based environment (default: 0)

### train_sonic_file_based.py
- `--model-type {ppo,a2c,dqn}` - Type of model to train (default: ppo)
- `--timesteps N` - Total timesteps for training (default: 100000)
- `--num-envs N` - Number of parallel environments (default: 1)
- `--output-dir PATH` - Output directory for models (default: models/file_based_training)
- `--test-only` - Only test existing model (don't train)
- `--model-path PATH` - Path to existing model for testing

### test_file_based_play.py
- `--compare` - Compare file-based and standard environments
- `--instance-id N` - Instance ID for testing (default: 0)

## Advantages of File-Based Approach

1. **Robustness**: Less prone to input isolation issues
2. **Debugging**: Easy to inspect input/output files
3. **Compatibility**: Works on systems where direct memory access fails
4. **Scalability**: Can run multiple instances with different instance IDs
5. **Transparency**: Clear separation between AI logic and emulator control

## Disadvantages of File-Based Approach

1. **Slower**: File I/O adds latency compared to direct memory access
2. **Complexity**: More moving parts (files, Lua scripts)
3. **Debugging**: Need to check multiple files for issues
4. **Synchronization**: Potential for timing issues between AI and emulator

## Troubleshooting

### Common Issues

1. **Emulator not starting**
   - Check BizHawk installation path in config
   - Ensure ROM file exists and is accessible
   - Check Lua script path

2. **Communication files not created**
   - Check working directory permissions
   - Ensure BizHawk has write access to the directory
   - Check instance ID conflicts

3. **Inputs not being executed**
   - Check input file format (FRAME:ACTION)
   - Ensure emulator is reading the correct file
   - Check for file permission issues

4. **Game state not being logged**
   - Check memory addresses in Lua script
   - Ensure emulator is in correct game state
   - Check for JSON encoding errors

### Debugging Steps

1. **Check status file**: `bizhawk_comm_X/status.txt` should contain "READY"
2. **Monitor input file**: Check if AI is writing commands correctly
3. **Check game log**: Verify emulator is logging game state
4. **Check completion file**: Should appear after each execution
5. **Monitor console output**: Both Python and Lua scripts provide debug info

## File Structure

```
project/
├── environment/
│   ├── sonic_env.py              # Standard environment
│   └── file_based_env.py         # File-based environment
├── emulator/
│   ├── bizhawk_bridge.lua        # Standard bridge
│   └── input_player.lua          # File-based input player
├── play_sonic.py                 # Play script (supports both approaches)
├── train_sonic_file_based.py     # File-based training script
├── test_file_based_play.py       # Testing script
└── bizhawk_comm_X/               # Communication directories
    ├── ai_inputs.txt
    ├── game_log.txt
    ├── status.txt
    └── execution_complete.txt
```

## Configuration

The file-based approach uses the same configuration files as the standard approach. Key settings:

```yaml
# configs/training_config.yaml
game:
  rom_path: "roms/Sonic The Hedgehog (USA, Europe).md"
  max_steps: 10000
  frame_skip: 4

emulator:
  bizhawk_dir: "C:/Program Files (x86)/BizHawk-2.10-win-x64"
  lua_script_path: "emulator/input_player.lua"  # For file-based approach

actions:
  available_actions: ["NOOP", "LEFT", "RIGHT", "UP", "DOWN", "A", "B", "C", "START"]
```

## Migration from Standard Approach

If you're having issues with the standard approach, you can easily switch to file-based:

1. **For playing**: Add `--file-based` flag to `play_sonic.py`
2. **For training**: Use `train_sonic_file_based.py` instead of `train_sonic.py`
3. **For testing**: Use `test_file_based_play.py` to verify setup

## Performance Considerations

- **Single environment**: Use `--num-envs 1` for testing
- **Multiple environments**: Use different `--instance-id` values for parallel training
- **Training speed**: File-based approach is slower, so use smaller timestep counts for initial testing
- **Memory usage**: Each instance creates its own communication directory

## Examples

### Quick Test
```bash
# Test if file-based approach works
python test_file_based_play.py --compare

# Test file-based environment specifically
python test_file_based_play.py --instance-id 0
```

### Training
```bash
# Train a PPO model with file-based approach
python train_sonic_file_based.py --model-type ppo --timesteps 50000 --num-envs 1

# Train with multiple environments
python train_sonic_file_based.py --model-type ppo --timesteps 100000 --num-envs 4
```

### Playing
```bash
# Play with file-based approach
python play_sonic.py --model models/ppo_file_based_final.zip --file-based --episodes 3

# Play with specific instance
python play_sonic.py --model models/ppo_file_based_final.zip --file-based --instance-id 1
```

This file-based approach provides a robust alternative when the standard memory-based approach encounters issues, while maintaining the same training and playing capabilities.
