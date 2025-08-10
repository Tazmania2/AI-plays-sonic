# Direct Input System Guide

## Overview

The Direct Input System provides Windows API-based input injection as the primary method for AI control of BizHawk, with file-based communication as a secondary fallback. This system addresses the user's requirement for direct AI input commands while maintaining reliability through automatic fallback mechanisms.

## Architecture

### Primary Method: Direct Input Injection
- **Technology**: Windows API (win32gui, win32con)
- **Method**: Direct window message posting to BizHawk windows
- **Advantages**: 
  - No file I/O overhead
  - Lower latency
  - Direct control without intermediate layers
  - Real-time input injection

### Secondary Method: File-Based Communication
- **Technology**: Lua bridge with file-based communication
- **Method**: Writing commands to files that BizHawk reads via Lua script
- **Advantages**:
  - Reliable fallback when direct input fails
  - Works even when window focus issues occur
  - Proven stability from previous implementation

### Automatic Switching
The system automatically switches between input methods based on:
- Window availability detection
- Input success/failure tracking
- Error handling and recovery

## Components

### 1. DirectInputManager
**Location**: `src/utils/direct_input_manager.py`

Core class that handles Windows API-based input injection:
- Window detection and targeting
- Input queuing and processing
- Thread-safe operation
- Multi-instance support

```python
from utils.direct_input_manager import get_direct_input_manager

# Create input manager
input_manager = get_direct_input_manager(num_instances=4)

# Send input to specific environment
input_manager.send_action(env_id, 'RIGHT', duration=0.016)
```

### 2. DirectInputSonicEnvironment
**Location**: `src/environment/direct_input_env.py`

Gym environment that uses direct input as primary method:
- Automatic input method switching
- Fallback to file-based communication
- Standard gym interface
- Input method tracking and reporting

```python
from environment.direct_input_env import DirectInputSonicEnvironment

# Create environment
env = DirectInputSonicEnvironment(config, env_id=0)

# Use standard gym interface
obs, reward, done, truncated, info = env.step(action)
input_method = info.get('input_method')  # 'direct' or 'file'
```

### 3. MultiInstanceDirectInputManager
**Location**: `src/utils/direct_input_manager.py`

Manages multiple input instances for parallel training:
- Environment-to-instance mapping
- Instance status monitoring
- Coordinated shutdown

## Usage

### Command Line Usage

#### 1. Using Direct Input (Primary)
```bash
python play_sonic.py --model path/to/model --direct-input
```

#### 2. Using File-Based Input (Secondary)
```bash
python play_sonic.py --model path/to/model --file-based
```

#### 3. Using Standard Environment (Legacy)
```bash
python play_sonic.py --model path/to/model
```

### Programmatic Usage

#### 1. Creating Direct Input Environment
```python
import yaml
from environment.direct_input_env import DirectInputSonicEnvironment

# Load configuration
with open('configs/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create environment
env = DirectInputSonicEnvironment(config, env_id=0)

# Use standard gym interface
obs, info = env.reset()
obs, reward, done, truncated, info = env.step(action)

# Check input method
input_method = info.get('input_method')  # 'direct' or 'file'
```

#### 2. Using Direct Input Manager Directly
```python
from utils.direct_input_manager import get_direct_input_manager

# Create input manager
input_manager = get_direct_input_manager(num_instances=4)

# Assign environment to instance
input_manager.assign_environment(env_id=0, instance_id=0)

# Send inputs
input_manager.send_action(env_id=0, action='RIGHT', duration=0.016)
input_manager.send_actions(env_id=0, actions=['RIGHT', 'A'], duration=0.016)

# Check status
status = input_manager.get_instance_status()
```

## Input Mapping

The system maps actions to Windows virtual key codes:

| Action | Virtual Key | Description |
|--------|-------------|-------------|
| LEFT | VK_LEFT | Move left |
| RIGHT | VK_RIGHT | Move right |
| UP | VK_UP | Move up |
| DOWN | VK_DOWN | Move down |
| A | VK_Z | Jump (Genesis A button) |
| B | VK_X | Spin dash (Genesis B button) |
| C | VK_C | Genesis C button |
| START | VK_RETURN | Start button |
| SELECT | VK_SPACE | Select button |
| SAVE | VK_F5 | Save state |
| LOAD | VK_F7 | Load state |
| RESET | VK_F1 | Reset game |

## Configuration

### Environment Configuration
The direct input environment uses the same configuration as other environments:

```yaml
# configs/training_config.yaml
game:
  rom_path: "roms/sonic1.md"
  max_steps: 10000
  frame_skip: 4

environment:
  # Standard environment settings
  # Direct input settings are handled automatically

actions:
  basic: ["LEFT", "RIGHT", "UP", "DOWN", "A", "B", "C", "START"]
  combinations: [["LEFT", "A"], ["RIGHT", "A"], ["DOWN", "B"]]
```

### Input Manager Configuration
The direct input manager can be configured programmatically:

```python
# Number of instances for parallel training
input_manager = get_direct_input_manager(num_instances=4)

# Instance assignment
input_manager.assign_environment(env_id=0, instance_id=0)
input_manager.assign_environment(env_id=1, instance_id=1)
```

## Testing

### Run Complete Test Suite
```bash
python test_direct_input.py
```

### Run Demonstration
```bash
python demo_direct_input.py
```

### Test Individual Components
```python
# Test direct input manager
from utils.direct_input_manager import get_direct_input_manager
input_manager = get_direct_input_manager(num_instances=1)
status = input_manager.get_instance_status()
print(f"Instance status: {status}")

# Test environment
from environment.direct_input_env import DirectInputSonicEnvironment
env = DirectInputSonicEnvironment(config, env_id=0)
obs, info = env.reset()
print(f"Input method: {info.get('input_method')}")
```

## Troubleshooting

### Issue: "No BizHawk window found"
**Cause**: BizHawk is not running or window detection failed
**Solution**: 
- Ensure BizHawk is running and visible
- Check that BizHawk process name is correct
- Verify window titles contain expected keywords

### Issue: Direct input failures
**Cause**: Window focus issues or API limitations
**Solution**:
- The system automatically switches to file-based input
- Check window focus and visibility
- Ensure BizHawk window is not minimized

### Issue: Input not being received
**Cause**: Multiple possible causes
**Solutions**:
1. Check if system switched to file-based input
2. Verify BizHawk is in focus
3. Check Lua script is loaded correctly
4. Monitor input method in environment info

### Issue: Performance problems
**Cause**: Too many instances or resource contention
**Solutions**:
- Reduce number of instances: `num_instances=2`
- Use single environment mode
- Monitor CPU/GPU usage

## Performance Characteristics

### Direct Input (Primary)
- **Latency**: ~1-5ms
- **CPU Usage**: Low
- **Reliability**: High (with automatic fallback)
- **Scalability**: Good (multi-instance support)

### File-Based Input (Fallback)
- **Latency**: ~10-50ms
- **CPU Usage**: Medium
- **Reliability**: Very High
- **Scalability**: Good

## Migration from Previous Systems

### From File-Based Only
```python
# Old way
from environment.file_based_env import FileBasedSonicEnvironment
env = FileBasedSonicEnvironment(config, instance_id=0)

# New way
from environment.direct_input_env import DirectInputSonicEnvironment
env = DirectInputSonicEnvironment(config, env_id=0)
```

### From Standard Environment
```python
# Old way
from environment.sonic_env import SonicEnvironment
env = SonicEnvironment(config)

# New way (with direct input)
from environment.direct_input_env import DirectInputSonicEnvironment
env = DirectInputSonicEnvironment(config, env_id=0)
```

## Advanced Features

### Custom Input Mapping
```python
# Modify action mapping in DirectInputManager
class CustomDirectInputManager(DirectInputManager):
    def __init__(self, instance_id: int = 0):
        super().__init__(instance_id)
        # Custom key mappings
        self.action_to_vk['CUSTOM_ACTION'] = VK_SPACE
```

### Input Timing Control
```python
# Precise input timing
input_manager.send_action(env_id, 'RIGHT', duration=0.016)  # 60 FPS
input_manager.send_action(env_id, 'A', duration=0.1)        # Hold for 100ms
```

### Window Management
```python
# Refresh window detection
instance = input_manager.input_managers[0]
instance.refresh_window_target()

# Check window status
is_active = instance.is_window_active()
```

## Best Practices

1. **Always use DirectInputSonicEnvironment** for new projects
2. **Monitor input method** in environment info
3. **Handle automatic switching** gracefully
4. **Use appropriate instance counts** for your hardware
5. **Test with both input methods** during development
6. **Monitor performance** and adjust accordingly

## Future Enhancements

- Cross-platform support (Linux/macOS)
- Enhanced window detection algorithms
- Input recording and playback
- Advanced input timing controls
- Integration with other emulators
