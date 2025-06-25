import time
from environment.sonic_env import SonicEnvironment
import yaml

# Load config from YAML
with open("configs/training_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create the environment with config
env = SonicEnvironment(config)
obs = env.reset()
done = False

print("Logging game state to game_state_log.jsonl. Play the game and watch the log file grow! Press Ctrl+C to stop.")

try:
    while True:
        # Log and print the current game state
        state = env.emulator.get_game_state()
        print(state)
        time.sleep(0.2)  # Slow down for readability
except KeyboardInterrupt:
    print("\nLogging stopped by user.")
finally:
    env.close() 