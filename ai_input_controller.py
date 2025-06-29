#!/usr/bin/env python3
"""
AI Input Controller
Manages the file-based input system for AI training.
"""

import os
import time
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import random

class AIInputController:
    def __init__(self, 
                 bizhawk_dir: str = r"C:\Program Files (x86)\BizHawk-2.10-win-x64",
                 lua_script: str = "emulator/input_player.lua",
                 rom_path: str = "roms/Sonic The Hedgehog (USA, Europe).md",
                 working_dir: str = "ai_training"):
        self.bizhawk_dir = bizhawk_dir
        self.lua_script = lua_script
        self.rom_path = rom_path
        self.working_dir = Path(working_dir)
        self.process = None
        
        # File paths
        self.input_file = self.working_dir / "ai_inputs.txt"
        self.log_file = self.working_dir / "game_log.txt"
        self.completion_file = self.working_dir / "execution_complete.txt"
        
        # Create working directory
        self.working_dir.mkdir(exist_ok=True)
        
        print(f"[AIController] Using working directory: {self.working_dir}")
    
    def start_emulator(self):
        """Start BizHawk with the input player Lua script."""
        # Convert relative paths to absolute paths
        from pathlib import Path
        
        # Get absolute paths
        current_dir = Path.cwd()
        lua_script_abs = str(current_dir / self.lua_script)
        rom_path_abs = str(current_dir / self.rom_path)
        
        # Create a custom config directory to avoid permission issues
        config_dir = self.working_dir / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Set environment variable to use custom config
        env = os.environ.copy()
        env['BIZHAWK_CONFIG_PATH'] = str(config_dir)
        
        cmd = [
            os.path.join(self.bizhawk_dir, "EmuHawk.exe"),
            f"--lua={lua_script_abs}",
            rom_path_abs
        ]
        
        print(f"[AIController] Starting BizHawk: {' '.join(cmd)}")
        print(f"[AIController] Using config directory: {config_dir}")
        
        # Launch from the working directory (for input/output files) but use absolute paths
        self.process = subprocess.Popen(cmd, cwd=self.working_dir, env=env)
        
        # Wait for emulator to start
        time.sleep(5)
        print("[AIController] BizHawk started")
    
    def create_input_sequence(self, actions: List[str], max_frames: int = 300) -> List[str]:
        """Create an input sequence for the AI to execute."""
        input_sequence = []
        current_frame = 0
        
        for action in actions:
            # Add some randomness to frame timing
            frame_delay = random.randint(5, 15)
            current_frame += frame_delay
            
            if current_frame > max_frames:
                break
            
            # Format: "FRAME:ACTION"
            input_sequence.append(f"{current_frame}:{action}")
        
        return input_sequence
    
    def write_input_file(self, actions: List[str]):
        """Write actions to the input file."""
        input_sequence = self.create_input_sequence(actions)
        
        with open(self.input_file, 'w') as f:
            for line in input_sequence:
                f.write(line + '\n')
        
        print(f"[AIController] Wrote {len(input_sequence)} inputs to {self.input_file}")
    
    def wait_for_completion(self, timeout: int = 60) -> bool:
        """Wait for input execution to complete."""
        print("[AIController] Waiting for input execution...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.completion_file.exists():
                with open(self.completion_file, 'r') as f:
                    completion_info = f.read().strip()
                print(f"[AIController] Execution complete: {completion_info}")
                
                # Clean up completion file
                self.completion_file.unlink()
                return True
            
            time.sleep(0.1)
        
        print("[AIController] Timeout waiting for completion")
        return False
    
    def read_game_log(self) -> List[Dict[str, Any]]:
        """Read and parse the game log."""
        if not self.log_file.exists():
            print("[AIController] No game log found")
            return []
        
        game_states = []
        
        with open(self.log_file, 'r') as f:
            # Skip header
            next(f)
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse log entry: FRAME:INPUT:X:Y:RINGS:LIVES:LEVEL:ACT:SCORE:TIME
                parts = line.split('|')
                if len(parts) >= 10:
                    try:
                        state = {
                            'frame': int(parts[0].split(':')[1]),
                            'input': parts[1].split(':')[1] if ':' in parts[1] else parts[1],
                            'x': int(parts[2].split(':')[1]),
                            'y': int(parts[3].split(':')[1]),
                            'rings': int(parts[4].split(':')[1]),
                            'lives': int(parts[5].split(':')[1]),
                            'level': int(parts[6].split(':')[1]),
                            'act': int(parts[7].split(':')[1]),
                            'score': int(parts[8].split(':')[1]),
                            'time': parts[9].split(':')[1]
                        }
                        game_states.append(state)
                    except (ValueError, IndexError) as e:
                        print(f"[AIController] Error parsing log line: {line} - {e}")
        
        print(f"[AIController] Read {len(game_states)} game states from log")
        return game_states
    
    def analyze_results(self, game_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the results of the input sequence."""
        if not game_states:
            return {'success': False, 'reason': 'No game states recorded'}
        
        initial_state = game_states[0]
        final_state = game_states[-1]
        
        # Calculate metrics
        distance_traveled = abs(final_state['x'] - initial_state['x'])
        rings_gained = final_state['rings'] - initial_state['rings']
        lives_lost = initial_state['lives'] - final_state['lives']
        score_gained = final_state['score'] - initial_state['score']
        
        # Determine success criteria
        success = True
        reason = "Completed successfully"
        
        if lives_lost > 0:
            success = False
            reason = f"Lost {lives_lost} lives"
        elif distance_traveled < 100:
            success = False
            reason = f"Low progress: {distance_traveled} pixels"
        
        return {
            'success': success,
            'reason': reason,
            'distance_traveled': distance_traveled,
            'rings_gained': rings_gained,
            'lives_lost': lives_lost,
            'score_gained': score_gained,
            'final_x': final_state['x'],
            'final_y': final_state['y'],
            'total_frames': len(game_states)
        }
    
    def generate_next_actions(self, results: Dict[str, Any]) -> List[str]:
        """Generate the next action sequence based on results."""
        # Simple AI logic - can be replaced with more sophisticated algorithms
        actions = []
        
        if results['success']:
            # If successful, try more aggressive actions
            actions = ['RIGHT', 'RIGHT+A', 'RIGHT', 'A', 'RIGHT+B', 'RIGHT', 'A']
        else:
            # If failed, try more conservative actions
            if results['lives_lost'] > 0:
                # Lost lives, be more careful
                actions = ['RIGHT', 'RIGHT', 'A', 'RIGHT', 'RIGHT']
            else:
                # Low progress, try different approach
                actions = ['RIGHT+A', 'RIGHT', 'A', 'RIGHT', 'B', 'RIGHT']
        
        return actions
    
    def reset_game(self):
        """Reset the game for a new episode."""
        # Write a reset command
        with open(self.input_file, 'w') as f:
            f.write("1:RESET\n")
        
        # Wait for reset
        time.sleep(2)
        
        # Clear the reset command
        if self.input_file.exists():
            self.input_file.unlink()
        
        print("[AIController] Game reset")
    
    def run_training_episode(self, episode: int, max_episodes: int = 10):
        """Run a single training episode."""
        print(f"\n[AIController] Starting episode {episode}/{max_episodes}")
        
        # Generate initial actions (or use previous results)
        if episode == 1:
            actions = ['RIGHT', 'A', 'RIGHT', 'RIGHT', 'A']
        else:
            # This would normally come from the AI model
            actions = ['RIGHT+A', 'RIGHT', 'A', 'RIGHT', 'B']
        
        # Write input file
        self.write_input_file(actions)
        
        # Wait for execution
        if not self.wait_for_completion():
            print("[AIController] Episode failed - execution timeout")
            return False
        
        # Read and analyze results
        game_states = self.read_game_log()
        results = self.analyze_results(game_states)
        
        print(f"[AIController] Episode {episode} results:")
        print(f"  Success: {results['success']}")
        print(f"  Reason: {results['reason']}")
        print(f"  Distance: {results['distance_traveled']} pixels")
        print(f"  Rings: +{results['rings_gained']}")
        print(f"  Lives: -{results['lives_lost']}")
        print(f"  Score: +{results['score_gained']}")
        
        # Generate next actions (for next episode)
        next_actions = self.generate_next_actions(results)
        print(f"[AIController] Next actions: {next_actions}")
        
        # Reset for next episode
        if episode < max_episodes:
            self.reset_game()
        
        return results['success']
    
    def run_training(self, max_episodes: int = 10):
        """Run the full training loop."""
        print(f"[AIController] Starting training with {max_episodes} episodes")
        
        # Start emulator
        self.start_emulator()
        
        try:
            successful_episodes = 0
            
            for episode in range(1, max_episodes + 1):
                success = self.run_training_episode(episode, max_episodes)
                if success:
                    successful_episodes += 1
                
                print(f"[AIController] Progress: {episode}/{max_episodes} successful")
            
            print(f"\n[AIController] Training complete!")
            print(f"  Successful episodes: {successful_episodes}/{max_episodes}")
            print(f"  Success rate: {successful_episodes/max_episodes*100:.1f}%")
            
        finally:
            # Cleanup
            if self.process:
                self.process.terminate()
                print("[AIController] Emulator closed")

def main():
    """Test the AI input controller."""
    controller = AIInputController()
    controller.run_training(max_episodes=5)

if __name__ == "__main__":
    main() 