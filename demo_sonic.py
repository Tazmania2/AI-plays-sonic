#!/usr/bin/env python3
"""
Sonic AI Demo Script

A demonstration script that simulates Sonic gameplay for testing
the AI system without requiring an actual emulator.
"""

import numpy as np
import cv2
import time
import random
from typing import Dict, Any, Tuple


class SonicSimulator:
    """
    Simple Sonic game simulator for demo purposes.
    
    This simulates basic Sonic gameplay mechanics:
    - Movement (left/right)
    - Jumping
    - Ring collection
    - Enemy avoidance
    - Level progression
    """
    
    def __init__(self, width: int = 224, height: int = 256):
        self.width = width
        self.height = height
        
        # Game state
        self.sonic_x = 50
        self.sonic_y = height - 50
        self.sonic_vel_x = 0
        self.sonic_vel_y = 0
        self.on_ground = True
        
        # Game objects
        self.rings = []
        self.enemies = []
        self.platforms = []
        self.score = 0
        self.rings_collected = 0
        self.lives = 3
        self.level_progress = 0
        
        # Game parameters
        self.gravity = 0.8
        self.jump_power = -15
        self.move_speed = 3
        self.max_speed = 8
        
        # Initialize game objects
        self._generate_level()
        
        # Action mapping
        self.action_mappings = {
            0: "NOOP",
            1: "LEFT",
            2: "RIGHT", 
            3: "UP",
            4: "DOWN",
            5: "A",  # Jump
            6: "B",  # Spin dash
            7: "START",
            8: "SELECT"
        }
        
    def _generate_level(self):
        """Generate a simple level layout."""
        # Generate platforms
        for i in range(10):
            x = i * 100 + random.randint(-20, 20)
            y = self.height - 30 - random.randint(0, 50)
            width = random.randint(60, 120)
            self.platforms.append((x, y, width))
        
        # Generate rings
        for i in range(20):
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 100)
            self.rings.append((x, y))
        
        # Generate enemies
        for i in range(5):
            x = random.randint(100, self.width - 100)
            y = self.height - 40
            self.enemies.append((x, y, 1))  # (x, y, direction)
    
    def _check_collision(self, x1, y1, w1, h1, x2, y2, w2, h2):
        """Check collision between two rectangles."""
        return (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2)
    
    def _update_physics(self):
        """Update Sonic's physics."""
        # Apply gravity
        if not self.on_ground:
            self.sonic_vel_y += self.gravity
        
        # Update position
        self.sonic_x += self.sonic_vel_x
        self.sonic_y += self.sonic_vel_y
        
        # Check platform collisions
        self.on_ground = False
        for platform_x, platform_y, platform_w in self.platforms:
            if self._check_collision(self.sonic_x, self.sonic_y, 20, 30,
                                   platform_x, platform_y, platform_w, 10):
                if self.sonic_vel_y > 0:  # Falling
                    self.sonic_y = platform_y - 30
                    self.sonic_vel_y = 0
                    self.on_ground = True
                    break
        
        # Ground collision
        if self.sonic_y > self.height - 30:
            self.sonic_y = self.height - 30
            self.sonic_vel_y = 0
            self.on_ground = True
        
        # Wall boundaries
        if self.sonic_x < 0:
            self.sonic_x = 0
            self.sonic_vel_x = 0
        elif self.sonic_x > self.width - 20:
            self.sonic_x = self.width - 20
            self.sonic_vel_x = 0
    
    def _update_rings(self):
        """Update ring collection."""
        rings_to_remove = []
        for ring_x, ring_y in self.rings:
            if self._check_collision(self.sonic_x, self.sonic_y, 20, 30,
                                   ring_x, ring_y, 10, 10):
                rings_to_remove.append((ring_x, ring_y))
                self.rings_collected += 1
                self.score += 10
        
        for ring in rings_to_remove:
            self.rings.remove(ring)
    
    def _update_enemies(self):
        """Update enemy movement and collisions."""
        for i, (enemy_x, enemy_y, direction) in enumerate(self.enemies):
            # Move enemy
            enemy_x += direction * 2
            
            # Bounce off walls
            if enemy_x < 0 or enemy_x > self.width - 20:
                direction *= -1
            
            # Check collision with Sonic
            if self._check_collision(self.sonic_x, self.sonic_y, 20, 30,
                                   enemy_x, enemy_y, 20, 20):
                if self.sonic_vel_y > 0:  # Sonic is falling (defeat enemy)
                    self.enemies[i] = (enemy_x, enemy_y, direction)
                    self.score += 5
                else:  # Sonic gets hit
                    self.lives -= 1
                    self.sonic_x = 50
                    self.sonic_y = self.height - 50
                    self.sonic_vel_x = 0
                    self.sonic_vel_y = 0
                    break
            
            self.enemies[i] = (enemy_x, enemy_y, direction)
    
    def _update_level_progress(self):
        """Update level progress based on Sonic's position."""
        self.level_progress = min(100, (self.sonic_x / self.width) * 100)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the game."""
        # Process action
        action_name = self.action_mappings.get(action, "NOOP")
        
        if action_name == "LEFT":
            self.sonic_vel_x = max(-self.max_speed, self.sonic_vel_x - self.move_speed)
        elif action_name == "RIGHT":
            self.sonic_vel_x = min(self.max_speed, self.sonic_vel_x + self.move_speed)
        elif action_name == "A" and self.on_ground:
            self.sonic_vel_y = self.jump_power
            self.on_ground = False
        elif action_name == "B":
            # Spin dash
            if self.on_ground:
                self.sonic_vel_x = self.max_speed * (1 if self.sonic_vel_x >= 0 else -1)
        
        # Update physics
        self._update_physics()
        
        # Update game objects
        self._update_rings()
        self._update_enemies()
        self._update_level_progress()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        done = self.lives <= 0 or self.level_progress >= 100
        
        # Get observation
        obs = self._get_observation()
        
        # Get info
        info = {
            'score': self.score,
            'rings': self.rings_collected,
            'lives': self.lives,
            'level_progress': self.level_progress,
            'position': (self.sonic_x, self.sonic_y),
            'velocity': (self.sonic_vel_x, self.sonic_vel_y)
        }
        
        return obs, reward, done, info
    
    def _calculate_reward(self) -> float:
        """Calculate reward for the current state."""
        reward = 0
        
        # Ring collection reward
        reward += self.rings_collected * 10
        
        # Progress reward
        reward += self.level_progress * 0.1
        
        # Speed reward
        reward += abs(self.sonic_vel_x) * 0.1
        
        # Survival reward
        reward += 0.1
        
        # Penalty for losing lives
        if self.lives < 3:
            reward -= 50
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get the current game observation as an image."""
        # Create blank image
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw background (sky blue)
        img[:, :] = [135, 206, 235]
        
        # Draw platforms (green)
        for platform_x, platform_y, platform_w in self.platforms:
            cv2.rectangle(img, (int(platform_x), int(platform_y)), 
                         (int(platform_x + platform_w), int(platform_y + 10)), 
                         (34, 139, 34), -1)
        
        # Draw rings (yellow)
        for ring_x, ring_y in self.rings:
            cv2.circle(img, (int(ring_x), int(ring_y)), 5, (0, 255, 255), -1)
        
        # Draw enemies (red)
        for enemy_x, enemy_y, _ in self.enemies:
            cv2.rectangle(img, (int(enemy_x), int(enemy_y)), 
                         (int(enemy_x + 20), int(enemy_y + 20)), 
                         (0, 0, 255), -1)
        
        # Draw Sonic (blue)
        cv2.rectangle(img, (int(self.sonic_x), int(self.sonic_y)), 
                     (int(self.sonic_x + 20), int(self.sonic_y + 30)), 
                     (255, 0, 0), -1)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard observation size
        obs = cv2.resize(gray, (84, 84))
        
        return obs
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the game."""
        # Reset Sonic
        self.sonic_x = 50
        self.sonic_y = self.height - 50
        self.sonic_vel_x = 0
        self.sonic_vel_y = 0
        self.on_ground = True
        
        # Reset game state
        self.score = 0
        self.rings_collected = 0
        self.lives = 3
        self.level_progress = 0
        
        # Regenerate level
        self.rings = []
        self.enemies = []
        self.platforms = []
        self._generate_level()
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs, {}
    
    def render(self):
        """Render the current game state."""
        obs = self._get_observation()
        
        # Resize for display
        display_img = cv2.resize(obs, (obs.shape[1] * 2, obs.shape[0] * 2))
        
        # Add text overlay
        cv2.putText(display_img, f"Score: {self.score}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_img, f"Rings: {self.rings_collected}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_img, f"Lives: {self.lives}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_img, f"Progress: {self.level_progress:.1f}%", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Sonic AI Demo', display_img)
        cv2.waitKey(1)


def demo_random_agent():
    """Demo with a random agent."""
    print("Sonic AI Demo - Random Agent")
    print("="*40)
    
    # Create simulator
    simulator = SonicSimulator()
    
    # Run episodes
    for episode in range(5):
        print(f"\nEpisode {episode + 1}")
        
        obs, info = simulator.reset()
        total_reward = 0
        step_count = 0
        
        while step_count < 1000:
            # Random action
            action = random.randint(0, 8)
            
            # Take step
            obs, reward, done, info = simulator.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Render
            simulator.render()
            
            # Print progress
            if step_count % 100 == 0:
                print(f"Step {step_count}: Score={info['score']}, "
                      f"Rings={info['rings']}, Progress={info['level_progress']:.1f}%")
            
            if done:
                break
            
            time.sleep(0.05)  # Slow down for visualization
        
        print(f"Episode finished! Total reward: {total_reward:.2f}")
        print(f"Final score: {info['score']}, Rings: {info['rings']}")
    
    cv2.destroyAllWindows()


def demo_manual_control():
    """Demo with manual control."""
    print("Sonic AI Demo - Manual Control")
    print("="*40)
    print("Controls:")
    print("  Arrow keys: Move")
    print("  Space: Jump")
    print("  S: Spin dash")
    print("  Q: Quit")
    print("="*40)
    
    # Create simulator
    simulator = SonicSimulator()
    
    # Action mapping for manual control
    key_to_action = {
        ord('a'): 1,  # LEFT
        ord('d'): 2,  # RIGHT
        ord('w'): 3,  # UP
        ord('s'): 4,  # DOWN
        ord(' '): 5,  # A (Jump)
        ord('s'): 6,  # B (Spin dash)
    }
    
    obs, info = simulator.reset()
    total_reward = 0
    step_count = 0
    
    while step_count < 2000:
        # Get key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        # Map key to action
        action = key_to_action.get(key, 0)  # Default to NOOP
        
        # Take step
        obs, reward, done, info = simulator.step(action)
        
        total_reward += reward
        step_count += 1
        
        # Render
        simulator.render()
        
        # Print info
        if step_count % 50 == 0:
            print(f"Step {step_count}: Score={info['score']}, "
                  f"Rings={info['rings']}, Progress={info['level_progress']:.1f}%")
        
        if done:
            print(f"Game over! Final score: {info['score']}")
            break
    
    cv2.destroyAllWindows()


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sonic AI Demo")
    parser.add_argument("--mode", type=str, choices=["random", "manual"], 
                       default="random", help="Demo mode")
    
    args = parser.parse_args()
    
    if args.mode == "random":
        demo_random_agent()
    elif args.mode == "manual":
        demo_manual_control()
    else:
        print("Invalid mode. Use 'random' or 'manual'")


if __name__ == "__main__":
    main() 