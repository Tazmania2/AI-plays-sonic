import numpy as np
from typing import Dict, Any, Tuple


class SimplifiedRewardCalculator:
    """
    Simplified reward calculator inspired by Mario AI approaches.
    
    Key principles:
    1. Distance-based rewards (move right = good)
    2. Simple survival rewards (stay alive = good)
    3. Clear penalties (die = bad, get stuck = bad)
    4. Minimal complexity for faster learning
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Simple reward weights
        self.distance_reward = config.get('distance_reward', 1.0)
        self.survival_reward = config.get('survival_reward', 0.1)
        self.ring_reward = config.get('ring_collected', 5.0)
        
        # Simple penalties
        self.death_penalty = config.get('death_penalty', -100.0)
        self.stuck_penalty = config.get('stuck_penalty', -1.0)
        
        # Previous state for delta calculations
        self.previous_state = None
        self.previous_x = 0
        
    def calculate_reward(self, prev_state: Dict[str, Any], 
                        curr_state: Dict[str, Any]) -> float:
        """
        Calculate reward based on simple Mario AI principles.
        
        Args:
            prev_state: Previous game state
            curr_state: Current game state
            
        Returns:
            Calculated reward
        """
        if prev_state is None:
            return 0.0
        
        total_reward = 0.0
        
        # 1. Distance-based reward (Mario AI principle)
        total_reward += self._calculate_distance_reward(prev_state, curr_state)
        
        # 2. Survival reward (small constant reward for staying alive)
        total_reward += self.survival_reward
        
        # 3. Simple collection reward
        total_reward += self._calculate_ring_reward(prev_state, curr_state)
        
        # 4. Clear penalties
        total_reward += self._calculate_death_penalty(prev_state, curr_state)
        total_reward += self._calculate_stuck_penalty(prev_state, curr_state)
        
        return total_reward
    
    def _calculate_distance_reward(self, prev_state: Dict[str, Any], 
                                  curr_state: Dict[str, Any]) -> float:
        """
        Calculate reward based on horizontal progress (Mario AI style).
        
        This is the primary reward signal - moving right is good.
        """
        prev_x = prev_state.get('x', 0)
        curr_x = curr_state.get('x', 0)
        
        # Only reward forward progress (moving right)
        distance_gained = max(0, curr_x - prev_x)
        return distance_gained * self.distance_reward
    
    def _calculate_ring_reward(self, prev_state: Dict[str, Any], 
                              curr_state: Dict[str, Any]) -> float:
        """Calculate reward for collecting rings."""
        prev_rings = prev_state.get('rings', 0)
        curr_rings = curr_state.get('rings', 0)
        
        rings_collected = curr_rings - prev_rings
        return rings_collected * self.ring_reward
    
    def _calculate_death_penalty(self, prev_state: Dict[str, Any], 
                                curr_state: Dict[str, Any]) -> float:
        """Calculate penalty for dying."""
        prev_lives = prev_state.get('lives', 3)
        curr_lives = curr_state.get('lives', 3)
        
        lives_lost = prev_lives - curr_lives
        return lives_lost * self.death_penalty
    
    def _calculate_stuck_penalty(self, prev_state: Dict[str, Any], 
                                curr_state: Dict[str, Any]) -> float:
        """Calculate penalty for being stuck (not moving)."""
        prev_x = prev_state.get('x', 0)
        curr_x = curr_state.get('x', 0)
        
        # If we haven't moved forward, apply small penalty
        if curr_x <= prev_x:
            return self.stuck_penalty
        
        return 0.0
    
    def reset(self):
        """Reset the calculator state."""
        self.previous_state = None
        self.previous_x = 0


class MarioStyleRewardCalculator(SimplifiedRewardCalculator):
    """
    Mario-style reward calculator with additional Mario AI principles.
    
    Additional features:
    1. Height-based rewards (jumping is good)
    2. Speed-based rewards (moving fast is good)
    3. Time pressure (encourage efficiency)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Additional Mario-style rewards
        self.height_reward = config.get('height_reward', 0.5)
        self.speed_reward = config.get('speed_reward', 0.2)
        self.time_penalty = config.get('time_penalty', -0.01)
        
    def calculate_reward(self, prev_state: Dict[str, Any], 
                        curr_state: Dict[str, Any]) -> float:
        """Calculate reward with Mario-style additions."""
        base_reward = super().calculate_reward(prev_state, curr_state)
        
        # Add Mario-style rewards
        height_bonus = self._calculate_height_reward(prev_state, curr_state)
        speed_bonus = self._calculate_speed_reward(prev_state, curr_state)
        time_penalty = self.time_penalty  # Small constant time penalty
        
        return base_reward + height_bonus + speed_bonus + time_penalty
    
    def _calculate_height_reward(self, prev_state: Dict[str, Any], 
                                curr_state: Dict[str, Any]) -> float:
        """Reward for being at higher positions (jumping)."""
        curr_y = curr_state.get('y', 0)
        
        # Reward being higher up (jumping over obstacles)
        # Assuming lower Y values are higher positions
        height_bonus = max(0, (200 - curr_y) / 200)  # Normalize to 0-1
        return height_bonus * self.height_reward
    
    def _calculate_speed_reward(self, prev_state: Dict[str, Any], 
                               curr_state: Dict[str, Any]) -> float:
        """Reward for moving fast."""
        prev_x = prev_state.get('x', 0)
        curr_x = curr_state.get('x', 0)
        
        # Calculate speed (distance per step)
        speed = curr_x - prev_x
        
        # Reward high speed
        if speed > 2:  # Moving fast
            return self.speed_reward
        
        return 0.0
