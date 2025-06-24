import numpy as np
from typing import Dict, Any, Tuple


class RewardCalculator:
    """
    Calculates rewards for the Sonic RL agent.
    
    Handles various reward components:
    - Ring collection
    - Enemy defeat
    - Level completion
    - Movement progress
    - Survival time
    - Speed bonuses
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Reward weights
        self.ring_reward = config.get('ring_collected', 10.0)
        self.enemy_reward = config.get('enemy_defeated', 5.0)
        self.power_up_reward = config.get('power_up_collected', 15.0)
        self.level_complete_reward = config.get('level_completed', 1000.0)
        self.game_over_penalty = config.get('game_over', -100.0)
        
        # Movement rewards
        self.progress_reward = config.get('forward_progress', 1.0)
        self.speed_reward = config.get('speed_bonus', 2.0)
        self.height_reward = config.get('height_bonus', 0.5)
        
        # Penalties
        self.time_penalty = config.get('time_penalty', -0.1)
        self.stuck_penalty = config.get('stuck_penalty', -1.0)
        self.fall_penalty = config.get('fall_penalty', -10.0)
        
        # Previous state for delta calculations
        self.previous_state = None
        
    def calculate_reward(self, prev_state: Dict[str, Any], 
                        curr_state: Dict[str, Any]) -> float:
        """
        Calculate reward based on state changes.
        
        Args:
            prev_state: Previous game state
            curr_state: Current game state
            
        Returns:
            Calculated reward
        """
        if prev_state is None:
            return 0.0
        
        total_reward = 0.0
        
        # Basic rewards
        total_reward += self._calculate_ring_reward(prev_state, curr_state)
        total_reward += self._calculate_enemy_reward(prev_state, curr_state)
        total_reward += self._calculate_power_up_reward(prev_state, curr_state)
        total_reward += self._calculate_level_reward(prev_state, curr_state)
        total_reward += self._calculate_game_over_penalty(prev_state, curr_state)
        
        # Movement rewards
        total_reward += self._calculate_progress_reward(prev_state, curr_state)
        total_reward += self._calculate_speed_reward(prev_state, curr_state)
        total_reward += self._calculate_height_reward(prev_state, curr_state)
        
        # Penalties
        total_reward += self._calculate_time_penalty(prev_state, curr_state)
        total_reward += self._calculate_stuck_penalty(prev_state, curr_state)
        total_reward += self._calculate_fall_penalty(prev_state, curr_state)
        
        return total_reward
    
    def _calculate_ring_reward(self, prev_state: Dict[str, Any], 
                              curr_state: Dict[str, Any]) -> float:
        """Calculate reward for collecting rings."""
        prev_rings = prev_state.get('rings', 0)
        curr_rings = curr_state.get('rings', 0)
        
        rings_collected = curr_rings - prev_rings
        return rings_collected * self.ring_reward
    
    def _calculate_enemy_reward(self, prev_state: Dict[str, Any], 
                               curr_state: Dict[str, Any]) -> float:
        """Calculate reward for defeating enemies."""
        # This would require tracking enemy count or positions
        # For now, we'll use a simplified approach based on score
        prev_score = prev_state.get('score', 0)
        curr_score = curr_state.get('score', 0)
        
        score_increase = curr_score - prev_score
        # Assume some score increase is from enemy defeats
        return score_increase * 0.1  # Small reward for score increase
    
    def _calculate_power_up_reward(self, prev_state: Dict[str, Any], 
                                  curr_state: Dict[str, Any]) -> float:
        """Calculate reward for collecting power-ups."""
        # This would require tracking power-up states
        # For now, return 0 (can be enhanced with memory reading)
        return 0.0
    
    def _calculate_level_reward(self, prev_state: Dict[str, Any], 
                               curr_state: Dict[str, Any]) -> float:
        """Calculate reward for completing a level."""
        prev_level = prev_state.get('level', 0)
        curr_level = curr_state.get('level', 0)
        
        if curr_level > prev_level:
            return self.level_complete_reward
        return 0.0
    
    def _calculate_game_over_penalty(self, prev_state: Dict[str, Any], 
                                    curr_state: Dict[str, Any]) -> float:
        """Calculate penalty for game over."""
        prev_lives = prev_state.get('lives', 0)
        curr_lives = curr_state.get('lives', 0)
        
        if curr_lives < prev_lives:
            return self.game_over_penalty
        return 0.0
    
    def _calculate_progress_reward(self, prev_state: Dict[str, Any], 
                                  curr_state: Dict[str, Any]) -> float:
        """Calculate reward for forward progress."""
        prev_pos = prev_state.get('position', (0, 0))
        curr_pos = curr_state.get('position', (0, 0))
        
        if isinstance(prev_pos, (list, tuple)) and isinstance(curr_pos, (list, tuple)):
            progress = curr_pos[0] - prev_pos[0]  # X-coordinate progress
            return max(0, progress) * self.progress_reward
        return 0.0
    
    def _calculate_speed_reward(self, prev_state: Dict[str, Any], 
                               curr_state: Dict[str, Any]) -> float:
        """Calculate reward for high speed."""
        curr_speed = curr_state.get('speed', 0)
        
        # Reward for maintaining high speed
        if curr_speed > 5:  # Threshold for "fast" speed
            return self.speed_reward
        return 0.0
    
    def _calculate_height_reward(self, prev_state: Dict[str, Any], 
                                curr_state: Dict[str, Any]) -> float:
        """Calculate reward for height (jumping)."""
        prev_pos = prev_state.get('position', (0, 0))
        curr_pos = curr_state.get('position', (0, 0))
        
        if isinstance(prev_pos, (list, tuple)) and isinstance(curr_pos, (list, tuple)):
            height_gain = prev_pos[1] - curr_pos[1]  # Y-coordinate (lower is higher)
            return max(0, height_gain) * self.height_reward
        return 0.0
    
    def _calculate_time_penalty(self, prev_state: Dict[str, Any], 
                               curr_state: Dict[str, Any]) -> float:
        """Calculate small penalty for time passing."""
        return self.time_penalty
    
    def _calculate_stuck_penalty(self, prev_state: Dict[str, Any], 
                                curr_state: Dict[str, Any]) -> float:
        """Calculate penalty for being stuck."""
        # This would require more sophisticated stuck detection
        # For now, return 0 (can be enhanced)
        return 0.0
    
    def _calculate_fall_penalty(self, prev_state: Dict[str, Any], 
                               curr_state: Dict[str, Any]) -> float:
        """Calculate penalty for falling."""
        prev_pos = prev_state.get('position', (0, 0))
        curr_pos = curr_state.get('position', (0, 0))
        
        if isinstance(prev_pos, (list, tuple)) and isinstance(curr_pos, (list, tuple)):
            fall_distance = curr_pos[1] - prev_pos[1]  # Y-coordinate increase
            if fall_distance > 50:  # Significant fall
                return self.fall_penalty
        return 0.0


class AdvancedRewardCalculator(RewardCalculator):
    """
    Advanced reward calculator with additional features.
    
    Features:
    - Ring chain bonuses
    - Speed combo rewards
    - Exploration rewards
    - Risk-reward balancing
    - Adaptive reward scaling
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Advanced reward parameters
        self.ring_chain_bonus = config.get('ring_chain_bonus', 5.0)
        self.speed_combo_bonus = config.get('speed_combo_bonus', 3.0)
        self.exploration_bonus = config.get('exploration_bonus', 2.0)
        self.risk_penalty = config.get('risk_penalty', -2.0)
        
        # State tracking
        self.ring_chain_count = 0
        self.speed_combo_count = 0
        self.visited_positions = set()
        self.last_speed = 0
        
    def calculate_reward(self, prev_state: Dict[str, Any], 
                        curr_state: Dict[str, Any]) -> float:
        """Enhanced reward calculation with advanced features."""
        base_reward = super().calculate_reward(prev_state, curr_state)
        
        # Advanced rewards
        chain_reward = self._calculate_ring_chain_reward(prev_state, curr_state)
        combo_reward = self._calculate_speed_combo_reward(prev_state, curr_state)
        exploration_reward = self._calculate_exploration_reward(prev_state, curr_state)
        risk_reward = self._calculate_risk_reward(prev_state, curr_state)
        
        total_reward = base_reward + chain_reward + combo_reward + exploration_reward + risk_reward
        
        return total_reward
    
    def _calculate_ring_chain_reward(self, prev_state: Dict[str, Any], 
                                    curr_state: Dict[str, Any]) -> float:
        """Calculate bonus for collecting rings in quick succession."""
        prev_rings = prev_state.get('rings', 0)
        curr_rings = curr_state.get('rings', 0)
        
        rings_collected = curr_rings - prev_rings
        
        if rings_collected > 0:
            self.ring_chain_count += 1
            # Bonus increases with chain length
            return self.ring_chain_bonus * self.ring_chain_count
        else:
            # Reset chain if no rings collected
            self.ring_chain_count = 0
            return 0.0
    
    def _calculate_speed_combo_reward(self, prev_state: Dict[str, Any], 
                                     curr_state: Dict[str, Any]) -> float:
        """Calculate bonus for maintaining high speed."""
        curr_speed = curr_state.get('speed', 0)
        
        if curr_speed > 5 and curr_speed >= self.last_speed:
            self.speed_combo_count += 1
            return self.speed_combo_bonus * self.speed_combo_count
        else:
            self.speed_combo_count = 0
            return 0.0
        
        self.last_speed = curr_speed
    
    def _calculate_exploration_reward(self, prev_state: Dict[str, Any], 
                                     curr_state: Dict[str, Any]) -> float:
        """Calculate reward for exploring new areas."""
        curr_pos = curr_state.get('position', (0, 0))
        
        if isinstance(curr_pos, (list, tuple)):
            # Discretize position for exploration tracking
            discrete_pos = (curr_pos[0] // 50, curr_pos[1] // 50)
            
            if discrete_pos not in self.visited_positions:
                self.visited_positions.add(discrete_pos)
                return self.exploration_bonus
        
        return 0.0
    
    def _calculate_risk_reward(self, prev_state: Dict[str, Any], 
                              curr_state: Dict[str, Any]) -> float:
        """Calculate reward/penalty for risky behavior."""
        curr_pos = curr_state.get('position', (0, 0))
        curr_speed = curr_state.get('speed', 0)
        
        # Risk increases with speed and height
        risk_factor = curr_speed * 0.1
        
        if isinstance(curr_pos, (list, tuple)):
            # Higher risk when high up and moving fast
            if curr_pos[1] < 100 and curr_speed > 8:  # High and fast
                return -self.risk_penalty * risk_factor
        
        return 0.0
    
    def reset(self):
        """Reset advanced reward state."""
        self.ring_chain_count = 0
        self.speed_combo_count = 0
        self.visited_positions.clear()
        self.last_speed = 0


class SonicSpecificRewardCalculator(AdvancedRewardCalculator):
    """
    Sonic-specific reward calculator.
    
    Optimized for Sonic games with:
    - Spin dash rewards
    - Homing attack bonuses
    - Special stage rewards
    - Boss battle rewards
    - Time attack bonuses
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Sonic-specific rewards
        self.spin_dash_reward = config.get('spin_dash_reward', 5.0)
        self.homing_attack_reward = config.get('homing_attack_reward', 10.0)
        self.special_stage_reward = config.get('special_stage_reward', 50.0)
        self.boss_battle_reward = config.get('boss_battle_reward', 100.0)
        self.time_attack_bonus = config.get('time_attack_bonus', 2.0)
        
        # Sonic-specific state tracking
        self.is_spin_dashing = False
        self.last_ring_count = 0
        self.boss_battle_active = False
        
    def calculate_reward(self, prev_state: Dict[str, Any], 
                        curr_state: Dict[str, Any]) -> float:
        """Sonic-specific reward calculation."""
        base_reward = super().calculate_reward(prev_state, curr_state)
        
        # Sonic-specific rewards
        spin_dash_reward = self._calculate_spin_dash_reward(prev_state, curr_state)
        homing_attack_reward = self._calculate_homing_attack_reward(prev_state, curr_state)
        special_stage_reward = self._calculate_special_stage_reward(prev_state, curr_state)
        boss_battle_reward = self._calculate_boss_battle_reward(prev_state, curr_state)
        time_attack_reward = self._calculate_time_attack_reward(prev_state, curr_state)
        
        total_reward = (base_reward + spin_dash_reward + homing_attack_reward + 
                       special_stage_reward + boss_battle_reward + time_attack_reward)
        
        return total_reward
    
    def _calculate_spin_dash_reward(self, prev_state: Dict[str, Any], 
                                   curr_state: Dict[str, Any]) -> float:
        """Calculate reward for successful spin dash."""
        # This would require detecting spin dash state
        # For now, reward based on speed increase
        prev_speed = prev_state.get('speed', 0)
        curr_speed = curr_state.get('speed', 0)
        
        if curr_speed > prev_speed + 3:  # Significant speed increase
            return self.spin_dash_reward
        return 0.0
    
    def _calculate_homing_attack_reward(self, prev_state: Dict[str, Any], 
                                       curr_state: Dict[str, Any]) -> float:
        """Calculate reward for homing attacks (Sonic 3+)."""
        # This would require detecting homing attack state
        # For now, reward based on enemy defeats
        return 0.0
    
    def _calculate_special_stage_reward(self, prev_state: Dict[str, Any], 
                                       curr_state: Dict[str, Any]) -> float:
        """Calculate reward for special stage completion."""
        # This would require detecting special stage state
        return 0.0
    
    def _calculate_boss_battle_reward(self, prev_state: Dict[str, Any], 
                                     curr_state: Dict[str, Any]) -> float:
        """Calculate reward for boss battle progress."""
        # This would require detecting boss battle state
        return 0.0
    
    def _calculate_time_attack_reward(self, prev_state: Dict[str, Any], 
                                     curr_state: Dict[str, Any]) -> float:
        """Calculate bonus for fast level completion."""
        # This would require tracking level time
        return 0.0
    
    def reset(self):
        """Reset Sonic-specific state."""
        super().reset()
        self.is_spin_dashing = False
        self.last_ring_count = 0
        self.boss_battle_active = False 