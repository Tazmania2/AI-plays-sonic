import gymnasium as gym
import numpy as np

class HierarchicalShapingWrapper(gym.Wrapper):
    """
    Hierarchical shaping wrapper for Sonic RL.
    Implements phase-based reward shaping with micro/mid/macro/termination levels.
    """
    def __init__(self, env, reward_mode='baseline', shaping_phase_steps=500_000):
        super().__init__(env)
        self.reward_mode = reward_mode
        self.shaping_phase_steps = shaping_phase_steps
        self.current_step = 0
        self.in_shaping_phase = (reward_mode == 'shaping')
        self.performance_window = []  # For backtracking
        self.performance_threshold = 0.8  # e.g. 80% jump success
        self.window_size = 1000
        self._reset_counters()
        # Action index to meaning mapping
        self.action_meanings = getattr(env, 'action_meanings', None)
        if self.action_meanings is None and hasattr(env, 'action_config'):
            self.action_meanings = env.action_config.get('basic', [])

    def _reset_counters(self):
        self.cnt_move_right = 0
        self.cnt_jump = 0
        self.cnt_obstacle_jumps = 0
        self.cnt_hazard_avoidance = 0
        self.cnt_rings = 0
        self.cnt_enemies = 0
        self.cnt_exploration = 0
        self.reached_end = 0
        self.total_reward = 0.0
        self.visited_cells = set()
        self.last_x = None
        self.last_lives = None
        self.last_rings = None
        self.last_level = None
        self.last_pos = None
        self.last_vy = None
        self.last_speed = None
        self.jump_successes = 0
        self.jump_attempts = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_counters()
        self.current_step = 0
        self.in_shaping_phase = (self.reward_mode == 'shaping')
        self.last_x = None
        self.last_lives = None
        self.last_rings = None
        self.last_level = None
        self.last_pos = None
        self.last_vy = None
        self.last_speed = None
        self.jump_successes = 0
        self.jump_attempts = 0
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
            truncated = False
        self.current_step += 1
        shaping_active = self.in_shaping_phase and (self.current_step < self.shaping_phase_steps)

        # --- Get full game state from emulator ---
        game_state = self.env.emulator.get_game_state()
        x, y = game_state.get('position', (0, 0))
        vx, vy = game_state.get('velocity', (0, 0))
        rings = game_state.get('rings', 0)
        score = game_state.get('score', 0)
        lives = game_state.get('lives', 0)
        zone, act = game_state.get('zone_act', (0, 0))
        game_mode = game_state.get('game_mode', 0)
        timer = game_state.get('timer', (0, 0, 0))
        invincibility = game_state.get('invincibility', False)
        shield = game_state.get('shield', False)

        # --- Baseline reward: forward progress, rings, score, death penalty ---
        baseline_reward = 0.0
        if not hasattr(self, '_last_x'):
            self._last_x = x
        if not hasattr(self, '_last_rings'):
            self._last_rings = rings
        if not hasattr(self, '_last_score'):
            self._last_score = score
        if not hasattr(self, '_last_lives'):
            self._last_lives = lives
        if not hasattr(self, '_last_zone'):
            self._last_zone = zone
        if not hasattr(self, '_last_act'):
            self._last_act = act
        if not hasattr(self, '_first_completion_logged'):
            self._first_completion_logged = False
        if not hasattr(self, '_best_score'):
            self._best_score = 0
        if not hasattr(self, '_best_custom_reward'):
            self._best_custom_reward = 0.0
        if not hasattr(self, '_first_completion_step'):
            self._first_completion_step = None
        if not hasattr(self, '_first_completion_episode'):
            self._first_completion_episode = None

        # Forward progress
        dx = x - self._last_x
        if dx > 0:
            baseline_reward += dx * 0.01
        drings = rings - self._last_rings
        if drings > 0:
            baseline_reward += drings * 1.0
        dscore = score - self._last_score
        if dscore > 0:
            baseline_reward += dscore * 0.001
        if lives < self._last_lives:
            baseline_reward -= 100.0
            done = True

        # --- Robust act/zone/game_mode completion detection ---
        act_complete = False
        if (zone > self._last_zone) or (act > self._last_act):
            act_complete = True
        if game_mode in [0x18, 0x1C]:
            act_complete = True
        if act_complete:
            baseline_reward += 1000.0
            done = True

        # Time penalty
        baseline_reward -= 0.01

        self._last_x = x
        self._last_rings = rings
        self._last_score = score
        self._last_lives = lives
        self._last_zone = zone
        self._last_act = act

        # --- Shaping reward: add velocity, speed, invincibility, shield, timer, etc. ---
        shaping_reward = baseline_reward
        if shaping_active:
            shaping_reward += abs(vx) * 0.001
            if invincibility:
                shaping_reward += 0.5
            if shield:
                shaping_reward += 0.2
            min_, sec, frame = timer
            time_penalty = (min_ * 60 + sec) * 0.001
            shaping_reward -= time_penalty
            if hasattr(self, '_last_obs') and np.array_equal(obs, self._last_obs):
                shaping_reward -= 1.0
        self._last_obs = obs

        # --- Choose reward based on mode ---
        if self.reward_mode == 'shaping' and shaping_active:
            final_reward = shaping_reward
        else:
            final_reward = baseline_reward

        # --- Track and log first and best completions for A/B test ---
        if act_complete:
            if not self._first_completion_logged:
                self._first_completion_logged = True
                self._first_completion_step = self.current_step
                self._first_completion_episode = info.get('episode', 1)
                self._first_completion_score = score
                self._first_completion_custom = final_reward
                print(f"[A/B TEST] FIRST COMPLETION: steps={self.current_step}, episode={info.get('episode', 1)}, score={score}, custom_reward={final_reward:.2f}")
            # Track best score and custom reward
            if score > self._best_score:
                self._best_score = score
            if final_reward > self._best_custom_reward:
                self._best_custom_reward = final_reward

        # --- Info dict update ---
        info = info or {}
        info.update({
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'rings': rings,
            'score': score,
            'lives': lives,
            'zone': zone,
            'act': act,
            'game_mode': game_mode,
            'timer': timer,
            'invincibility': invincibility,
            'shield': shield,
            'reward_mode': self.reward_mode,
            'shaping_active': shaping_active,
            'final_reward': final_reward,
            'act_complete': act_complete,
            'first_completion_logged': self._first_completion_logged,
            'first_completion_step': self._first_completion_step,
            'first_completion_episode': self._first_completion_episode,
            'first_completion_score': getattr(self, '_first_completion_score', None),
            'first_completion_custom': getattr(self, '_first_completion_custom', None),
            'best_score': self._best_score,
            'best_custom_reward': self._best_custom_reward
        })

        return obs, final_reward, done, truncated, info

    def _apply_hierarchical_shaping(self, reward, obs, action, info, done):
        shaping_active = self.in_shaping_phase and (self.current_step < self.shaping_phase_steps)
        if not shaping_active and self.reward_mode == 'shaping':
            if len(self.performance_window) >= self.window_size:
                avg_jump = np.mean(self.performance_window)
                if avg_jump < self.performance_threshold:
                    shaping_active = True
            if self.jump_attempts > 0:
                self.performance_window.append(self.jump_successes / max(1, self.jump_attempts))
                if len(self.performance_window) > self.window_size:
                    self.performance_window.pop(0)
        shaped_reward = 0.0
        x = info.get('position', (0, 0))[0]
        y = info.get('position', (0, 0))[1]
        vy = info.get('velocity', (0, 0))[1] if 'velocity' in info else None
        speed = info.get('velocity', (0, 0))[0] if 'velocity' in info else None
        rings = info.get('rings', 0)
        lives = info.get('lives', 0)
        level = info.get('level', 0)
        score = info.get('score', 0)
        cell = (x // 50, y // 50)
        if cell not in self.visited_cells:
            self.visited_cells.add(cell)
            if shaping_active:
                shaped_reward += 0.5
                self.cnt_exploration += 1
        if shaping_active and self.action_meanings:
            if self.action_meanings[action] == 'RIGHT':
                shaped_reward += 0.5
                self.cnt_move_right += 1
            if self.action_meanings[action] == 'A':
                shaped_reward += 1.0
                self.cnt_jump += 1
                self.jump_attempts += 1
                if vy is not None and vy > 2:
                    shaped_reward += 0.5
                    self.jump_successes += 1
        if shaping_active:
            if self.last_rings is not None and rings > self.last_rings:
                shaped_reward += 2.0
                self.cnt_rings += (rings - self.last_rings)
        if shaping_active or not self.in_shaping_phase:
            if self.last_level is not None and level > self.last_level:
                shaped_reward += 5.0
                self.reached_end += 1
            if speed is not None and speed > 5:
                shaped_reward += 1.0
        # Termination-level
        if isinstance(done, (np.ndarray, list)):
            done_flag = np.any(done)
        else:
            done_flag = done
        if done_flag:
            if info.get('level_completed', False):
                shaped_reward += 1500.0
            elif lives <= 0:
                shaped_reward -= 200.0
        if self.reward_mode == 'baseline' or (not shaping_active and self.reward_mode == 'shaping'):
            shaped_reward += reward
        self.total_reward += shaped_reward
        self.last_x = x
        self.last_lives = lives
        self.last_rings = rings
        self.last_level = level
        self.last_pos = (x, y)
        self.last_vy = vy
        self.last_speed = speed
        info['cnt_move_right'] = self.cnt_move_right
        info['cnt_jump'] = self.cnt_jump
        info['cnt_obstacle_jumps'] = self.cnt_obstacle_jumps
        info['cnt_hazard_avoidance'] = self.cnt_hazard_avoidance
        info['cnt_rings'] = self.cnt_rings
        info['cnt_enemies'] = self.cnt_enemies
        info['cnt_exploration'] = self.cnt_exploration
        info['reached_end'] = self.reached_end
        info['total_reward'] = self.total_reward
        info['jump_success_rate'] = self.jump_successes / max(1, self.jump_attempts)
        return shaped_reward 