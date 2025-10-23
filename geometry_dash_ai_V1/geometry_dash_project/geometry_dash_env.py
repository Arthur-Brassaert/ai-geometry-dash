import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import os
from dataclasses import dataclass, field
from typing import Optional, List

# Add the current directory to path so we can import the game
sys.path.append(os.path.dirname(__file__))

# Import game components directly
try:
    # Try to import from the main game file
    from geometry_dash_game import Player as GamePlayer, Obstacle as GameObstacle, WIDTH, HEIGHT, GROUP_SIZES, GROUP_INTERNAL_GAP, SPIKE_CHANCE
    HAS_GAME_IMPORT = True
except ImportError as e:
    print(f"Could not import from game file: {e}")
    print("Creating minimal game classes...")
    HAS_GAME_IMPORT = False
    # Define minimal versions if import fails
    WIDTH = 1000
    HEIGHT = 600
    GROUP_SIZES = [1, 2, 3]
    GROUP_INTERNAL_GAP = 0
    SPIKE_CHANCE = 0.25

# Define the Player class locally to avoid import issues
@dataclass
class EnvConfig:
    """Environment-level configuration for obstacle spawning and world sizes.

    These defaults match the previous behaviour but are grouped so they can be
    adjusted easily from training scripts.
    """
    width: int = WIDTH
    height: int = HEIGHT
    spawn_min: int = 15
    spawn_max: int = 30
    group_sizes: List[int] = field(default_factory=lambda: GROUP_SIZES.copy())
    group_internal_gap: int = GROUP_INTERNAL_GAP
    spike_chance: float = SPIKE_CHANCE
    speed_px_s: float = 6.0 * 60
    player_start_x_ratio: float = 0.2


@dataclass
class RewardConfig:
    """Central place to tune reward/penalty magnitudes."""
    frame_reward: float = 0.1
    pass_reward: float = 10.0
    death_penalty: float = -100.0
    ground_bonus: float = 0.05
    # penalty for jumping when no close obstacle (discourage unnecessary jumps)
    jump_penalty: float = 0.0
    # penalty proportional to abrupt changes in vertical velocity (encourage smoothness)
    smoothness_penalty: float = 0.0


class Player:
    def __init__(self):
        self.w = 40
        self.h = 40
        self.x = int(WIDTH * 0.2)
        self.y = HEIGHT - 80 - self.h
        self.v = 0.0
        self.gravity_s = 1.0 * (60 ** 2)  # px/s^2
        self.jump_strength_s = -16 * 60   # px/s
        self.jump_held = False
        self.jump_time = 0.0
        self.max_jump_hold = 0.12
        self.hold_gravity_s = 0.75 * (60 ** 2)
        self.auto_jump_on_land = True

    def jump(self):
        if self.on_ground():
            self.v = self.jump_strength_s

    def on_ground(self):
        return self.y >= HEIGHT - 80 - self.h - 1

    def step(self, dt: float):
        if self.jump_held and self.v < 0 and self.jump_time < self.max_jump_hold:
            self.jump_time += dt
            effective_gravity_s = self.hold_gravity_s
        else:
            effective_gravity_s = self.gravity_s

        self.v += effective_gravity_s * dt
        self.y += self.v * dt

        if self.y > HEIGHT - 80 - self.h:
            self.y = HEIGHT - 80 - self.h
            self.v = 0
            self.jump_time = 0.0
            if self.jump_held and self.auto_jump_on_land:
                self.v = self.jump_strength_s

# Define the Obstacle class locally
class Obstacle:
    def __init__(self, x, w=None, h=None, kind='normal'):
        self.w = w if w is not None else 40
        self.h = h if h is not None else 40
        self.x = x
        self.y = HEIGHT - 80 - self.h
        self.kind = kind
        self.tex = None
        self.cleared = False

    def step(self, speed, dt: float):
        self.x -= speed * dt

class GeometryDashEnv(gym.Env):
    def __init__(self, headless: bool = False, env_cfg: Optional[EnvConfig] = None, reward_cfg: Optional[RewardConfig] = None, seed: Optional[int] = None):
        """Geometry Dash gym environment.

        Accepts grouped configuration objects for easy tuning from training scripts.
        """
        super(GeometryDashEnv, self).__init__()

        self.headless = headless
        if not headless:
            try:
                pygame.init()
            except Exception:
                pass

        # Configs
        self.env_cfg = env_cfg or EnvConfig()
        self.reward_cfg = reward_cfg or RewardConfig()

        # Per-environment RNG for reproducible random levels per-episode
        self._seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

        # Action space: 0 = don't jump, 1 = jump
        self.action_space = spaces.Discrete(2)

        # Observation space (normalized):
        # [player_y_norm (0..1), player_v_norm (-1..1), next_dist_norm (0..1), next_h_norm (0..1),
        #  next_type (0/1), ttc_norm (0..1), second_dist_norm (0..1), dist_to_ground_norm (0..1), jump_avail (0/1)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.WIDTH = self.env_cfg.width
        self.HEIGHT = self.env_cfg.height
        self.player = Player()
        self.obstacles = []

        # Game state variables
        self.score = 0
        self.game_over = False
        self.spawn_timer = int(self.rng.randint(self.env_cfg.spawn_min, self.env_cfg.spawn_max))
        self.spawn_min = self.env_cfg.spawn_min
        self.spawn_max = self.env_cfg.spawn_max
        self.speed = self.env_cfg.speed_px_s
        self.elapsed_time = 0.0

        # For obstacle spawning
        self.last_group_right_x = -9999
        self.min_group_gap = self.player.w * 1
        
    def reset(self, seed=None, options=None):
        # Reset game state
        if seed is not None:
            # create a local RNG for this episode
            self._seed = seed
            self.rng = np.random.RandomState(seed)

        # Recreate player and world state
        self.player = Player()
        self.obstacles = []
        self.score = 0
        self.game_over = False
        self.spawn_timer = int(self.rng.randint(self.spawn_min, self.spawn_max))
        self.elapsed_time = 0.0
        self.last_group_right_x = -9999

        # action tracking for jump penalties and smoothing
        self.last_action = 0
        self.last_v = self.player.v

        obs = self._get_observation()
        info = {}
        return obs, info
    
    def step(self, action):
        dt = 1.0 / 60.0  # Fixed timestep for consistency
        
        # Process action
        # Only allow jump when on ground (game rule)
        if action == 1 and self.player.on_ground():
            self.player.jump()
        # record action for reward shaping
        self.last_action = int(bool(action))
        
        # Update game state
        self._update_game_state(dt)
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        terminated = self.game_over
        truncated = False
        
        # Info dictionary
        info = {
            'score': self.score,
            'obstacles_passed': len([o for o in self.obstacles if o.cleared])
        }
        
        return obs, reward, terminated, truncated, info
    
    def _update_game_state(self, dt):
        # Update player
        self.player.step(dt)
        
        # Spawn obstacles
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self._spawn_obstacle_group()
            self.spawn_timer = int(self.rng.randint(self.spawn_min, self.spawn_max))
        
        # Move obstacles
        for obstacle in self.obstacles:
            obstacle.step(self.speed, dt)
        
        # Remove off-screen obstacles
        self.obstacles = [o for o in self.obstacles if o.x + o.w > -50]
        
        # Check collisions
        self._check_collisions()
        
        # Update score
        self._update_score()
        
        self.elapsed_time += dt
    
    def _spawn_obstacle_group(self):
        group_count = int(self.rng.choice(self.env_cfg.group_sizes))
        group_w = self.player.w
        group_h = self.player.h

        desired_x = self.WIDTH + 20
        chosen_group_gap = int(self.rng.randint(self.player.w, self.player.w * 2))
        x_start_candidate = self.last_group_right_x + chosen_group_gap
        x_start = max(desired_x, x_start_candidate)

        group_right = x_start
        for i in range(group_count):
            x_pos = x_start + i * (group_w + self.env_cfg.group_internal_gap)
            kind = 'spike' if self.rng.random_sample() < self.env_cfg.spike_chance else 'normal'
            self.obstacles.append(Obstacle(x_pos, w=group_w, h=group_h, kind=kind))
            group_right = max(group_right, x_pos + group_w)

        self.last_group_right_x = group_right

    def apply_env_updates(self, updates: dict):
        """Apply dynamic updates to the environment configuration.

        Expected keys: 'spike_chance', 'speed_px_s', 'spawn_min', 'spawn_max'
        This method is safe to call during training and will coerce values to
        the appropriate internal fields.
        """
        try:
            if 'spike_chance' in updates:
                v = float(updates['spike_chance'])
                # clamp to [0,1]
                self.env_cfg.spike_chance = max(0.0, min(1.0, v))
                self.spawn_timer = int(self.rng.randint(self.spawn_min, self.spawn_max))
            if 'speed_px_s' in updates:
                self.speed = float(updates['speed_px_s'])
                self.env_cfg.speed_px_s = float(updates['speed_px_s'])
            if 'spawn_min' in updates:
                self.spawn_min = int(updates['spawn_min'])
                self.env_cfg.spawn_min = int(updates['spawn_min'])
            if 'spawn_max' in updates:
                self.spawn_max = int(updates['spawn_max'])
                self.env_cfg.spawn_max = int(updates['spawn_max'])
        except Exception:
            pass

    def get_env_cfg(self):
        """Return a small dict with current environment configuration values for inspection."""
        try:
            return {
                'spike_chance': float(self.env_cfg.spike_chance),
                'speed_px_s': float(getattr(self, 'speed', self.env_cfg.speed_px_s)),
                'spawn_min': int(getattr(self, 'spawn_min', self.env_cfg.spawn_min)),
                'spawn_max': int(getattr(self, 'spawn_max', self.env_cfg.spawn_max)),
            }
        except Exception:
            return None
    
    def _check_collisions(self):
        hit_w = int(self.player.w * 0.5)
        hit_h = int(self.player.h * 0.5)
        hit_x = int(self.player.x + (self.player.w - hit_w) / 2)
        hit_y = int(self.player.y + (self.player.h - hit_h) / 2)
        
        for obstacle in self.obstacles:
            if (hit_x < obstacle.x + obstacle.w and 
                hit_x + hit_w > obstacle.x and 
                hit_y < obstacle.y + obstacle.h and 
                hit_y + hit_h > obstacle.y):
                self.game_over = True
                break
    
    def _update_score(self):
        for obstacle in self.obstacles:
            if not obstacle.cleared and (obstacle.x + obstacle.w) < self.player.x:
                obstacle.cleared = True
                self.score += 1
    
    def _get_observation(self):
        # Find the next obstacle
        next_obstacle = None
        min_distance = float('inf')
        second_obstacle = None
        min_distance_2 = float('inf')
        
        for obstacle in self.obstacles:
            if obstacle.x > self.player.x:  # Only obstacles ahead of player
                distance = obstacle.x - self.player.x
                if distance < min_distance:
                    min_distance = distance
                    next_obstacle = obstacle
                
            # find second-next obstacle
            for obstacle in self.obstacles:
                if obstacle.x > self.player.x and obstacle is not next_obstacle:
                    distance = obstacle.x - self.player.x
                    if distance < min_distance_2:
                        min_distance_2 = distance
                        second_obstacle = obstacle
        
        # compute normalized time-to-collision (TTC) for next obstacle (seconds)
        ttc = 1.0
        if next_obstacle and self.speed > 0:
            # TTC in seconds (distance / speed)
            ttc_val = min_distance / max(1e-6, self.speed)
            # normalize to [0,1] by capping at 5s
            ttc = min(ttc_val / 5.0, 1.0)

        if next_obstacle:
            obstacle_type = 1.0 if next_obstacle.kind == 'spike' else 0.0
            second_dist = min(min_distance_2 / self.WIDTH, 1.0) if second_obstacle else 1.0
            distance_norm = min(min_distance / self.WIDTH, 1.0)
            dist_to_ground = (self.HEIGHT - 80 - self.player.y - self.player.h) / float(self.HEIGHT)
            jump_avail = 1.0 if self.player.on_ground() else 0.0
            return np.array([
                self.player.y / self.HEIGHT,        # Normalized player Y position
                self.player.v / 1000.0,             # Normalized player velocity
                distance_norm,                      # Distance to next obstacle (norm)
                next_obstacle.h / 200.0,            # Next obstacle height
                obstacle_type,                      # Next obstacle type
                ttc,                                # Normalized time-to-collision
                second_dist,                        # Second-next obstacle distance (norm)
                dist_to_ground,                     # Distance to ground (norm)
                jump_avail                          # Jump available flag
            ], dtype=np.float32)
        else:
            # No obstacles ahead
            dist_to_ground = (self.HEIGHT - 80 - self.player.y - self.player.h) / float(self.HEIGHT)
            jump_avail = 1.0 if self.player.on_ground() else 0.0
            return np.array([
                self.player.y / self.HEIGHT,
                self.player.v / 1000.0,
                1.0,  # Max distance
                0.0,  # No obstacle height
                0.0,  # No obstacle type
                1.0,  # ttc max
                1.0,  # second obstacle max
                dist_to_ground,
                jump_avail
            ], dtype=np.float32)
    
    def _calculate_reward(self):
        r = 0.0
        # per-frame survival reward
        r += self.reward_cfg.frame_reward
        # reward for each obstacle passed this episode (scaled)
        r += self.score * self.reward_cfg.pass_reward
        # death penalty
        if self.game_over:
            r += self.reward_cfg.death_penalty
        # small bonus when on ground
        if self.player.on_ground():
            r += self.reward_cfg.ground_bonus
        # penalty for jumping when no nearby obstacle
        try:
            # find next obstacle distance in pixels
            next_obs = None
            next_dist = float('inf')
            for o in self.obstacles:
                if o.x > self.player.x and (o.x - self.player.x) < next_dist:
                    next_dist = o.x - self.player.x
                    next_obs = o
            # if last action was a jump and the nearest obstacle is far, apply penalty
            if self.last_action == 1:
                # threshold: 0.25 * width
                if next_dist > 0.25 * self.WIDTH:
                    r += -abs(self.reward_cfg.jump_penalty)
        except Exception:
            pass

        # smoothness penalty: penalize large instantaneous vertical velocity changes
        try:
            dv = abs(self.player.v - getattr(self, 'last_v', self.player.v))
            if self.reward_cfg.smoothness_penalty and dv > 0:
                r += -self.reward_cfg.smoothness_penalty * (dv / 1000.0)
        except Exception:
            pass

        # update last_v for next frame
        self.last_v = self.player.v
        return r
    
    def render(self):
        if self.headless:
            return
            
        # Simple rendering for training visualization
        if not hasattr(self, 'screen'):
            try:
                self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
                self.clock = pygame.time.Clock()
                self.font = pygame.font.SysFont(None, 36)
            except:
                return
        
        self.screen.fill((0, 0, 0))
        
        # Draw ground
        pygame.draw.rect(self.screen, (34, 139, 34), (0, self.HEIGHT - 80, self.WIDTH, 80))
        
        # Draw player
        pygame.draw.rect(self.screen, (220, 60, 60), 
                        (self.player.x, self.player.y, self.player.w, self.player.h))
        
        # Draw obstacles
        for obstacle in self.obstacles:
            if obstacle.kind == 'spike':
                # Draw triangle for spikes
                points = [
                    (obstacle.x, obstacle.y + obstacle.h),
                    (obstacle.x + obstacle.w, obstacle.y + obstacle.h),
                    (obstacle.x + obstacle.w // 2, obstacle.y)
                ]
                pygame.draw.polygon(self.screen, (120, 120, 120), points)
            else:
                # Draw normal obstacles as isosceles triangles as well to match the main game
                points = [
                    (obstacle.x, obstacle.y + obstacle.h),
                    (obstacle.x + obstacle.w, obstacle.y + obstacle.h),
                    (obstacle.x + obstacle.w // 2, obstacle.y)
                ]
                pygame.draw.polygon(self.screen, (120, 120, 120), points)
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(60) 
    
    def close(self):
        if hasattr(self, 'screen'):
            try:
                pygame.quit()
            except:
                pass

        # If the real game provides Player/Obstacle implementations, prefer those so
        # the Gym environment matches the real gameplay exactly.
        try:
            if HAS_GAME_IMPORT:
                # overwrite local classes with game implementations
                Player = GamePlayer  # type: ignore
                Obstacle = GameObstacle  # type: ignore
        except Exception:
            # If anything goes wrong, continue using the local fallback classes.
            pass