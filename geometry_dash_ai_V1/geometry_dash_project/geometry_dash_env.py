import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import os

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
    def __init__(self, headless=False):
        super(GeometryDashEnv, self).__init__()
        
        self.headless = headless
        if not headless:
            try:
                pygame.init()
            except:
                pass
        
        # Action space: 0 = don't jump, 1 = jump
        self.action_space = spaces.Discrete(2)
        
        # Observation space: [player_y, player_velocity, next_obstacle_distance, next_obstacle_height, next_obstacle_type]
        self.observation_space = spaces.Box(
            low=np.array([0, -50, 0, 0, 0], dtype=np.float32), 
            high=np.array([600, 50, 1000, 200, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.player = Player()
        self.obstacles = []
        
        # Game state variables
        self.score = 0
        self.game_over = False
        self.spawn_timer = 0
        self.spawn_min = 15
        self.spawn_max = 30
        self.speed = 6.0 * 60  # Convert to px/s
        self.elapsed_time = 0.0
        
        # For obstacle spawning
        self.last_group_right_x = -9999
        self.min_group_gap = self.player.w * 1
        
    def reset(self, seed=None, options=None):
        # Reset game state
        if seed is not None:
            np.random.seed(seed)
            
        self.player = Player()
        self.obstacles = []
        self.score = 0
        self.game_over = False
        self.spawn_timer = 0
        self.elapsed_time = 0.0
        self.last_group_right_x = -9999
        
        obs = self._get_observation()
        info = {}
        return obs, info
    
    def step(self, action):
        dt = 1.0 / 60.0  # Fixed timestep for consistency
        
        # Process action
        if action == 1 and self.player.on_ground():
            self.player.jump()
        
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
            self.spawn_timer = np.random.randint(self.spawn_min, self.spawn_max)
        
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
        group_count = np.random.choice(GROUP_SIZES)
        group_w = self.player.w
        group_h = self.player.h
        
        desired_x = self.WIDTH + 20
        chosen_group_gap = np.random.randint(self.player.w, self.player.w * 2)
        x_start_candidate = self.last_group_right_x + chosen_group_gap
        x_start = max(desired_x, x_start_candidate)
        
        group_right = x_start
        for i in range(group_count):
            x_pos = x_start + i * (group_w + GROUP_INTERNAL_GAP)
            kind = 'spike' if np.random.random() < SPIKE_CHANCE else 'normal'
            self.obstacles.append(Obstacle(x_pos, w=group_w, h=group_h, kind=kind))
            group_right = max(group_right, x_pos + group_w)
        
        self.last_group_right_x = group_right
    
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
        
        for obstacle in self.obstacles:
            if obstacle.x > self.player.x:  # Only obstacles ahead of player
                distance = obstacle.x - self.player.x
                if distance < min_distance:
                    min_distance = distance
                    next_obstacle = obstacle
        
        if next_obstacle:
            obstacle_type = 1.0 if next_obstacle.kind == 'spike' else 0.0
            return np.array([
                self.player.y / self.HEIGHT,  # Normalized player Y position
                self.player.v / 1000.0,       # Normalized player velocity
                min(min_distance / self.WIDTH, 1.0),  # Normalized distance to next obstacle
                next_obstacle.h / 200.0,      # Normalized obstacle height
                obstacle_type                 # Obstacle type (0=normal, 1=spike)
            ], dtype=np.float32)
        else:
            # No obstacles ahead
            return np.array([
                self.player.y / self.HEIGHT,
                self.player.v / 1000.0,
                1.0,  # Max distance
                0.0,  # No obstacle height
                0.0   # No obstacle type
            ], dtype=np.float32)
    
    def _calculate_reward(self):
        reward = 0.0
        
        # Small reward for surviving each frame
        reward += 0.1
        
        # Reward for passing obstacles
        reward += self.score * 10.0
        
        # Penalty for dying
        if self.game_over:
            reward -= 100.0
        
        # Reward for being close to the ground (encourages timing jumps)
        if self.player.on_ground():
            reward += 0.05
        
        return reward
    
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