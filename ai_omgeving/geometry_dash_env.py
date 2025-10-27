import gymnasium as gym
import numpy as np
import pygame
import random
import os
from gymnasium import spaces
from pathlib import Path
import sys
import config

# -------------------------------
# Helper classes
# -------------------------------
class Player:
    def __init__(self):
        # Use the same per-second physics as the visual game:
        # velocity in px/s, gravity and jump impulses converted from config constants
        self.w = config.PLAYER_W
        self.h = config.PLAYER_H
        self.x = int(config.WIDTH * 0.2)
        self.y = config.HEIGHT - 80 - self.h
        # velocity in px/s (positive = downward)
        self.v = 0.0
        # convert gravity and jump (from config, nominally per-frame or tuned)
        # into per-second units used by the visual game code
        self.gravity_s = config.GRAVITY * (config.FPS ** 2)
        self.jump_strength_s = config.JUMP_STRENGTH * config.FPS
        # variable jump support
        self.jump_held = False
        self.jump_time = 0.0
        self.max_jump_hold = getattr(config, 'MAX_JUMP_HOLD', 0.12)
        self.hold_gravity_s = getattr(config, 'HOLD_GRAVITY', 0.75) * (config.FPS ** 2)
        self.auto_jump_on_land = getattr(config, 'AUTO_JUMP_ON_LAND', False)

    def jump(self):
        if self.on_ground():
            # apply per-second impulse
            self.v = self.jump_strength_s

    def on_ground(self):
        return self.y >= config.HEIGHT - 80 - self.h - 1

    def step(self, dt):
        # dt is in seconds. Use variable-hold gravity when jump is held and within hold window
        if self.jump_held and self.v < 0 and self.jump_time < self.max_jump_hold:
            # while holding, increment jump_time and apply reduced gravity
            self.jump_time += dt
            effective_gravity_s = self.hold_gravity_s
        else:
            effective_gravity_s = self.gravity_s

        # integrate velocity (px/s) and position
        self.v += effective_gravity_s * dt
        self.y += self.v * dt

        # landing clamp
        if self.y > config.HEIGHT - 80 - self.h:
            self.y = config.HEIGHT - 80 - self.h
            self.v = 0
            self.jump_time = 0.0
            if self.jump_held and self.auto_jump_on_land:
                self.v = self.jump_strength_s

class Obstacle:
    def __init__(self, x, w, h, kind='normal'):
        self.x = x
        self.w = w
        self.h = h
        self.y = config.HEIGHT - 80 - h
        self.kind = kind

    def step(self, speed, dt):
        self.x -= speed * dt

# -------------------------------
# Gym Environment
# -------------------------------
class GeometryDashEnv(gym.Env):
    metadata = {"render_modes": ["human", "headless"], "render_fps": config.FPS}

    def __init__(self, render_mode='headless', **kwargs):
        """Gym env used for training.

        Accepts optional keyword args used by the training script so callers can
        pass reward shaping and observation parameters. Unknown kwargs are
        ignored.
        """
        super().__init__()
        self.render_mode = render_mode

        # Reward shaping params (defaults preserved for backward-compat)
        self.reward_survival = float(kwargs.get('reward_survival', 1.0))
        self.reward_jump_success = float(kwargs.get('reward_jump_success', 0.0))
        self.reward_obstacle_avoid = float(kwargs.get('reward_obstacle_avoid', 0.0))
        self.penalty_crash = float(kwargs.get('penalty_crash', 0.0))
        self.penalty_late_jump = float(kwargs.get('penalty_late_jump', 0.0))
        self.penalty_early_jump = float(kwargs.get('penalty_early_jump', 0.0))
        self.reward_progress_scale = float(kwargs.get('reward_progress_scale', 0.0))

        # Observation parameters
        self.obs_horizon = int(kwargs.get('obs_horizon', config.OBS_HORIZON))
        self.obs_resolution = int(kwargs.get('obs_resolution', config.OBS_RESOLUTION))

        # Randomization toggle used by training script
        self.random_levels = bool(kwargs.get('random_levels', False))

        self.action_space = spaces.Discrete(2)  # 0=nothing, 1=jump
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(int(self.obs_horizon / self.obs_resolution),),
                                            dtype=np.float32)

        self._setup_game()
        self.screen = None

    def _setup_game(self):
        self.player = Player()
        self.obstacles = []
        self.speed = config.BASE_SPEED
        self.spawn_timer = random.randint(config.START_SPAWN_MIN, config.START_SPAWN_MAX)
        self.last_group_right_x = 0
        self.score = 0
        self.prev_x = 0.0
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup_game()
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        obs = np.zeros(int(self.obs_horizon / self.obs_resolution), dtype=np.float32)
        for o in self.obstacles:
            if o.x > self.player.x and (o.x - self.player.x) < self.obs_horizon:
                idx = int((o.x - self.player.x) / self.obs_resolution)
                if 0 <= idx < len(obs):
                    obs[idx] = 1.0
        return obs

    def step(self, action):
        dt = 1.0 / config.FPS
        self.current_step += 1

        # Jump logic
        if action == 1 and self.player.on_ground():
            self.player.jump_held = True
            self.player.jump()
        else:
            self.player.jump_held = False

        self.player.step(dt)

        # Spawn obstacles
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            group_count = np.random.randint(1, 3)
            group_w = np.random.uniform(config.PLAYER_W, config.PLAYER_W * 2)
            group_h = np.random.uniform(config.PLAYER_H, config.PLAYER_H * 2)
            gap = np.random.uniform(config.SPAWN_MIN_FLOOR, config.SPAWN_MAX_FLOOR)
            x_start = max(config.WIDTH + 20, self.last_group_right_x + gap)
            group_right = x_start
            for i in range(group_count):
                x_pos = x_start + i * (group_w + config.GROUP_INTERNAL_GAP)
                kind = 'spike' if np.random.random() < config.SPIKE_CHANCE else 'normal'
                obs = Obstacle(x_pos, group_w, group_h, kind)
                self.obstacles.append(obs)
                group_right = max(group_right, x_pos + group_w)
            self.last_group_right_x = group_right
            self.spawn_timer = random.randint(config.START_SPAWN_MIN, config.START_SPAWN_MAX)

        # Move obstacles
        for o in self.obstacles:
            o.step(self.speed, dt)
        self.obstacles = [o for o in self.obstacles if o.x + o.w > -50]

        # Collision detection
        hit_w = int(config.PLAYER_W * config.HITBOX_SCALE)
        hit_h = int(config.PLAYER_H * config.HITBOX_SCALE)
        hit_x = int(self.player.x + (self.player.w - hit_w) / 2)
        hit_y = int(self.player.y + (self.player.h - hit_h) / 2)
        collided = False
        for o in self.obstacles:
            if hit_x < o.x + o.w and hit_x + hit_w > o.x and hit_y < o.y + o.h and hit_y + hit_h > o.y:
                collided = True
                break

        # Reward (use configured survival reward by default)
        reward = float(self.reward_survival)
        done = collided
        truncated = self.current_step > 10000

        obs = self._get_obs()
        return obs, reward, done, truncated, {"score": self.score}

    def render(self):
        if self.render_mode != "human":
            return

        if not self.screen:
            pygame.init()
            self.screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
            pygame.display.set_caption("AI Geometry Dash")

        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((0, 0, 0))  # clear screen

        # Draw player
        pygame.draw.rect(self.screen, (0, 255, 0),
                         (self.player.x, self.player.y, self.player.w, self.player.h))

        # Draw obstacles
        for o in self.obstacles:
            color = (255, 0, 0) if o.kind == 'spike' else (255, 255, 255)
            pygame.draw.rect(self.screen, color, (o.x, o.y, o.w, o.h))

        pygame.display.flip()
        pygame.time.delay(int(1000 / config.FPS))

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None
