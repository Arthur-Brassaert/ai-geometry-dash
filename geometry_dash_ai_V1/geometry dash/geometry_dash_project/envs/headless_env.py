import numpy as np
import torch

STATE_SIZE = 4
ACTION_SIZE = 2  # 0=niets, 1=jump
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GRAVITY = -0.5
JUMP_VEL = 10.0
PLAYER_Y_MIN = 0.0
PLAYER_Y_MAX = 100.0
OBSTACLE_SPEED = 1.0
MAX_OBSTACLES = 5

class HeadlessGeometryDashEnv:
    def __init__(self):
        self.num_envs = 1  # Enkel voor single env in headless
        self.reset()

    def reset(self):
        self.player_y = 0.0
        self.player_vy = 0.0
        self.obs_x = np.random.randint(20, 100, size=(MAX_OBSTACLES,))
        self.obs_y = np.random.randint(0, 50, size=(MAX_OBSTACLES,))
        self.done = False
        self.score = 0
        return self._get_state()

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True, {}

        # Jump
        if action == 1:
            self.player_vy = JUMP_VEL

        # Physics
        self.player_vy += GRAVITY
        self.player_y += self.player_vy
        self.player_y = np.clip(self.player_y, PLAYER_Y_MIN, PLAYER_Y_MAX)

        # Obstacles
        self.obs_x -= OBSTACLE_SPEED

        # Collision
        collision = np.any((self.obs_x < 1.0) & (np.abs(self.obs_y - self.player_y) < 5.0))
        self.done = collision

        # Respawn obstacles
        respawn = self.obs_x < 0
        self.obs_x[respawn] += 100.0
        self.obs_y[respawn] = np.random.randint(0, 50, size=respawn.sum())

        # Reward
        reward = 1.0 if not self.done else -10.0
        self.score += reward

        return self._get_state(), reward, self.done, {}

    def _get_state(self):
        # player_y, player_vy, nearest_obs_x, nearest_obs_y
        nearest_idx = np.argmin(self.obs_x)
        nearest_x = self.obs_x[nearest_idx]
        nearest_y = self.obs_y[nearest_idx]
        return np.array([self.player_y, self.player_vy, nearest_x, nearest_y], dtype=np.float32)
