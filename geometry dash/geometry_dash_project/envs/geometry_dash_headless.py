# geometry_dash_headless.py
import numpy as np
import torch
from gymnasium import Env

STATE_SIZE = 4        # [player_y, player_vy, nearest_obs_x, nearest_obs_y]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GRAVITY = -0.5
JUMP_VEL = 10.0
PLAYER_Y_MIN = 0.0
PLAYER_Y_MAX = 100.0
OBSTACLE_SPEED = 1.0

class HeadlessGeometryDashEnv(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = np.zeros(STATE_SIZE, dtype=np.float32)
        self.action_space = np.zeros(1, dtype=np.int32)

        self.player_y = 0.0
        self.player_vy = 0.0
        self.next_obstacle_x = 800.0
        self.done = False
        self.score = 0.0
        self.reset()

    def reset(self, **kwargs):
        self.player_y = 50.0
        self.player_vy = 0.0
        self.next_obstacle_x = 800.0
        self.done = False
        self.score = 0.0
        return self._get_state().cpu().numpy(), {}

    def step(self, action):
        action = torch.tensor(action, device=DEVICE, dtype=torch.int32).reshape(1)

        if action.item() == 1 and not self.done:
            self.player_vy = JUMP_VEL

        # Physics
        self.player_vy += GRAVITY
        self.player_y += self.player_vy
        self.player_y = float(np.clip(self.player_y, PLAYER_Y_MIN, PLAYER_Y_MAX))

        # Move obstacle
        self.next_obstacle_x -= OBSTACLE_SPEED

        # Collision detection
        if abs(self.next_obstacle_x - 100) < 10 and abs(50.0 - self.player_y) < 10:
            self.done = True

        # Reward
        reward = -10.0 if self.done else 1.0
        self.score += reward

        obs = self._get_state().cpu().numpy()
        terminated = self.done
        truncated = False
        info = {}
        return obs, np.array([reward], dtype=np.float32), terminated, truncated, info

    def _get_state(self):
        import torch
        player_y = torch.tensor([self.player_y], device=DEVICE, dtype=torch.float32)
        player_vy = torch.tensor([self.player_vy], device=DEVICE, dtype=torch.float32)
        nearest_x = torch.tensor([self.next_obstacle_x], device=DEVICE, dtype=torch.float32)
        nearest_y = torch.tensor([50.0], device=DEVICE, dtype=torch.float32)
        state = torch.cat([player_y, player_vy, nearest_x, nearest_y], dim=0)
        return state

    def get_render_state(self):
        """
        Return player and obstacle positions for Pygame rendering
        """
        player_x = 100                 # constante x van speler
        player_y = self.player_y * 3   # schaal 0-100 -> pixels
        obs_x = self.next_obstacle_x
        obs_y = 50 * 3                 # vaste hoogte
        return player_x, player_y, obs_x, obs_y
