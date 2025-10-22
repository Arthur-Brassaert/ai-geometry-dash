# geometry_dash_render_env.py
import numpy as np
import pygame
from gymnasium import Env
from gymnasium.spaces import Box
from .geometry_dash_headless import HeadlessGeometryDashEnv

class GeometryDashRenderEnv(Env):
    def __init__(self, render=True):
        super().__init__()
        self.headless_env = HeadlessGeometryDashEnv()
        self.render_game = render

        self.observation_space = Box(low=0, high=1000, shape=(4,), dtype=np.float32)
        self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.int32)

        if self.render_game:
            pygame.init()
            self.screen_width = 800
            self.screen_height = 400
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Geometry Dash AI")
            self.clock = pygame.time.Clock()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        state, info = self.headless_env.reset()
        if self.render_game:
            self._render()
        return state, info

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action[0])

        state, reward, terminated, truncated, info = self.headless_env.step(action)
        if self.render_game:
            self._render()
        return state, reward, terminated, truncated, info

    def _render(self):
        if not self.render_game:
            return

        self.screen.fill((30, 30, 30))

        # krijg posities van speler en obstakel
        player_x, player_y, obs_x, obs_y = self.headless_env.get_render_state()

        # teken speler
        pygame.draw.rect(self.screen, (0, 255, 0), (player_x, self.screen_height - player_y - 50, 50, 50))
        # teken obstakel
        pygame.draw.rect(self.screen, (255, 0, 0), (obs_x, self.screen_height - obs_y - 50, 50, 50))

        pygame.display.update()
        self.clock.tick(60)

    def close(self):
        if self.render_game:
            pygame.quit()
