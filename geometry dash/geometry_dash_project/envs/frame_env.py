import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .headless_env import HeadlessGeometryDashEnv

class GeometryDashFrameEnv(gym.Env):
    def __init__(self, width=84, height=84):
        super().__init__()
        self.width = width
        self.height = height
        self.headless_env = HeadlessGeometryDashEnv()

        self.observation_space = spaces.Box(0, 255, shape=(height, width, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(2)  # 0=nothing, 1=jump

    def reset(self, *, seed=None, options=None):
        # Als SB3 een seed doorgeeft
        if seed is not None:
            np.random.seed(seed)

        # originele reset-logica
        state = self.headless_env.reset()  # of hoe je normaal reset doet

        # return de observatie (bijv. frame of state)
        return state
    def step(self, action):
        state, reward, done, info = self.headless_env.step(action)
        frame = self._state_to_frame(state)
        return frame, reward, done, False, info

    def _state_to_frame(self, state):
        frame = np.zeros((self.height, self.width), dtype=np.uint8)
        x = int(min(self.width - 1, state[2] / 100 * self.width))  # nearest obstacle x
        y = int(min(self.height - 1, state[0] / 100 * self.height)) # player y
        frame[y, x] = 255
        return frame[:, :, None]
