import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from .geometry_dash_headless import VecGeometryDashEnv, STATE_SIZE, ACTION_SIZE

class GeometryDashGymEnv(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(STATE_SIZE,), dtype=np.float32)
        self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.int64)
        self.env = VecGeometryDashEnv(num_envs=1)  # 1 env per Gym env

    def reset(self, *, seed=None, options=None):
        # Zorg dat SB3 seed kan doorgeven
        if seed is not None:
            np.random.seed(seed)

        # originele reset-logica
        state = self.headless_env.reset()  # of hoe je reset normaal doet

        # Als je render frames maakt:
        # frame = self._state_to_frame(state)

        return state  # of frame als dat je observatie is

    def step(self, action):
        state, reward, done, info = self.env.step(np.array([action]))
        return state[0].cpu().numpy(), reward[0].item(), done[0].item(), False, info  # <- CPU/Numpy
