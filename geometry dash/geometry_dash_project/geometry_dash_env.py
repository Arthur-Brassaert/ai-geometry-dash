import numpy as np

STATE_SIZE = 4
ACTION_SIZE = 2  # 0 = niets, 1 = jump

class Player:
    def __init__(self):
        self.x = 50
        self.y = 300
        self.w = 50
        self.h = 50
        self.vy = 0
        self.gravity = 1000  # pixels/sec^2
        self.jump_strength = -400

class Obstacle:
    def __init__(self, x, y, w=50, h=50):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class HeadlessGeometryDashEnv:
    def __init__(self, render=False):
        # render ignored, this is fully headless
        self.player = Player()
        self.obstacles = []
        self.time = 0.0
        self.score = 0.0
        self.done = False
        self.timestep = 1/60 if render else 1.0
        self.next_obstacle_x = 800

    def reset(self):
        self.player = Player()
        self.obstacles = []
        self.time = 0.0
        self.score = 0.0
        self.done = False
        self.next_obstacle_x = 800
        return self._get_state()

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True, {}

        # --- Player physics ---
        if action == 1:
            self.player.vy = self.player.jump_strength

        self.player.vy += self.player.gravity * self.timestep
        self.player.y += self.player.vy * self.timestep

        # Prevent falling below ground
        if self.player.y > 550:
            self.player.y = 550
            self.player.vy = 0

        # --- Obstacles ---
        # Move existing obstacles
        for obs in self.obstacles:
            obs.x -= 300 * self.timestep  # obstacle speed

        # Remove passed obstacles
        self.obstacles = [o for o in self.obstacles if o.x + o.w > 0]

        # Generate new obstacles every 400 px
        if self.next_obstacle_x - (self.obstacles[-1].x if self.obstacles else 0) <= 0:
            self.obstacles.append(Obstacle(self.next_obstacle_x, 550))
            self.next_obstacle_x += 400

        # --- Collision detection ---
        self.done = False
        for obs in self.obstacles:
            if (self.player.x < obs.x + obs.w and
                self.player.x + self.player.w > obs.x and
                self.player.y < obs.y + obs.h and
                self.player.y + self.player.h > obs.y):
                self.done = True

        # --- Scoring ---
        self.score += 1.0

        # --- State vector ---
        state = self._get_state()

        # --- Reward ---
        reward = 1.0 if not self.done else -10.0

        return state, reward, self.done, {}

    def _get_state(self):
        """
        Simple state: [player_y, player_vy, distance_to_next_obstacle, obstacle_height]
        """
        next_obs = None
        for obs in self.obstacles:
            if obs.x + obs.w > self.player.x:
                next_obs = obs
                break
        if next_obs is None:
            next_obs = Obstacle(self.next_obstacle_x, 550)

        dist = next_obs.x - self.player.x
        return np.array([self.player.y, self.player.vy, dist, next_obs.h], dtype=np.float32)
