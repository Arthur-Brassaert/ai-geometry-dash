import pygame
from stable_baselines3 import PPO
from geometry_dash_game import main as gd_main  # use your full main game loop
from geometry_dash_game import Player, Obstacle  # if needed for observation

# Define default window size used by the renderer so WIDTH/HEIGHT are available.
# Adjust these values to match your game's expected resolution.
WIDTH, HEIGHT = 800, 600

class GDEnv:
    """Wrap the real Geometry Dash loop for AI control."""
    def __init__(self):
        # initialize the real game but skip audio and interactive input
        self.player = Player()
        self.obstacles = []
        self.done = False
        self.score = 0
        self.spawn_timer = 0
        # you can initialize other game parameters if needed

    def reset(self):
        self.player = Player()
        self.obstacles = []
        self.done = False
        self.score = 0
        self.spawn_timer = 0
        return self._get_obs()

    def _get_obs(self):
        """Observation: player y, velocity, distance to next obstacle, obstacle width"""
        next_ob = None
        for ob in self.obstacles:
            if ob.x + ob.w > self.player.x:
                next_ob = ob
                break
        if next_ob is None:
            dist = 500  # arbitrary far distance if no obstacle
            w = 50
        else:
            dist = next_ob.x - self.player.x
            w = next_ob.w
        return [self.player.y, self.player.v, dist, w]

    def step(self, action):
        # AI jump logic
        self.player.jump_held = bool(action)
        if action:
            self.player.jump()

        dt = 1/60  # assuming fixed timestep
        self.player.step(dt)
        speed_s = 200  # example game speed in px/s
        for o in self.obstacles:
            o.step(speed_s, dt)
        self.obstacles = [o for o in self.obstacles if o.x + o.w > -50]

        # spawn obstacles (simplified for AI)
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            # spawn a simple obstacle for testing
            self.obstacles.append(Obstacle(x=600))
            self.spawn_timer = 60

        # collision detection
        hit_w = int(self.player.w * 0.8)
        hit_h = int(self.player.h * 0.8)
        hit_x = int(self.player.x + (self.player.w - hit_w)/2)
        hit_y = int(self.player.y + (self.player.h - hit_h)/2)
        collided = False
        for o in self.obstacles:
            if hit_x < o.x + o.w and hit_x + hit_w > o.x and hit_y < o.y + o.h and hit_y + hit_h > o.y:
                collided = True
                break
        if collided:
            self.done = True

        # scoring
        for o in self.obstacles:
            if not getattr(o, 'cleared', False) and (o.x + o.w) < self.player.x:
                o.cleared = True
                self.score += 1

        return self._get_obs(), 1, self.done, {}  # reward = 1 per step

    def render(self):
        # call your original rendering code here
        screen = pygame.display.get_surface()
        if screen is None:
            screen = pygame.display.set_mode((WIDTH, HEIGHT))
        screen.fill((50, 50, 50))  # optional background
        # draw player
        pygame.draw.rect(screen, (0, 255, 0), (self.player.x, self.player.y, self.player.w, self.player.h))
        # draw obstacles
        for o in self.obstacles:
            pygame.draw.rect(screen, (255, 0, 0), (o.x, o.y, o.w, o.h))
        pygame.display.flip()

def run_ai(model_path):
    model = PPO.load(model_path)
    env = GDEnv()
    obs = env.reset()
    running = True
    clock = pygame.time.Clock()
    while running:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action[0])
        env.render()
        clock.tick(60)
        if done:
            print(f"Game over! Score: {env.score}")
            obs = env.reset()

if __name__ == "__main__":
    run_ai(r"G:\test\ai-geometry-dash\geometry dash\geometry_dash_project\models\final_model.zip")
