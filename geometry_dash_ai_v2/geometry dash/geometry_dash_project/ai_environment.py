import sys
import os
import pygame
import numpy as np
import random
import colorsys
from geometry_dash_game import main, Player, Obstacle, WIDTH, HEIGHT, FPS, GROUND_COLOR, PLAYER_COLOR, PLAYER_W, PLAYER_H, HITBOX_SCALE, DEFAULT_MUSIC_DIR, DEFAULT_JUMP_MP3, HIGHSCORE_FILE, START_SPAWN_MIN, START_SPAWN_MAX, BASE_SPEED, LEVEL_DURATION, SPEED_INCREASE_PER_LEVEL, SPAWN_MIN_DECREASE, SPAWN_MAX_DECREASE, SPAWN_MIN_FLOOR, SPAWN_MAX_FLOOR, GROUP_SIZES, GROUP_INTERNAL_GAP, SPIKE_CHANCE, PARTICLE_COUNT_BASE, PARTICLE_LIFE, LEVEL_HUE_SHIFT, RAINBOW_PRESETS, DEFAULT_RAINBOW_STYLE
import audio  # Assuming audio module is available or create a placeholder
import visuals  # Assuming visuals module is available or create a placeholder

class GeometryDashEnv:
    def __init__(self, use_rl=True, no_audio=False):
        self.use_rl = use_rl
        self.no_audio = no_audio
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AI Playing Geometry Dash")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.player = None
        self.obstacles = []
        self.spawn_timer = 0
        self.speed = BASE_SPEED
        self.score = 0
        self.game_over = False
        self.elapsed_time = 0.0
        self.level = 0
        self.last_group_right_x = -9999
        self.highscore = self._load_highscore()
        self.particles = []
        self.hue_boost = 0.0
        self.bg_image = None
        self.player_tex = None
        self.spike_surf = None
        self.ground_surf = None
        self.pre_surf1, self.pre_surf2 = (None, None)
        self.jump_sound = None
        self.music = None
        self.running = True

        self._load_assets()
        self._setup_audio()
        self.reset()

    def _load_highscore(self):
        try:
            with open(HIGHSCORE_FILE, 'r') as f:
                return int(f.read().strip() or 0)
        except Exception:
            return 0

    def _save_highscore(self, value):
        try:
            with open(HIGHSCORE_FILE, 'w') as f:
                f.write(str(int(value)))
        except Exception:
            pass

    def _load_assets(self):
        images_dir = os.path.join(os.path.dirname(__file__), 'images')

        def collect_recursive(subfolder):
            res = []
            base = os.path.join(images_dir, subfolder)
            if os.path.isdir(base):
                for root, _, files in os.walk(base):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                            res.append(os.path.join(root, f))
            return res

        bg_files = collect_recursive('background')
        ground_files = collect_recursive('floor')
        obstacle_files = collect_recursive('obstacles')
        block_files = collect_recursive('blocks')

        if bg_files:
            bg_image = pygame.image.load(random.choice(bg_files)).convert()
            scale = max(WIDTH / bg_image.get_width(), HEIGHT / bg_image.get_height())
            new_w, new_h = int(bg_image.get_width() * scale), int(bg_image.get_height() * scale)
            self.bg_image = pygame.transform.scale(bg_image, (new_w, new_h))
        if ground_files:
            self.ground_surf = pygame.image.load(ground_files[0]).convert()
        spike_candidate = next((p for p in obstacle_files if 'spike' in os.path.basename(p).lower()), obstacle_files[0] if obstacle_files else None)
        if spike_candidate:
            self.spike_surf = pygame.image.load(spike_candidate).convert_alpha()
        if block_files:
            block_path = block_files[0]
            try:
                block_surf = pygame.image.load(block_path).convert_alpha()
                self.player_tex = pygame.transform.scale(block_surf, (PLAYER_W, PLAYER_H))
            except Exception:
                self.player_tex = None

        if hasattr(visuals, 'create_rainbow_surfaces'):
            try:
                self.pre_surf1, self.pre_surf2 = visuals.create_rainbow_surfaces(WIDTH, HEIGHT, DEFAULT_RAINBOW_STYLE)
            except Exception:
                self.pre_surf1, self.pre_surf2 = (None, None)

    def _setup_audio(self):
        if not self.no_audio:
            self.jump_sound = self._load_jump_sound()
            self.music = audio.MusicManager(DEFAULT_MUSIC_DIR, enabled=True)
            try:
                self.music.shuffle_and_start()
            except Exception:
                pass

    def _load_jump_sound(self):
        try:
            if os.path.exists(DEFAULT_JUMP_MP3):
                return pygame.mixer.Sound(DEFAULT_JUMP_MP3)
            path = os.path.join(os.path.dirname(__file__), 'jump.wav')
            if os.path.exists(path):
                return pygame.mixer.Sound(path)
        except Exception:
            return None

    def reset(self):
        self.game_over = False
        self.score = 0
        self.elapsed_time = 0.0
        self.level = 0
        self.speed = BASE_SPEED
        self.obstacles = []
        self.spawn_timer = 0
        self.last_group_right_x = -9999
        self.particles = []
        self.hue_boost = 0.0
        self.player = Player()
        if self.music and not self.no_audio:
            try:
                self.music.shuffle_and_start()
            except Exception:
                pass
        return self._get_state()

    def _get_state(self):
        upcoming = [o for o in self.obstacles if o.x + o.w > self.player.x]
        default_obj = Obstacle(WIDTH + 100)
        closest = min(upcoming, key=lambda o: o.x, default=default_obj)
        return np.array([self.player.y / HEIGHT, (closest.x - self.player.x) / WIDTH, closest.y / HEIGHT], dtype=np.float32)

    def _simulate_key_event(self, key, down=True):
        event_type = pygame.KEYDOWN if down else pygame.KEYUP
        event = pygame.event.Event(event_type, key=key)
        pygame.event.post(event)

    def step(self, action):
        reward = 0
        done = False

        # Simulate player input based on AI action (0: no jump, 1: jump)
        if action == 1:
            self._simulate_key_event(pygame.K_SPACE, down=True)
            if self.player.on_ground() and self.jump_sound and not self.no_audio:
                self.jump_sound.play()
        else:
            self._simulate_key_event(pygame.K_SPACE, down=False)

        # Advance the game state by one frame
        dt = 1.0 / FPS
        self.player.step(dt)
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            group_count = random.choice(GROUP_SIZES)
            group_w = self.player.w
            group_h = self.player.h
            x_start = WIDTH + 20
            chosen_group_gap = random.randint(self.player.w, self.player.w * 2)
            x_start = max(x_start, self.last_group_right_x + chosen_group_gap)
            for i in range(group_count):
                x_pos = x_start + i * (group_w + GROUP_INTERNAL_GAP)
                kind = 'spike' if random.random() < SPIKE_CHANCE else 'normal'
                o = Obstacle(x_pos, w=group_w, h=group_h, kind=kind)
                if kind == 'spike' and self.spike_surf:
                    o.tex = pygame.transform.scale(self.spike_surf, (o.w, o.h))
                self.obstacles.append(o)
            self.last_group_right_x = x_start + group_count * group_w
            self.spawn_timer = random.randint(START_SPAWN_MIN, START_SPAWN_MAX)

        speed_s = self.speed * FPS
        for o in self.obstacles:
            o.step(speed_s * dt)
        self.obstacles = [o for o in self.obstacles if o.x + o.w > -50]

        hit_w = int(self.player.w * HITBOX_SCALE)
        hit_h = int(self.player.h * HITBOX_SCALE)
        hit_x = int(self.player.x + (self.player.w - hit_w) / 2)
        hit_y = int(self.player.y + (self.player.h - hit_h) / 2)
        for o in self.obstacles:
            if hit_x < o.x + o.w and hit_x + hit_w > o.x and hit_y < o.y + o.h and hit_y + hit_h > o.y:
                reward = -10
                done = True
                if self.score > self.highscore:
                    self._save_highscore(self.score)
                break

        for o in self.obstacles:
            if not getattr(o, 'cleared', False) and o.x + o.w < self.player.x:
                o.cleared = True
                self.score += 1
                reward = 1

        self.elapsed_time += dt
        if self.elapsed_time >= (self.level + 1) * LEVEL_DURATION:
            self.level += 1
            self.speed += SPEED_INCREASE_PER_LEVEL
            self.spawn_timer = random.randint(max(START_SPAWN_MIN - (self.level * SPAWN_MIN_DECREASE), SPAWN_MIN_FLOOR),
                                            max(START_SPAWN_MAX - (self.level * SPAWN_MAX_DECREASE), SPAWN_MAX_FLOOR))
            for _ in range(PARTICLE_COUNT_BASE + self.level * 8):
                self.particles.append({'x': random.uniform(0, WIDTH), 'y': random.uniform(0, HEIGHT * 0.35),
                                    'vx': random.uniform(-1, 1) * random.uniform(60, 320),
                                    'vy': random.uniform(-100, -20), 'life': random.uniform(PARTICLE_LIFE * 0.8, PARTICLE_LIFE),
                                    'max': PARTICLE_LIFE, 'color': (220, 220, 255), 'size': random.uniform(3, 8)})
            if self.music and not self.no_audio:
                try:
                    self.music.shuffle_and_start()
                except Exception:
                    pass

        self.render()
        return self._get_state(), reward, done

    def render(self):
        base_hue = (self.elapsed_time * 0.05) % 1.0
        hue_boost = max(0.0, self.hue_boost - 1.0 / FPS * 0.4)
        hue = (base_hue + hue_boost) % 1.0
        rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.2, 0.12))

        if self.bg_image:
            self.screen.blit(self.bg_image, ((WIDTH - self.bg_image.get_width()) // 2, (HEIGHT - self.bg_image.get_height()) // 2))
        else:
            self.screen.fill(rgb)
            if self.pre_surf1 and self.pre_surf2:
                offset1 = int((self.elapsed_time * 0.10 * WIDTH) % WIDTH)
                offset2 = int((self.elapsed_time * 0.18 * WIDTH) % WIDTH)
                self.screen.blit(self.pre_surf1, (-offset1, 0), (0, 0, WIDTH, HEIGHT))
                self.screen.blit(self.pre_surf1, (WIDTH - offset1, 0), (WIDTH, 0, WIDTH, HEIGHT))
                self.screen.blit(self.pre_surf2, (-offset2, 0), (0, 0, WIDTH, HEIGHT))
                self.screen.blit(self.pre_surf2, (WIDTH - offset2, 0), (WIDTH, 0, WIDTH, HEIGHT))

        if self.ground_surf:
            gw, gh = self.ground_surf.get_size()
            for tx in range(0, WIDTH, gw):
                self.screen.blit(self.ground_surf, (tx, HEIGHT - 80))
        else:
            pygame.draw.rect(self.screen, GROUND_COLOR, (0, HEIGHT - 80, WIDTH, 80))

        if self.player_tex:
            self.screen.blit(self.player_tex, (self.player.x, int(self.player.y)))
        else:
            pygame.draw.rect(self.screen, PLAYER_COLOR, (self.player.x, int(self.player.y), self.player.w, self.player.h))

        for o in self.obstacles:
            ox, oy, ow, oh = int(o.x), int(o.y), o.w, o.h
            if o.tex:
                self.screen.blit(o.tex, (ox, oy))
                p1, p2, p3 = (ox, oy + oh), (ox + ow, oy + oh), (ox + ow // 2, oy)
                pygame.draw.polygon(self.screen, (0, 0, 0), [p1, p2, p3], width=5)
            else:
                p1, p2, p3 = (ox, oy + oh), (ox + ow, oy + oh), (ox + ow // 2, oy)
                pygame.draw.polygon(self.screen, (120, 120, 120), [p1, p2, p3])
                pygame.draw.polygon(self.screen, (0, 0, 0), [p1, p2, p3], width=5)

        new_particles = []
        for p in self.particles:
            p['life'] -= 1.0 / FPS
            if p['life'] > 0:
                p['vy'] += 40 * (1.0 / FPS)
                p['x'] += p['vx'] * (1.0 / FPS)
                p['y'] += p['vy'] * (1.0 / FPS)
                alpha = max(0.0, p['life'] / p['max'])
                col = (int(p['color'][0] * alpha), int(p['color'][1] * alpha), int(p['color'][2] * alpha))
                pygame.draw.circle(self.screen, col, (int(p['x']), int(p['y'])), int(p['size']))
                new_particles.append(p)
        self.particles = new_particles

        self.screen.blit(self.font.render(f"Score: {self.score}", True, (255, 255, 255)), (10, 10))
        self.screen.blit(self.font.render(f"Highscore: {self.highscore}", True, (255, 255, 255)), (WIDTH - 220, 10))
        self.screen.blit(self.font.render(f"Level: {self.level}", True, (255, 255, 255)), (10, 50))

        pygame.display.flip()
        self.clock.tick(FPS)

    def run_with_ai(self):
        state = self.reset()
        total_reward = 0

        while self.running:
            # Simple AI policy: jump if an obstacle is close
            action = 1 if any(o.x - self.player.x < 200 for o in self.obstacles) else 0
            state, reward, done = self.step(action)
            total_reward += reward

            if done:
                print(f"Game Over! Score: {self.score}, Total Reward: {total_reward}")
                self.reset()
                total_reward = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    env = GeometryDashEnv(use_rl=True, no_audio=False)
    env.run_with_ai()