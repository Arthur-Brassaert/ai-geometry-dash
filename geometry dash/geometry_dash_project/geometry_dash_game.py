import sys
import os
import colorsys
import pygame
import random
import argparse
import config
import audio
import visuals
import numpy as np



WIDTH = 1000
HEIGHT = 600
FPS = 60

# Color customization: change these to pick different colors for ground and player
GROUND_COLOR = (34, 139, 34)   # default green (ForestGreen)
PLAYER_COLOR = (220, 60, 60)    # default red-ish

# Difficulty config (easy to tweak)
# Note: spawn intervals are in frames (not seconds). With FPS=60,
#  60 frames = 1.0 second. Consider converting to seconds if you
#  want FPS-independent timing.

# Starting spawn interval range (frames). Each new obstacle spawn
# chooses a random interval between these two values.
# Larger values -> fewer obstacles; smaller -> more frequent spawns.
START_SPAWN_MIN = 15         # min frames between spawns at level 0 (~0.3s @60FPS)
START_SPAWN_MAX = 30          # max frames between spawns at level 0 (~0.8s @60FPS) — reduced so groups appear closer

# Base obstacle speed in pixels per frame. Higher -> obstacles move faster.
BASE_SPEED = 6.0              # px/frame (≈360 px/s at 60 FPS)

# Level timing: how many seconds per difficulty level before increasing
# (e.g. 15.0 = level up every 15 seconds). Adjust to make progression
# faster or slower.
LEVEL_DURATION = 60.0         # seconds per level

# How much to add to speed per level (linear increment):
# speed = BASE_SPEED + level * SPEED_INCREASE_PER_LEVEL
SPEED_INCREASE_PER_LEVEL = 1.5

# How much to decrease the spawn_min / spawn_max (in frames) per level.
# These values are subtracted from the current spawn_min/max each level
# (floored by SPAWN_MIN_FLOOR / SPAWN_MAX_FLOOR below).
SPAWN_MIN_DECREASE = 8        # frames removed from spawn_min per level
SPAWN_MAX_DECREASE = 12       # frames removed from spawn_max per level

# Floors (minimums) for spawn intervals so spawns don't become impossibly fast
# (safety limits). Keep these > ~10 to avoid huge spawn bursts.
SPAWN_MIN_FLOOR = 12
SPAWN_MAX_FLOOR = 20

# -------------------------------
# Instelbare spel-parameters
# Alle parameters hieronder kort beschreven in het Nederlands
# -------------------------------
# Speler (blok) afmetingen
PLAYER_W = 40            # breedte van het spelersblok (pixels)
PLAYER_H = 40            # hoogte van het spelersblok (pixels)

# Physics / springen
GRAVITY = 1.0            # valversnelling (pixels per frame)
JUMP_STRENGTH = -16     # verlaagde sprongimpuls (minder hoog)
MAX_JUMP_HOLD = 0.12    # kortere houd-tijd zodat lange sprong minder uitgesproken is
HOLD_GRAVITY = 0.75     # iets hogere hold-gravity zodat vasthouden minder extra lift geeft
AUTO_JUMP_ON_LAND = True # als True, vasthouden springt meteen opnieuw bij landen

# Groepen en tussenruimte
GAP_MIN = 4             # minimale interne gap tussen blokken in dezelfde groep (px)
GAP_MAX = 9           # maximale interne gap tussen blokken in dezelfde groep (px)
MIN_GROUP_GAP_MULT = 1  # reduce multiplier so groups spawn closer together (player width * 1)
# extra instellingen voor willekeurige tussenafstand tussen groepen (pixels)
# het daadwerkelijke gap wordt willekeurig gekozen tussen GROUP_GAP_MIN en GROUP_GAP_MAX
GROUP_GAP_MIN = PLAYER_W * MIN_GROUP_GAP_MULT          # minimale gap tussen groepen (px)
GROUP_GAP_MAX = PLAYER_W * (MIN_GROUP_GAP_MULT + 1)      # maximale gap tussen groepen (px)
# Vooraf gedefinieerde groepsgroottes: mogelijke groepen die bij elkaar staan
GROUP_SIZES = [1, 2, 3]   # mogelijke groepsgroottes; spawn kiest random één
# interne gap binnen groep: 0 zodat blokken direct naast elkaar staan
GROUP_INTERNAL_GAP = 0
# Chance that a spawned obstacle is a spike (0.0 - 1.0)
SPIKE_CHANCE = 0.25
# -------------------------------
# Hitbox instellingen
HITBOX_SCALE = 0.5  # schaalfactor voor de speler-hitbox (1.0 = volle grootte)
# Highscore bestandsnaam (in dezelfde map als dit script)
HIGHSCORE_FILE = os.path.join(os.path.dirname(__file__), 'highscore.txt')

# ---------------------------
# File-level configurable parameters
# Keep all tunables and paths here so they're easy to find and edit
# ---------------------------
# Particles / level-up visuals
PARTICLE_COUNT_BASE = 40
PARTICLE_LIFE = 2.0
LEVEL_HUE_SHIFT = 0.35

# Rainbow presets (name -> (col_w, sat, val, speed1, speed2, alpha1, alpha2))
RAINBOW_PRESETS = {
    'soft': (6, 0.65, 0.28, 0.06, 0.12, 20, 36),
    'pastel': (8, 0.45, 0.55, 0.04, 0.08, 16, 28),
    'vibrant': (4, 0.95, 0.35, 0.10, 0.18, 36, 64),
}
DEFAULT_RAINBOW_STYLE = 'vibrant'

# Environment defaults (can be overridden by env vars)
DEFAULT_GD_RAINBOW = '1'   # set GD_RAINBOW=0 to disable rainbow
DEFAULT_GD_MUSIC = '1'     # set GD_MUSIC=0 to disable music
DEFAULT_GD_RAINBOW_STYLE = DEFAULT_RAINBOW_STYLE

# Paths for sounds (adjust if your environment differs)
DEFAULT_MUSIC_DIR = r"E:\vives\2 de jaar\AI Programming\.geometrydash_venv\sounds\level songs"
DEFAULT_JUMP_MP3 = r"E:\vives\2 de jaar\AI Programming\.geometrydash_venv\sounds\sound effects\jump.mp3"

def load_jump_sound():
    """Try to load the provided jump MP3, fall back to local jump.wav, or synthesize a short tone."""
    try:
        if os.path.exists(DEFAULT_JUMP_MP3):
            try:
                return pygame.mixer.Sound(DEFAULT_JUMP_MP3)
            except Exception:
                pass
        path = os.path.join(os.path.dirname(__file__), 'jump.wav')
        if os.path.exists(path):
            try:
                return pygame.mixer.Sound(path)
            except Exception:
                pass
        # synthesize using numpy if available
        try:
            import numpy as _np
            freq = 880.0
            duration = 0.08
            samplerate = 22050
            t = _np.linspace(0, duration, int(samplerate * duration), False)
            tone = (0.3 * _np.sin(2 * _np.pi * freq * t)).astype(_np.float32)
            arr = _np.int16(tone * 32767)
            return pygame.mixer.Sound(arr)
        except Exception:
            return None
    except Exception:
        return None



class Player:
    def __init__(self):
        self.w = PLAYER_W
        self.h = PLAYER_H
        self.x = int(WIDTH * 0.2)
        self.y = HEIGHT - 80 - self.h
        # velocity is now in pixels/second
        self.v = 0.0
        # convert gravity/jump (originally expressed per-frame) to per-second
        # gravity (px/frame^2) -> gravity_s (px/s^2) requires multiplying by FPS^2
        self.gravity_s = GRAVITY * (FPS ** 2)
        # initial jump impulse in px/s (negative = upward)
        self.jump_strength_s = JUMP_STRENGTH * FPS
        # variable jump support
        self.jump_held = False      # whether jump key is currently held
        self.jump_time = 0.0        # how long jump has been held (seconds)
        # allow slightly longer hold for extra height (seconds)
        self.max_jump_hold = MAX_JUMP_HOLD
        # during hold we apply a reduced gravity (minder dan normale gravity)
        # hold gravity also converted to px/s^2
        self.hold_gravity_s = HOLD_GRAVITY * (FPS ** 2)
        self.auto_jump_on_land = AUTO_JUMP_ON_LAND

    def jump(self):
        if self.on_ground():
            self.v = self.jump_strength_s

    def on_ground(self):
        return self.y >= HEIGHT - 80 - self.h - 1

    def step(self, dt: float):
        """Advance physics one frame. dt is seconds since last frame (for timing jump holds)."""
        # choose gravity depending on whether jump is held and still within hold window
        if self.jump_held and self.v < 0 and self.jump_time < self.max_jump_hold:
            # while holding, we increment jump_time and apply reduced gravity
            self.jump_time += dt
            effective_gravity_s = self.hold_gravity_s
        else:
            effective_gravity_s = self.gravity_s

        # integrate velocity (px/s) and position
        self.v += effective_gravity_s * dt
        self.y += self.v * dt

        # landing
        if self.y > HEIGHT - 80 - self.h:
            self.y = HEIGHT - 80 - self.h
            self.v = 0
            # reset jump_time on landing
            self.jump_time = 0.0
            # if player is holding jump and auto_jump is enabled, immediately jump again
            if self.jump_held and self.auto_jump_on_land:
                self.v = self.jump_strength_s


class Obstacle:
    def __init__(self, x, w=None, h=None, kind='normal'):
        # allow passing a fixed size so groups can share the same dimensions
        self.w = w if w is not None else random.randint(20, 60)
        self.h = h if h is not None else random.randint(30, 100)
        self.x = x
        self.y = HEIGHT - 80 - self.h
        self.kind = kind
        # optional cached textured surface for spikes
        self.tex = None
        # whether this obstacle has already been counted as 'cleared' by the player
        self.cleared = False

    def step(self, speed, dt: float):
        # speed is pixels per second, dt is seconds
        self.x -= speed * dt


def main(use_rl: bool = False, model_path: str | None = None, no_audio: bool = False):
    """Main entry point for the Geometry Dash demo.

    This function initializes pygame, loads audio/visual helpers, and runs
    the primary game loop. The file keeps a small, procedural game loop for
    demonstration and tuning.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)
    # helper to draw black text with a white outline (makes text readable on images)
    def draw_text_outline(surface, text, font_obj, x, y, fg=(0, 0, 0), outline=(255, 255, 255)):
        """Draw text with an outline by blitting the outline color around the main text."""
        try:
            outline_surf = font_obj.render(text, True, outline)
            text_surf = font_obj.render(text, True, fg)
        except Exception:
            # fallback simple render
            text_surf = font_obj.render(text, True, fg)
            surface.blit(text_surf, (x, y))
            return
        # draw outline in 8 surrounding pixels
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            surface.blit(outline_surf, (x + dx, y + dy))
        # draw main text
        surface.blit(text_surf, (x, y))
    # Load jump sound via audio helper (skip when running with --no-audio)
    if no_audio:
        jump_sound = None
    else:
        jump_sound = audio.load_jump_sound()

    # helper: load/save highscore
    def load_highscore():
        try:
            with open(HIGHSCORE_FILE, 'r') as f:
                return int(f.read().strip() or 0)
        except Exception:
            return 0

    def save_highscore(value: int):
        try:
            with open(HIGHSCORE_FILE, 'w') as f:
                f.write(str(int(value)))
        except Exception:
            pass

    highscore = load_highscore()

    # Use file-level defaults and env-vars for settings
    # particles: ephemeral visual particles used on level-up
    particles = []
    # hue_boost: visual boost applied briefly after level-up
    hue_boost = 0.0

    # Settings from config module
    GD_RAINBOW_ENABLED = config.GD_RAINBOW
    RAINBOW_STYLE = config.GD_RAINBOW_STYLE
    RAINBOW_COL_W, RAINBOW_SAT, RAINBOW_VAL, RAINBOW_SPEED1, RAINBOW_SPEED2, RAINBOW_ALPHA1, RAINBOW_ALPHA2 = config.get_rainbow_params(RAINBOW_STYLE)

    GD_MUSIC_ENABLED = config.GD_MUSIC
    MUSIC_DIR = config.DEFAULT_MUSIC_DIR
    # honor CLI flag to disable audio entirely
    if no_audio:
        GD_MUSIC_ENABLED = False

    # music manager: collects tracks from MUSIC_DIR and controls playback
    music = audio.MusicManager(MUSIC_DIR, enabled=GD_MUSIC_ENABLED)
    # start playlist
    try:
        music.shuffle_and_start()
    except Exception as e:
        print(f"Music initialization error: {e}")

    # initialize volumes (from settings.json if available)
    settings_path = os.path.join(os.path.dirname(__file__), 'settings.json')

    def load_settings():
        try:
            import json
            if os.path.exists(settings_path):
                with open(settings_path, 'r', encoding='utf-8') as sf:
                    data = json.load(sf)
                    return data
        except Exception as e:
            print(f"Settings load error: {e}")
        return {}

    def save_settings(data: dict):
        try:
            import json
            with open(settings_path, 'w', encoding='utf-8') as sf:
                json.dump(data, sf)
        except Exception as e:
            print(f"Settings save error: {e}")

    # load saved settings (volumes etc.)
    saved = load_settings()
    music_vol = float(saved.get('music_volume', getattr(config, 'DEFAULT_VOLUME', 0.5)))
    sfx_vol = float(saved.get('sfx_volume', getattr(config, 'DEFAULT_VOLUME', 0.5)))
    try:
        music.set_volume(music_vol)
    except Exception as e:
        print(f"Music volume set error: {e}")
    try:
        if jump_sound is not None and hasattr(jump_sound, 'set_volume'):
            jump_sound.set_volume(sfx_vol)
    except Exception as e:
        print(f"SFX volume set error: {e}")

    # pre-render rainbow surfaces once to save per-frame CPU
    pre_surf1, pre_surf2 = (None, None)
    # try to load background/floor/obstacle images from structured images/ subfolders
    bg_image = None
    try:
        images_dir = os.path.join(os.path.dirname(__file__), 'images')

        def collect_recursive(subfolder):
            res = []
            base = os.path.join(images_dir, subfolder)
            if not os.path.isdir(base):
                return res
            # Supported image extensions (be permissive)
            exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
            for root, _, files in os.walk(base):
                for f in files:
                    if f.lower().endswith(exts):
                        res.append(os.path.join(root, f))
            return res

        # Use folder names from config so they can be changed centrally
        # Be flexible with singular/plural folder names (e.g., 'background' vs 'backgrounds')
        def resolve_folder(name):
            candidates = [name]
            if not name.endswith('s'):
                candidates.append(name + 's')
            else:
                candidates.append(name[:-1])
            for c in candidates:
                if os.path.isdir(os.path.join(images_dir, c)):
                    return c
            return name

        bg_folder = resolve_folder(config.IMAGE_SUBFOLDERS.get('background', 'background'))
        floor_folder = resolve_folder(config.IMAGE_SUBFOLDERS.get('floor', 'floor'))
        obstacles_folder = resolve_folder(config.IMAGE_SUBFOLDERS.get('obstacles', 'obstacles'))

        bg_files = collect_recursive(bg_folder)
        ground_files = collect_recursive(floor_folder)
        obstacle_files = collect_recursive(obstacles_folder)

        # If structured folders are empty, fall back to legacy behavior:
        # scan the images/ root recursively so files placed directly under images/
        # (or in other non-structured folders) are still discovered. This keeps
        # strict-folder preference but restores compatibility for existing projects.
        if not (bg_files or ground_files or obstacle_files):
            legacy_files = []
            for root, _, files in os.walk(images_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        legacy_files.append(os.path.join(root, f))
            # heuristically assign legacy files into categories by filename
            # NOTE: per user request, only files from images/obstacles/ are used for obstacles.
            for p in legacy_files:
                lname = os.path.basename(p).lower()
                if any(k in lname for k in ('background', 'bg', 'back', 'menu')):
                    bg_files.append(p)
                elif any(k in lname for k in ('ground', 'floor', 'land')):
                    ground_files.append(p)
                else:
                    # ignore legacy files for obstacles — only images/obstacles/ will be used
                    print(f"[bg] Legacy file ignored for obstacles: {p}")

            print('[bg] Legacy scan assigned files to categories (obstacles ignored):', {'bg': bg_files, 'ground': ground_files})

            # debug: print resolved folder names and their discovered files
            print('[bg] Resolved folders -> background:', bg_folder, ' floor:', floor_folder, ' obstacles:', obstacles_folder)
            print('[bg] Found background files:', bg_files)
            print('[bg] Found ground files:', ground_files)
            print('[bg] Found obstacle files:', obstacle_files)
            # Extra debug: show raw directory listing for obstacles folder if it exists
            obs_dir = os.path.join(images_dir, obstacles_folder)
            try:
                raw = os.listdir(obs_dir) if os.path.isdir(obs_dir) else []
                print('[bg] Raw obstacles directory listing:', raw)
            except Exception as e:
                print('[bg] Failed to list obstacles directory:', e)

        bg_images = []
        loaded_names = []
        for path in bg_files:
            try:
                surf = pygame.image.load(path)
                try:
                    surf = surf.convert_alpha()
                except Exception:
                    surf = surf.convert()
                iw, ih = surf.get_size()
                if iw <= 0 or ih <= 0:
                    print(f"[bg] Skipping zero-size image: {path}")
                    continue
                scale = max(WIDTH / iw, HEIGHT / ih)
                new_w = int(iw * scale)
                new_h = int(ih * scale)
                try:
                    surf = pygame.transform.smoothscale(surf, (new_w, new_h))
                except Exception:
                    surf = pygame.transform.scale(surf, (new_w, new_h))
                final = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                bx = (WIDTH - new_w) // 2
                by = (HEIGHT - new_h) // 2
                final.blit(surf, (bx, by))
                bg_images.append(final)
                loaded_names.append(os.path.relpath(path, images_dir))
                print(f"[bg] Loaded background: {path}")
            except Exception as ex:
                print(f"[bg] Failed to load background {path}: {ex}")
                continue

        if bg_images:
            idx = random.randrange(len(bg_images))
            bg_image = bg_images[idx]
            try:
                chosen_name = loaded_names[idx]
                print(f"[bg] Chosen background: {chosen_name}")
            except Exception:
                pass
        else:
            bg_image = None
            if GD_RAINBOW_ENABLED:
                print('[bg] No background images found — using rainbow fallback')
            else:
                print('[bg] No background images found — using solid color background')

        # pick a ground texture (first available) to tile across floor
        ground_surf = None
        if ground_files:
            gpath = ground_files[0]
            try:
                gs = pygame.image.load(gpath)
                try:
                    gs = gs.convert()
                except Exception:
                    pass
                ground_surf = gs
                print(f"[bg] Loaded ground texture: {gpath}")
            except Exception as ex:
                print(f"[bg] Failed to load ground texture {gpath}: {ex}")
        else:
            ground_surf = None
            print('[bg] No ground texture found — using solid color ground')

        # load player block textures from images/blocks (optional)
        block_surf = None
        try:
            blocks_folder = resolve_folder('blocks')
            block_files = collect_recursive(blocks_folder)
            if block_files:
                bpath = block_files[0]
                try:
                    bs = pygame.image.load(bpath)
                    try:
                        bs = bs.convert_alpha()
                    except Exception:
                        bs = bs.convert()
                    block_surf = bs
                    print(f"[bg] Loaded player block texture: {bpath}")
                except Exception as ex:
                    print(f"[bg] Failed to load player block texture {bpath}: {ex}")
        except Exception:
            block_surf = None

        # load an obstacle/spike image if present (prioritize filenames with 'spike')
        spike_surf = None
        spike_candidate = None
        for p in obstacle_files:
            if 'spike' in os.path.basename(p).lower():
                spike_candidate = p
                break
        if spike_candidate is None and obstacle_files:
            spike_candidate = obstacle_files[0]
        if spike_candidate:
            try:
                ss = pygame.image.load(spike_candidate)
                try:
                    ss = ss.convert_alpha()
                except Exception:
                    ss = ss.convert()
                spike_surf = ss
                try:
                    sw, sh = spike_surf.get_size()
                    print(f"[bg] Loaded obstacle image (used for spikes if needed): {spike_candidate} size=({sw},{sh})")
                except Exception:
                    print(f"[bg] Loaded obstacle image (used for spikes if needed): {spike_candidate}")
            except Exception as ex:
                print(f"[bg] Failed to load obstacle image {spike_candidate}: {ex}")
        else:
            # No obstacle-specific image found — per user preference we only
            # use files from images/obstacles for obstacles. Spikes remain plain triangles.
            print('[bg] No obstacle images found in images/obstacles — spikes will be drawn as triangles')
    except Exception as e:
        print(f"Image loading error: {e}")
        bg_image = None

    if GD_RAINBOW_ENABLED:
        try:
            pre_surf1, pre_surf2 = visuals.create_rainbow_surfaces(WIDTH, HEIGHT, RAINBOW_STYLE)
        except Exception as e:
            print(f"Rainbow surface creation error: {e}")
            pre_surf1, pre_surf2 = (None, None)

    # helper to prepare a textured triangular surface for spikes
    def prepare_spike_texture(spike_src, ow, oh):
        try:
            if spike_src is None:
                return None
            sw, sh = spike_src.get_size()
            scale = max(ow / max(1, sw), oh / max(1, sh))
            tex_w = max(1, int(sw * scale))
            tex_h = max(1, int(sh * scale))
            # use faster scaling when preparing at spawn-time
            tex = pygame.transform.scale(spike_src, (tex_w, tex_h))
            tri_surf = pygame.Surface((ow, oh), pygame.SRCALPHA)
            tx = (ow - tex_w) // 2
            ty = (oh - tex_h) // 2
            tri_surf.blit(tex, (tx, ty))
            mask = pygame.Surface((ow, oh), pygame.SRCALPHA)
            p1 = (0, oh)
            p2 = (ow, oh)
            p3 = (ow // 2, 0)
            pygame.draw.polygon(mask, (255, 255, 255, 255), [p1, p2, p3])
            tri_surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            return tri_surf
        except Exception as e:
            print(f"Spike texture preparation error: {e}")
            return None


    # initial game state
    player = Player()
    # optional cached scaled player block surface
    player_tex = None
    obstacles = []
    # spawn interval range (in frames)
    spawn_min = START_SPAWN_MIN
    spawn_max = START_SPAWN_MAX
    # make a group spawn immediately at game start
    spawn_timer = 0
    base_speed = BASE_SPEED
    speed = base_speed
    score = 0
    running = True
    game_over = False

    # start music for level 0
    try:
        music.shuffle_and_start()
    except Exception as e:
        print(f"Music start error: {e}")

    # difficulty scaling
    elapsed_time = 0.0  # seconds
    level = 0
    level_duration = LEVEL_DURATION
    speed_increase_per_level = SPEED_INCREASE_PER_LEVEL
    spawn_min_decrease = SPAWN_MIN_DECREASE
    spawn_max_decrease = SPAWN_MAX_DECREASE
    spawn_min_floor = SPAWN_MIN_FLOOR
    spawn_max_floor = SPAWN_MAX_FLOOR

    # track the rightmost x coordinate of the last spawned group to enforce minimum gap
    last_group_right_x = -9999
    min_group_gap = player.w * MIN_GROUP_GAP_MULT  # veilige tussenafstand tussen groepen

    print('[bg] entering main loop')

    while running:
        # dragging state for the volume slider
        if 'dragging' not in locals():
            dragging = False
        # dt measured in seconds
        dt = clock.tick(FPS) / 1000.0
        if dt <= 0:
            dt = 1.0 / FPS
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == audio.MUSIC_END_EVENT:
                try:
                    music.handle_music_end_event(event)
                except Exception as e:
                    print(f"Music end event error: {e}")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # start holding jump; if on ground, apply initial impulse
                    player.jump_held = True
                    if not game_over and player.on_ground():
                        player.jump()
                        # play jump sound if available
                        try:
                            if jump_sound is not None:
                                jump_sound.play()
                        except Exception as e:
                            print(f"Jump sound play error: {e}")
                if event.key == pygame.K_r:
                    # restart
                    player = Player()
                    obstacles = []
                    spawn_min = START_SPAWN_MIN
                    spawn_max = START_SPAWN_MAX
                    # make spawn behave like initial run: force immediate group spawn
                    spawn_timer = 0
                    # reset the group-tracking sentinel so gaps are computed fresh
                    last_group_right_x = -9999
                    # clear any cached scaled player texture so the player is regenerated
                    try:
                        player_tex = None
                    except Exception:
                        pass
                    base_speed = BASE_SPEED
                    speed = base_speed
                    score = 0
                    elapsed_time = 0.0
                    level = 0
                    game_over = False
                    try:
                        music.shuffle_and_start()
                    except Exception as e:
                        print(f"Music restart error: {e}")
                    # pick a new background image on restart (if multiple loaded)
                    try:
                        if 'bg_images' in locals() and bg_images:
                            bg_image = random.choice(bg_images)
                    except Exception:
                        pass
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    # release jump
                    player.jump_held = False
                if event.key == pygame.K_m:
                    # toggle music pause
                    try:
                        music.toggle_pause()
                    except Exception as e:
                        print(f"Music toggle error: {e}")
                if event.key == pygame.K_RIGHT:
                    try:
                        music.play_next()
                    except Exception as e:
                        print(f"Music next error: {e}")
                # keyboard volume controls
                if event.key == pygame.K_UP:
                    # increase music volume
                    try:
                        n = min(1.0, music.get_volume() + 0.05)
                        music.set_volume(n)
                        saved['music_volume'] = float(n)
                        save_settings(saved)
                    except Exception as e:
                        print(f"Volume increase error: {e}")
                if event.key == pygame.K_DOWN:
                    try:
                        n = max(0.0, music.get_volume() - 0.05)
                        music.set_volume(n)
                        saved['music_volume'] = float(n)
                        save_settings(saved)
                    except Exception as e:
                        print(f"Volume decrease error: {e}")
                # Shift+Up/Down for SFX
                mods = pygame.key.get_mods()
                if mods & pygame.KMOD_SHIFT:
                    if event.key == pygame.K_UP:
                        try:
                            s = min(1.0, float(saved.get('sfx_volume', sfx_vol)) + 0.05)
                            saved['sfx_volume'] = s
                            if jump_sound is not None and hasattr(jump_sound, 'set_volume'):
                                jump_sound.set_volume(s)
                            save_settings(saved)
                        except Exception as e:
                            print(f"SFX volume increase error: {e}")
                    if event.key == pygame.K_DOWN:
                        try:
                            s = max(0.0, float(saved.get('sfx_volume', sfx_vol)) - 0.05)
                            saved['sfx_volume'] = s
                            if jump_sound is not None and hasattr(jump_sound, 'set_volume'):
                                jump_sound.set_volume(s)
                            save_settings(saved)
                        except Exception as e:
                            print(f"SFX volume decrease error: {e}")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # clicking on vertical slider (right side)
                if event.button == 1:
                    mx, my = event.pos
                    # music slider (rightmost)
                    slider_w = 14
                    slider_h = 180
                    slider_x = WIDTH - slider_w - 20
                    slider_y = HEIGHT // 2 - slider_h // 2
                    # sfx slider (left of music slider)
                    sfx_slider_x = slider_x - 36
                    sfx_slider_y = slider_y
                    if slider_x <= mx <= slider_x + slider_w and slider_y <= my <= slider_y + slider_h:
                        dragging = True
                        dragging_target = 'music'
                    elif sfx_slider_x <= mx <= sfx_slider_x + slider_w and sfx_slider_y <= my <= sfx_slider_y + slider_h:
                        dragging = True
                        dragging_target = 'sfx'
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
                    # if mouse released over control buttons, handle them
                    mx, my = event.pos
                    # play/pause button area
                    btn_w = 36
                    btn_h = 24
                    btn_x = slider_x - 8 - btn_w
                    btn_y = slider_y + slider_h + 12
                    if btn_x <= mx <= btn_x + btn_w and btn_y <= my <= btn_y + btn_h:
                        try:
                            music.toggle_pause()
                            # save volumes (persist current values)
                            try:
                                saved['music_volume'] = float(music.get_volume())
                                saved['sfx_volume'] = float(saved.get('sfx_volume', sfx_vol))
                                save_settings(saved)
                            except Exception as e:
                                print(f"Settings save error: {e}")
                        except Exception as e:
                            print(f"Music toggle error: {e}")
                    # next button (to the left of play/pause)
                    nbtn_x = btn_x - 8 - btn_w
                    if nbtn_x <= mx <= nbtn_x + btn_w and btn_y <= my <= btn_y + btn_h:
                        try:
                            music.play_next()
                            try:
                                saved['music_volume'] = float(music.get_volume())
                                saved['sfx_volume'] = float(saved.get('sfx_volume', sfx_vol))
                                save_settings(saved)
                            except Exception as e:
                                print(f"Settings save error: {e}")
                        except Exception as e:
                            print(f"Music next error: {e}")
            elif event.type == pygame.MOUSEMOTION:
                if 'dragging' in locals() and dragging:
                    mx, my = event.pos
                    slider_w = 14
                    slider_h = 180
                    slider_x = WIDTH - slider_w - 20
                    slider_y = HEIGHT // 2 - slider_h // 2
                    sfx_slider_x = slider_x - 36
                    sfx_slider_y = slider_y
                    # vertical: top = max volume (1.0), bottom = 0.0
                    rel = 1.0 - ((my - slider_y) / float(slider_h))
                    vol = max(0.0, min(1.0, rel))
                    if 'dragging_target' in locals() and dragging_target == 'music':
                        try:
                            music.set_volume(vol)
                        except Exception as e:
                            print(f"Music volume set error: {e}")
                        try:
                            if hasattr(jump_sound, 'set_volume') and jump_sound is not None:
                                # keep sfx as previously saved
                                pass
                        except Exception:
                            pass
                        try:
                            saved['music_volume'] = float(vol)
                            save_settings(saved)
                        except Exception as e:
                            print(f"Settings save error: {e}")
                    elif 'dragging_target' in locals() and dragging_target == 'sfx':
                        try:
                            saved['sfx_volume'] = float(vol)
                            if jump_sound is not None and hasattr(jump_sound, 'set_volume'):
                                jump_sound.set_volume(vol)
                            save_settings(saved)
                        except Exception as e:
                            print(f"SFX volume set error: {e}")

        if not game_over:
            elapsed_time += dt

            player.step(dt)
            spawn_timer -= 1
            if spawn_timer <= 0:
                # spawn a group of 1-3 obstacles with identical size (triangles)
                # choose a group size from predefined options (1,2,3)
                # for level 0 we prefer larger groups but still pick from GROUP_SIZES
                if level == 0:
                    group_count = random.choice([g for g in GROUP_SIZES if g >= 2])
                else:
                    group_count = random.choice(GROUP_SIZES)
                group_w = player.w
                group_h = player.h
                # internal gap between blocks in the same group (now zero = flush)
                gap_min, gap_max = GROUP_INTERNAL_GAP, GROUP_INTERNAL_GAP
                # determine x_start but ensure there is a randomized gap from last group
                desired_x = WIDTH + 20
                # choose a randomized gap between groups but no smaller than min_group_gap
                if level == 0:
                    # denser gaps for level 0 (more playable early game)
                    chosen_group_gap = random.randint(player.w, player.w * 2)
                else:
                    chosen_group_gap = random.randint(GROUP_GAP_MIN, GROUP_GAP_MAX)
                    chosen_group_gap = max(chosen_group_gap, min_group_gap)
                # add a small random offset so distances don't feel identical every spawn
                offset_max = int(player.w * 1.5)
                extra_offset = random.randint(-offset_max, offset_max)
                x_start_candidate = last_group_right_x + chosen_group_gap + extra_offset
                x_start = max(desired_x, x_start_candidate)
                # create group and update last_group_right_x
                group_right = x_start
                for i in range(group_count):
                    x_pos = x_start + i * (group_w + gap_min)
                    # randomly decide if this obstacle is a spike
                    kind = 'spike' if random.random() < SPIKE_CHANCE else 'normal'
                    obstacles.append(Obstacle(x_pos, w=group_w, h=group_h, kind=kind))
                    # if we have a spike texture, prepare a cached textured triangle for this obstacle
                    if kind == 'spike' and 'spike_surf' in locals() and spike_surf is not None:
                        try:
                            oref = obstacles[-1]
                            ow = oref.w
                            oh = oref.h
                            tri = prepare_spike_texture(spike_surf, ow, oh)
                            if tri is not None:
                                oref.tex = tri
                        except Exception as e:
                            print(f"[bg] Failed to prepare spike texture for obstacle: {e}")
                    group_right = max(group_right, x_pos + group_w)
                last_group_right_x = group_right
                # debug: report spawn info
                try:
                    xs = [int(o.x) for o in obstacles[-group_count:]]
                except Exception:
                    pass
                spawn_timer = random.randint(spawn_min, spawn_max)

            # move obstacles using speed in px/s (convert base_speed which is in px/frame)
            speed_s = speed * FPS  # convert px/frame -> px/s
            for o in obstacles:
                o.step(speed_s, dt)

            obstacles = [o for o in obstacles if o.x + o.w > -50]

            # Ensure level 0 is populated: sometimes no obstacles appear early —
            # force a group spawn near the right edge if none are present.
            try:
                if level == 0 and len(obstacles) == 0:
                    group_count = random.choice(GROUP_SIZES)
                    x_start = WIDTH + 20
                    gap_min = GROUP_INTERNAL_GAP
                    for i in range(group_count):
                        x_pos = x_start + i * (player.w + gap_min)
                        kind = 'spike' if random.random() < SPIKE_CHANCE else 'normal'
                        oref = Obstacle(x_pos, w=player.w, h=player.h, kind=kind)
                        # prepare spike texture cache if available
                        if kind == 'spike' and 'spike_surf' in locals() and spike_surf is not None:
                            try:
                                tri = prepare_spike_texture(spike_surf, oref.w, oref.h)
                                if tri is not None:
                                    oref.tex = tri
                            except Exception as e:
                                print(f"[bg] Failed to prepare spike texture for forced obstacle: {e}")
                        obstacles.append(oref)
                    try:
                        xs = [int(o.x) for o in obstacles[-group_count:]]
                        print(f"[bg] Forced spawn (level0): count={group_count} xs={xs} x_start={x_start}")
                    except Exception:
                        pass
            except Exception as e:
                print(f"Forced spawn error: {e}")

            # collision using a smaller, centered hitbox for the player
            hit_w = int(player.w * HITBOX_SCALE)
            hit_h = int(player.h * HITBOX_SCALE)
            hit_x = int(player.x + (player.w - hit_w) / 2)
            hit_y = int(player.y + (player.h - hit_h) / 2)
            collided = False
            for o in obstacles:
                if hit_x < o.x + o.w and hit_x + hit_w > o.x and hit_y < o.y + o.h and hit_y + hit_h > o.y:
                    collided = True
                    break
            if collided:
                game_over = True
                # update highscore when game ends
                try:
                    if score > highscore:
                        highscore = score
                        save_highscore(highscore)
                except Exception as e:
                    print(f"Highscore save error: {e}")

            # increment score when obstacles are cleared (pass left of the player)
            try:
                for o in obstacles:
                    # obstacle right edge
                    if not getattr(o, 'cleared', False) and (o.x + o.w) < player.x:
                        o.cleared = True
                        score += 1
            except Exception as e:
                print(f"Score update error: {e}")
                # fallback: if something goes wrong, keep frame-based scoring (defensive)
                score += 1

            # difficulty increase every level_duration seconds
            if elapsed_time >= (level + 1) * level_duration:
                level += 1
                # increase speed
                speed = base_speed + level * speed_increase_per_level
                # make spawns more frequent
                spawn_min = max(spawn_min_floor, spawn_min - spawn_min_decrease)
                spawn_max = max(spawn_max_floor, spawn_max - spawn_max_decrease)
                # --- level-up visual effects: hue boost + particles ---
                # temporary hue offset applied for a short time
                hue_boost = LEVEL_HUE_SHIFT
                # spawn particles across the top area (meer en groter)
                count = PARTICLE_COUNT_BASE + level * 8
                for _ in range(count):
                    px = random.uniform(0, WIDTH)
                    py = random.uniform(0, HEIGHT * 0.35)
                    speed_p = random.uniform(60, 320)
                    vx = random.uniform(-1, 1) * speed_p
                    vy = random.uniform(-100, -20)
                    life = random.uniform(PARTICLE_LIFE * 0.8, PARTICLE_LIFE)
                    color = (220, 220, 255)
                    size = random.uniform(3, 8)
                    particles.append({'x': px, 'y': py, 'vx': vx, 'vy': vy, 'life': life, 'max': life, 'color': color, 'size': size})
                # switch music to the appropriate level track
                try:
                    music.shuffle_and_start()
                except Exception as e:
                    print(f"Level up music error: {e}")

            # dynamic background: cycle hue over time for a subtle animated background
            base_hue = (elapsed_time * 0.05) % 1.0  # slow cycle
            # allow temporary hue boost on recent level-up (decays quickly)
            # decay hue_boost slower for a longer visible effect
            hue_boost = max(0.0, hue_boost - dt * 0.4)
            hue = (base_hue + hue_boost) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.2, 0.12)
            bg_color = tuple(int(c * 255) for c in rgb)
            # If a background image was successfully loaded, use it. Otherwise fill and optionally draw rainbow.
            if 'bg_image' in locals() and bg_image is not None:
                try:
                    screen.blit(bg_image, (0, 0))
                except Exception as e:
                    print(f"Background image blit error: {e}")
                    screen.fill(bg_color)
            else:
                screen.fill(bg_color)
                # --- blended rainbow overlay (two pass pre-rendered surfaces) ---
                if GD_RAINBOW_ENABLED and pre_surf1 is not None and pre_surf2 is not None:
                    try:
                        # animate by offsetting the pre-rendered wide surfaces
                        offset1 = int((elapsed_time * RAINBOW_SPEED1 * WIDTH) % WIDTH)
                        offset2 = int((elapsed_time * RAINBOW_SPEED2 * WIDTH) % WIDTH)
                        # blit wrapped: use WIDTH window into the double-width surface
                        screen.blit(pre_surf1, (-offset1, 0), (0, 0, WIDTH, HEIGHT))
                        screen.blit(pre_surf1, (WIDTH - offset1, 0), (WIDTH, 0, WIDTH, HEIGHT))
                        screen.blit(pre_surf2, (-offset2, 0), (0, 0, WIDTH, HEIGHT))
                        screen.blit(pre_surf2, (WIDTH - offset2, 0), (WIDTH, 0, WIDTH, HEIGHT))
                    except Exception as e:
                        print(f"Rainbow blit error: {e}")
            # draw ground: use ground_surf tiled if available, otherwise solid color
            if 'ground_surf' in locals() and ground_surf is not None:
                try:
                    gw, gh = ground_surf.get_size()
                    # tile across the ground rect
                    gx0 = 0
                    gy0 = HEIGHT - 80
                    for tx in range(0, WIDTH, gw):
                        screen.blit(ground_surf, (tx, gy0))
                except Exception as e:
                    print(f"Ground texture blit error: {e}")
                    pygame.draw.rect(screen, GROUND_COLOR, (0, HEIGHT - 80, WIDTH, 80))
            else:
                pygame.draw.rect(screen, GROUND_COLOR, (0, HEIGHT - 80, WIDTH, 80))
            # player (use texture if available)
            if 'block_surf' in locals() and block_surf is not None:
                try:
                    if player_tex is None:
                        try:
                            sw, sh = block_surf.get_size()
                            # use faster scale for player texture
                            player_tex = pygame.transform.scale(block_surf, (player.w, player.h))
                        except Exception as e:
                            print(f"Player texture scale error: {e}")
                            player_tex = None
                    if player_tex is not None:
                        screen.blit(player_tex, (player.x, int(player.y)))
                    else:
                        pygame.draw.rect(screen, PLAYER_COLOR, (player.x, int(player.y), player.w, player.h))
                except Exception as e:
                    print(f"Player texture blit error: {e}")
                    pygame.draw.rect(screen, PLAYER_COLOR, (player.x, int(player.y), player.w, player.h))
            else:
                pygame.draw.rect(screen, PLAYER_COLOR, (player.x, int(player.y), player.w, player.h))
            # obstacles (render as isosceles triangles pointing up)
            for o in obstacles:
                ox = int(o.x)
                oy = int(o.y)
                ow = o.w
                oh = o.h
                if o.tex is not None:
                    try:
                        screen.blit(o.tex, (ox, oy))
                        # draw a 1px black outline around the triangle
                        p1 = (ox, oy + oh)
                        p2 = (ox + ow, oy + oh)
                        p3 = (ox + ow // 2, oy)
                        pygame.draw.polygon(screen, (0, 0, 0), [p1, p2, p3], width=5)
                    except Exception as e:
                        print(f"Obstacle texture blit error: {e}")
                        p1 = (ox, oy + oh)
                        p2 = (ox + ow, oy + oh)
                        p3 = (ox + ow // 2, oy)
                        pygame.draw.polygon(screen, (120, 120, 120), [p1, p2, p3])
                        pygame.draw.polygon(screen, (0, 0, 0), [p1, p2, p3], width=5)
                elif 'spike_surf' in locals() and spike_surf is not None:
                    try:
                        # fallback to on-the-fly textured triangle if cache missing
                        tri_surf = pygame.Surface((ow, oh), pygame.SRCALPHA)
                        sw, sh = spike_surf.get_size()
                        scale = max(ow / sw, oh / sh)
                        tex_w = max(1, int(sw * scale))
                        tex_h = max(1, int(sh * scale))
                        tex = pygame.transform.smoothscale(spike_surf, (tex_w, tex_h))
                        tx = (ow - tex_w) // 2
                        ty = (oh - tex_h) // 2
                        tri_surf.blit(tex, (tx, ty))
                        mask = pygame.Surface((ow, oh), pygame.SRCALPHA)
                        p1 = (0, oh)
                        p2 = (ow, oh)
                        p3 = (ow // 2, 0)
                        pygame.draw.polygon(mask, (255, 255, 255, 255), [p1, p2, p3])
                        tri_surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                        screen.blit(tri_surf, (ox, oy))
                        # outline
                        p1 = (ox, oy + oh)
                        p2 = (ox + ow, oy + oh)
                        p3 = (ox + ow // 2, oy)
                        pygame.draw.polygon(screen, (0, 0, 0), [p1, p2, p3], width=5)
                    except Exception as e:
                        print(f"Textured spike render failed: {e}")
                        p1 = (ox, oy + oh)
                        p2 = (ox + ow, oy + oh)
                        p3 = (ox + ow // 2, oy)
                        pygame.draw.polygon(screen, (120, 120, 120), [p1, p2, p3])
                        pygame.draw.polygon(screen, (0, 0, 0), [p1, p2, p3], width=1)
                else:
                    # triangle points: left-bottom, right-bottom, center-top
                    p1 = (ox, oy + oh)
                    p2 = (ox + ow, oy + oh)
                    p3 = (ox + ow // 2, oy)
                    pygame.draw.polygon(screen, (120, 120, 120), [p1, p2, p3])
                    pygame.draw.polygon(screen, (0, 0, 0), [p1, p2, p3], width=5)

            # HUD: score, highscore
            draw_text_outline(screen, f"Score: {score}", font, 10, 10, fg=(0, 0, 0), outline=(255,255,255))
            draw_text_outline(screen, f"Highscore: {highscore}", font, WIDTH - 220, 10, fg=(0,0,0), outline=(255,255,255))
            # draw level and time to next level + progress bar
            time_in_level = elapsed_time - (level * level_duration)
            time_to_next = max(0.0, level_duration - time_in_level)
            draw_text_outline(screen, f"Level: {level}", font, 10, 44, fg=(0,0,0), outline=(255,255,255))
            draw_text_outline(screen, f"Next in: {int(time_to_next)}s", font, 10, 78, fg=(0,0,0), outline=(255,255,255))
            # progress bar background
            bar_w = 300
            bar_h = 16
            bar_x = WIDTH // 2 - bar_w // 2
            bar_y = 12
            pygame.draw.rect(screen, (80, 80, 80), (bar_x, bar_y, bar_w, bar_h))
            progress = min(1.0, max(0.0, time_in_level / level_duration))
            pygame.draw.rect(screen, (60, 200, 180), (bar_x, bar_y, int(bar_w * progress), bar_h))
            # Song title (under the progress bar) with marquee if too long
            try:
                song_title = music.get_current_title()
            except Exception as e:
                print(f"Song title error: {e}")
                song_title = None
            if song_title:
                now_label = f"Now: {song_title}"
                # measure available width (progress bar width)
                max_w = bar_w
                title_surf = font.render(now_label, True, (240, 240, 240))
                if title_surf.get_width() <= max_w:
                    tx = WIDTH // 2 - title_surf.get_width() // 2
                    ty = bar_y + bar_h + 6
                    draw_text_outline(screen, now_label, font, tx, ty, fg=(0,0,0), outline=(255,255,255))
                else:
                    # marquee: compute offset based on time
                    scroll_speed = 40  # px per second
                    offset = int((elapsed_time * scroll_speed) % (title_surf.get_width() + max_w))
                    base_x = WIDTH // 2 - max_w // 2
                    # draw two copies to create wrap
                    screen.set_clip((base_x, bar_y + bar_h + 6, max_w, title_surf.get_height()))
                    # render outline/title into temporary surface for marquee
                    tmp = pygame.Surface((title_surf.get_width(), title_surf.get_height()), pygame.SRCALPHA)
                    draw_text_outline(tmp, now_label, font, 0, 0, fg=(0,0,0), outline=(255,255,255))
                    screen.blit(tmp, (base_x - offset, bar_y + bar_h + 6))
                    screen.blit(tmp, (base_x - offset + title_surf.get_width() + 40, bar_y + bar_h + 6))
                    screen.set_clip(None)

            # Volume slider (vertical, right side)
            slider_w = 14
            slider_h = 180
            slider_x = WIDTH - slider_w - 20
            slider_y = HEIGHT // 2 - slider_h // 2
            # background
            pygame.draw.rect(screen, (60, 60, 60), (slider_x, slider_y, slider_w, slider_h), border_radius=6)
            # read current volume from music manager
            try:
                cur_vol = music.get_volume()
            except Exception as e:
                print(f"Volume get error: {e}")
                cur_vol = getattr(config, 'DEFAULT_VOLUME', 0.5)
            # filled portion (from bottom up)
            fill_h = int(slider_h * max(0.0, min(1.0, cur_vol)))
            fill_y = slider_y + (slider_h - fill_h)
            pygame.draw.rect(screen, (180, 180, 60), (slider_x, fill_y, slider_w, fill_h), border_radius=6)
            # knob
            knob_y = fill_y
            knob_radius = 8
            knob_x = slider_x + slider_w // 2
            pygame.draw.circle(screen, (230, 230, 230), (knob_x, knob_y), knob_radius)
            # volume text left of slider
            vol_txt = font.render(f"{int(cur_vol * 100)}%", True, (240, 240, 240))
            screen.blit(vol_txt, (slider_x - vol_txt.get_width() - 8, slider_y + slider_h // 2 - vol_txt.get_height() // 2))
            # Draw play/pause and next buttons below the slider
            btn_w = 36
            btn_h = 24
            btn_x = slider_x - 8 - btn_w
            btn_y = slider_y + slider_h + 12
            # play/pause
            is_paused = False
            try:
                is_paused = music.is_paused()
            except Exception as e:
                print(f"Pause check error: {e}")
                is_paused = False
            pygame.draw.rect(screen, (90, 90, 90), (btn_x, btn_y, btn_w, btn_h), border_radius=4)
            if is_paused:
                pygame.draw.polygon(screen, (220, 220, 220), [(btn_x+10, btn_y+6), (btn_x+10, btn_y+btn_h-6), (btn_x+btn_w-8, btn_y+btn_h//2)])
            else:
                # draw pause icon (two bars)
                pygame.draw.rect(screen, (220, 220, 220), (btn_x+8, btn_y+6, 6, btn_h-12))
                pygame.draw.rect(screen, (220, 220, 220), (btn_x+20, btn_y+6, 6, btn_h-12))
            # next button
            nbtn_x = btn_x - 8 - btn_w
            pygame.draw.rect(screen, (90, 90, 90), (nbtn_x, btn_y, btn_w, btn_h), border_radius=4)
            pygame.draw.polygon(screen, (220, 220, 220), [(nbtn_x+8, btn_y+6), (nbtn_x+8, btn_y+btn_h-6), (nbtn_x+btn_w-10, btn_y+btn_h//2)])

            # update and draw particles
            new_particles = []
            for p in particles:
                p['life'] -= dt
                if p['life'] <= 0:
                    continue
                # simple physics
                p['vy'] += 40 * dt  # gravity-like
                p['x'] += p['vx'] * dt
                p['y'] += p['vy'] * dt
                alpha = max(0.0, p['life'] / p['max'])
                col = (int(p['color'][0] * alpha), int(p['color'][1] * alpha), int(p['color'][2] * alpha))
                pygame.draw.circle(screen, col, (int(p['x']), int(p['y'])), int(p['size']))
                new_particles.append(p)
            particles[:] = new_particles

        if game_over:
            gtxt = font.render("Game Over - press R to restart", True, (240, 240, 240))
            screen.blit(gtxt, (WIDTH // 2 - 180, HEIGHT // 2 - 20))

        pygame.display.flip()

    pygame.quit()
    sys.exit()  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='geometry_dash_game', description='Run Geometry Dash demo')
    parser.add_argument('--rl', action='store_true', help='Run in RL mode (headless / controlled)')
    parser.add_argument('--model', type=str, default=None, help='Path to RL model (used when --rl is set)')
    parser.add_argument('--no-audio', action='store_true', help='Disable audio (music and sfx)')
    parser.add_argument('--seed', type=int, default=None, help='Optional RNG seed')
    args = parser.parse_args()
    if args.seed is not None:
        try:    
            random.seed(args.seed)
        except Exception as e:
            print(f"Seed set error: {e}")
    # Backwards compatible call; main accepts keywords so old callers won't crash
    main(use_rl=args.rl, model_path=args.model, no_audio=args.no_audio)

    # Add at the VERY END of the file (after the main() function and if __name__ block)

# Export classes for RL environment
if __name__ != '__main__':
    # These will be available when imported as a module
    __all__ = ['Player', 'Obstacle', 'WIDTH', 'HEIGHT', 'GROUP_SIZES', 'GROUP_INTERNAL_GAP', 'SPIKE_CHANCE']


    import random

def check_collision(player, ob):
    """Axis-aligned bounding box collision using the same scaled player hitbox as the main game."""
    hit_w = int(player.w * HITBOX_SCALE)
    hit_h = int(player.h * HITBOX_SCALE)
    hit_x = int(player.x + (player.w - hit_w) / 2)
    hit_y = int(player.y + (player.h - hit_h) / 2)
    return (hit_x < ob.x + ob.w and hit_x + hit_w > ob.x and hit_y < ob.y + ob.h and hit_y + hit_h > ob.y)

def spawn_obstacle_from_game():
    """Spawn a new obstacle using the game's real logic."""
    from geometry_dash_game import WIDTH, Obstacle  # local import to avoid circular dependency
    x = WIDTH + random.randint(150, 300)
    return Obstacle(x)

class GameWrapper:
    """Wraps your procedural Geometry Dash game to be AI-compatible."""
    def __init__(self, render=True):
        self.render_enabled = render
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.player = Player()
        self.obstacles = [spawn_obstacle_from_game()]  # use your game’s obstacle generator
        self.done = False
        self.score = 0

    def reset(self):
        self.player = Player()
        self.obstacles = [spawn_obstacle_from_game()]
        self.done = False
        self.score = 0
        return self.get_observation()

    def step(self, action):
        """Perform one game step. action: 1=jump, 0=do nothing"""
        dt = self.clock.tick(FPS) / 1000  # seconds
        if action == 1:
            self.player.jump_held = True
            self.player.jump()
        else:
            self.player.jump_held = False

        self.player.step(dt)
        for ob in self.obstacles:
            ob.step(speed=200, dt=dt)
            if check_collision(self.player, ob):
                self.done = True

        # remove passed obstacles and spawn new ones
        if self.obstacles and self.obstacles[0].x + self.obstacles[0].w < 0:
            self.obstacles.pop(0)
            self.obstacles.append(spawn_obstacle_from_game())

        # update score
        self.score += 1

        obs = self.get_observation()
        reward = 1  # simple reward per frame survived
        return obs, reward, self.done, {}

    def get_observation(self):
        """Return observation vector for AI: player y, velocity, distance & size of next obstacle"""
        next_ob = self.obstacles[0]
        return [self.player.y, self.player.v,
                next_ob.x - self.player.x,
                next_ob.w]  # reduce to 4 elements for SB3 Box space

    def render(self):
        if not self.render_enabled:
            return
        self.screen.fill((135, 206, 235))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.player.x, self.player.y, self.player.w, self.player.h))
        for ob in self.obstacles:
            pygame.draw.rect(self.screen, (0, 0, 0), (ob.x, ob.y, ob.w, ob.h))
        pygame.display.flip()