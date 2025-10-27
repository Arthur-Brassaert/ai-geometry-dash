import os

"""Centralized configuration for the Geometry Dash demo.

This module contains tunable constants and environment-driven defaults used
throughout the small demo project. Keep values here to make them easy to find
and edit.
"""

# --- Display Settings ---
WIDTH = 1000
HEIGHT = 600
FPS = 60

# --- Observation Settings (for AI / Gym interface) ---
OBS_HORIZON = 500        # How far ahead (in pixels) the AI can "see"
OBS_RESOLUTION = 10      # Pixel step size when sampling the environment
OBS_Y_OFFSET = 0         # Vertical offset for the observation line (0 = player height)

# --- Player Physics & Movement ---
PLAYER_W = 40
PLAYER_H = 40
GRAVITY = 0.8  # Verlaagd voor langzamere daling
JUMP_STRENGTH = -18  # Verhoogd voor hogere sprongen
MAX_JUMP_HOLD = 0.15  # Verhoogd voor variabel springen
HOLD_GRAVITY = 0.5  # Verlaagd voor meer lift
AUTO_JUMP_ON_LAND = False  # Uit voor preciezere controle
HITBOX_SCALE = 0.9  # Verhoogd voor vergevingsgezinder collisions

# --- Obstacle Spawning & Difficulty ---
START_SPAWN_MIN = 30  # Min frames tussen spawns (~0.5s)
START_SPAWN_MAX = 50  # Max frames tussen spawns (~0.8s)
SPAWN_MIN_FLOOR = 40  # Min spawn-tijd (safety)
SPAWN_MAX_FLOOR = 60  # Max spawn-tijd
SPAWN_MIN_DECREASE = 1  # Langzame afname per level
SPAWN_MAX_DECREASE = 2  # Langzame afname per level
GROUP_SIZES = [1]  # Start met 1 obstakel per groep
GROUP_INTERNAL_GAP = 0  # Geen gap binnen groepen
SPIKE_CHANCE = 0.0  # Geen spikes eerst
BASE_SPEED = 300.0  # Basissnelheid (pixels per second)
SPEED_INCREASE_PER_LEVEL = 0.3  # Langzame moeilijkheidsverhoging
LEVEL_DURATION = 30.0  # Langere levels voor leertijd
TRAINING_STEPS_PER_LEVEL = 5000  # Stappen voor moeilijkheidsverhoging

# --- Groups ---
GAP_MIN = 6
GAP_MAX = 18
MIN_GROUP_GAP_MULT = 2
GROUP_GAP_MIN = PLAYER_W * MIN_GROUP_GAP_MULT
GROUP_GAP_MAX = PLAYER_W * (MIN_GROUP_GAP_MULT + 3)

# --- Files / Paths ---
_BASE_DIR = os.path.dirname(__file__)
HIGHSCORE_FILE = os.path.join(_BASE_DIR, 'highscore.txt')
DEFAULT_MUSIC_DIR = os.path.join(_BASE_DIR, 'sounds', 'level songs')
DEFAULT_JUMP_MP3 = os.path.join(_BASE_DIR, 'sounds', 'sound effects', 'jump.mp3')

# --- Visuals ---
PARTICLE_COUNT_BASE = 40
PARTICLE_LIFE = 2.0
LEVEL_HUE_SHIFT = 0.35

RAINBOW_PRESETS = {
    'soft': (6, 0.65, 0.28, 0.06, 0.12, 20, 36),
    'pastel': (8, 0.45, 0.55, 0.04, 0.08, 16, 28),
    'vibrant': (4, 0.95, 0.35, 0.10, 0.18, 36, 64),
}
DEFAULT_RAINBOW_STYLE = 'vibrant'

# --- Environment Toggles ---
GD_RAINBOW = os.getenv('GD_RAINBOW', '1') != '0'
GD_MUSIC = os.getenv('GD_MUSIC', '1') != '0'
GD_RAINBOW_STYLE = os.getenv('GD_RAINBOW_STYLE', DEFAULT_RAINBOW_STYLE)

# --- Audio ---
DEFAULT_VOLUME = 0.5

# --- Image Subfolders ---
IMAGE_SUBFOLDERS = {
    'background': 'background',
    'floor': 'floor',
    'obstacles': 'obstacles',
}

# --- Utility Function ---
def get_rainbow_params(style=None):
    s = style if style is not None else GD_RAINBOW_STYLE
    return RAINBOW_PRESETS.get(s, RAINBOW_PRESETS[DEFAULT_RAINBOW_STYLE])
