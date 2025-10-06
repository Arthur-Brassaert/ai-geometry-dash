import os

"""Centralized configuration for the Geometry Dash demo.

This module contains tunable constants and environment-driven defaults used
throughout the small demo project. Keep values here to make them easy to find
and edit.
"""

# Display
WIDTH = 1000
HEIGHT = 600
FPS = 60

# Difficulty / spawning
START_SPAWN_MIN = 18
START_SPAWN_MAX = 80
BASE_SPEED = 6.0
LEVEL_DURATION = 15.0
SPEED_INCREASE_PER_LEVEL = 1.5
SPAWN_MIN_DECREASE = 8
SPAWN_MAX_DECREASE = 12
SPAWN_MIN_FLOOR = 20
SPAWN_MAX_FLOOR = 30

# Player
PLAYER_W = 40
PLAYER_H = 40

# Physics
GRAVITY = 1.0
JUMP_STRENGTH = -15
MAX_JUMP_HOLD = 0.12
HOLD_GRAVITY = 0.75
AUTO_JUMP_ON_LAND = True

# Groups
GAP_MIN = 6
GAP_MAX = 18
MIN_GROUP_GAP_MULT = 2
GROUP_GAP_MIN = PLAYER_W * MIN_GROUP_GAP_MULT
GROUP_GAP_MAX = PLAYER_W * (MIN_GROUP_GAP_MULT + 3)
GROUP_SIZES = [1, 2, 3]
GROUP_INTERNAL_GAP = 0

# Hitbox
HITBOX_SCALE = 0.7

# Files / paths
_BASE_DIR = os.path.dirname(__file__)
HIGHSCORE_FILE = os.path.join(_BASE_DIR, 'highscore.txt')
# Default music/sfx paths are now relative to the project directory so the
# project is portable between machines. Create a `sounds` folder next to
# this module and place your music under `sounds/level songs` and SFX under
# `sounds/sound effects`.
DEFAULT_MUSIC_DIR = os.path.join(_BASE_DIR, 'sounds', 'level songs')
DEFAULT_JUMP_MP3 = os.path.join(_BASE_DIR, 'sounds', 'sound effects', 'jump.mp3')

# Visuals
PARTICLE_COUNT_BASE = 40
PARTICLE_LIFE = 2.0
LEVEL_HUE_SHIFT = 0.35

RAINBOW_PRESETS = {
    'soft': (6, 0.65, 0.28, 0.06, 0.12, 20, 36),
    'pastel': (8, 0.45, 0.55, 0.04, 0.08, 16, 28),
    'vibrant': (4, 0.95, 0.35, 0.10, 0.18, 36, 64),
}
DEFAULT_RAINBOW_STYLE = 'vibrant'

# Env toggles
GD_RAINBOW = os.getenv('GD_RAINBOW', '1') != '0'
GD_MUSIC = os.getenv('GD_MUSIC', '1') != '0'
GD_RAINBOW_STYLE = os.getenv('GD_RAINBOW_STYLE', DEFAULT_RAINBOW_STYLE)

# Default master volume (0.0 - 1.0). Can be used to initialize UI or audio managers.
DEFAULT_VOLUME = 0.5

# Image subfolder names (under the project's `images/` directory)
# You can change these if you prefer different folder names.
IMAGE_SUBFOLDERS = {
    'background': 'background',
    'floor': 'floor',
    'obstacles': 'obstacles',
}


# Small convenience function
def get_rainbow_params(style=None):
    s = style if style is not None else GD_RAINBOW_STYLE
    return RAINBOW_PRESETS.get(s, RAINBOW_PRESETS[DEFAULT_RAINBOW_STYLE])
