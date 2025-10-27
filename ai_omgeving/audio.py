"""Audio helper utilities.

Provides a small MusicManager wrapper around pygame.mixer.music to collect
tracks from a directory, control playback (next/prev/pause/resume) and expose
a simple volume API. Also provides a helper to load a jump sound, with a
fallback to synthesizing a tone via numpy if no file is found.
"""

import os
import random
import pygame
from typing import List
import config

# Custom event used to detect when a music track ended (if set via set_endevent)
MUSIC_END_EVENT = pygame.USEREVENT + 1


def load_jump_sound() -> pygame.mixer.Sound | None:
    """Try to load provided jump sound, local jump.wav, or synthesize a tone."""
    try:
        if os.path.exists(config.DEFAULT_JUMP_MP3):
            try:
                return pygame.mixer.Sound(config.DEFAULT_JUMP_MP3)
            except Exception:
                pass
        local = os.path.join(os.path.dirname(__file__), 'jump.wav')
        if os.path.exists(local):
            try:
                return pygame.mixer.Sound(local)
            except Exception:
                pass
        # synthesize if numpy available
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


class MusicManager:
    def __init__(self, music_dir: str, enabled: bool = True):
        # configuration
        self.music_dir = music_dir
        self.enabled = enabled
        # master volume (0.0 - 1.0)
        self._volume = getattr(config, 'DEFAULT_VOLUME', 0.5)
        # paused/playing state
        self._paused = False
        # playlist state
        self.playlist: List[str] = []
        self.index = 0
        # currently loaded track path (None if nothing loaded)
        self.current_track: str | None = None

        # collect playlist files from the directory
        self._collect()

        # ensure pygame sends an event when music ends (if any tracks were found)
        if self.playlist:
            try:
                pygame.mixer.music.set_endevent(MUSIC_END_EVENT)
                try:
                    pygame.mixer.music.set_volume(self._volume)
                except Exception:
                    pass
            except Exception:
                pass

    def _collect(self):
        self.playlist = []
        if not self.enabled or not os.path.isdir(self.music_dir):
            return
        found = []
        for root, dirs, files in os.walk(self.music_dir):
            for fn in files:
                if fn.lower().endswith(('.mp3', '.ogg', '.wav')):
                    rel = os.path.relpath(os.path.join(root, fn), self.music_dir)
                    found.append((rel, os.path.join(root, fn)))
        for rel, full in sorted(found, key=lambda x: x[0]):
            self.playlist.append(full)

    def shuffle_and_start(self):
        if not self.enabled or not self.playlist:
            return
        random.shuffle(self.playlist)
        self.index = 0
        self.play_next()

    def play_next(self):
        if not self.enabled or not self.playlist:
            return
        path = self.playlist[self.index]
        try:
            pygame.mixer.music.load(path)
            # record current track path
            self.current_track = path
            try:
                pygame.mixer.music.set_volume(self._volume)
            except Exception:
                pass
            pygame.mixer.music.play()
            self._paused = False
        except Exception:
            pass
        self.index = (self.index + 1) % len(self.playlist)

    def play_prev(self):
        """Play the previous track in the playlist (if available)."""
        if not self.enabled or not self.playlist:
            return
        # index currently points to the next item to play; to go back one track,
        # subtract 2 so that play_next() will load the previous entry.
        try:
            self.index = (self.index - 2) % len(self.playlist)
        except Exception:
            self.index = 0
        self.play_next()

    def handle_music_end_event(self, event):
        if event.type == MUSIC_END_EVENT:
            self.play_next()

    def set_volume(self, vol: float):
        """Set master music volume (0.0 - 1.0). Applies to pygame.mixer.music immediately."""
        try:
            self._volume = max(0.0, min(1.0, float(vol)))
        except Exception:
            return
        try:
            pygame.mixer.music.set_volume(self._volume)
        except Exception:
            pass

    def pause(self):
        try:
            pygame.mixer.music.pause()
            self._paused = True
        except Exception:
            pass

    def resume(self):
        try:
            pygame.mixer.music.unpause()
            self._paused = False
        except Exception:
            pass

    def toggle_pause(self):
        if self._paused:
            self.resume()
        else:
            self.pause()

    def is_paused(self) -> bool:
        return bool(self._paused)

    def get_volume(self) -> float:
        return float(self._volume)

    def get_current_title(self) -> str | None:
        """Return a friendly title for the currently loaded track (filename without extension), or None."""
        try:
            if not self.current_track:
                return None
            base = os.path.basename(self.current_track)
            name, _ext = os.path.splitext(base)
            # replace underscores with spaces for nicer display
            return name.replace('_', ' ')
        except Exception:
            return None

