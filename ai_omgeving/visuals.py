"""Visual helper functions for procedural effects.

Currently contains functions to create and render a repeated rainbow
overlay used by the demo for a subtle background effect.
"""

import pygame
import colorsys
import config


def create_rainbow_surfaces(width: int, height: int, style: str):
    col_w, sat, val, speed1, speed2, a1, a2 = config.get_rainbow_params(style)
    try:
        surf1 = pygame.Surface((width * 2, height), pygame.SRCALPHA)
        for x in range(0, width * 2, col_w):
            h = (x / float(width)) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, sat, val)
            col = (int(r * 255), int(g * 255), int(b * 255), a1)
            pygame.draw.rect(surf1, col, (x, 0, col_w, height))
        col_w2 = max(2, col_w // 2)
        surf2 = pygame.Surface((width * 2, height), pygame.SRCALPHA)
        for x in range(0, width * 2, col_w2):
            h2 = ((x / float(width)) * 0.5) % 1.0
            r2, g2, b2 = colorsys.hsv_to_rgb(h2, sat, min(1.0, val * 1.1))
            col2 = (int(r2 * 255), int(g2 * 255), int(b2 * 255), a2)
            pygame.draw.rect(surf2, col2, (x, 0, col_w2, height))
        # Return two wide surfaces so the caller can blit them with wrapping
        return surf1, surf2
    except Exception:
        return None, None


def draw_rainbow(screen, surf1, surf2, elapsed_time, style: str):
    if surf1 is None or surf2 is None:
        return
    _, _, _, speed1, speed2, _, _ = config.get_rainbow_params(style)
    width = screen.get_width()
    offset1 = int((elapsed_time * speed1 * width) % width)
    offset2 = int((elapsed_time * speed2 * width) % width)
    try:
        screen.blit(surf1, (-offset1, 0), (0, 0, width, screen.get_height()))
        screen.blit(surf1, (width - offset1, 0), (width, 0, width, screen.get_height()))
        screen.blit(surf2, (-offset2, 0), (0, 0, width, screen.get_height()))
        screen.blit(surf2, (width - offset2, 0), (width, 0, width, screen.get_height()))
    except Exception:
        pass
