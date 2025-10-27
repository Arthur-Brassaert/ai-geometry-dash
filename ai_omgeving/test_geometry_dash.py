"""Convenience test runner — delegates to the visual runner.

This file forwards CLI args to `run_model_in_game.py` so that running
`python ai_omgeving/test_geometry_dash.py` behaves the same as running
the dedicated runner. Use --model, --max-steps, --deterministic as needed.
"""
from __future__ import annotations
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
if str(here) not in sys.path:
    sys.path.insert(0, str(here))

try:
    import run_model_in_game as runner
except Exception as e:
    print(f"Failed to import run_model_in_game: {e}")
    raise

if __name__ == '__main__':
    # Ensure audio and repository assets are enabled by default when using this convenience script.
    # If the user explicitly passed --no-audio, preserve that choice; otherwise remove any accidental
    # '--no-audio' removal to force enablement. Here we proactively remove '--no-audio' so music will play.
    cleaned = []
    skip_next = False
    for a in sys.argv[1:]:
        if a == '--no-audio':
            # drop it to enable audio
            continue
        cleaned.append(a)
    sys.argv = [sys.argv[0]] + cleaned
    runner.main()
