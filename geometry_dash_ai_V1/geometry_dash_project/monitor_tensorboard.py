"""monitor_tensorboard.py

Robust helper to find TensorBoard event folders and start TensorBoard
using the project's virtual environment Python when available.

This script does the following:
 - auto-detect likely `trained_models/logs` locations inside the project
 - if a venv is present (./.venv on Windows), use its Python to run
   `python -m tensorboard.main --logdir ...` so tensorboard runs with the
   environment that has the correct packages installed
 - runs TensorBoard in the foreground and (optionally) opens a browser

Usage:
  python monitor_tensorboard.py                # auto-detect and start TB on port 6006
  python monitor_tensorboard.py --logdir PATH  # force a specific logdir
  python monitor_tensorboard.py --no-browser   # don't open the browser
  python monitor_tensorboard.py --port 6007

"""

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from typing import List, Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def find_venv_python() -> Optional[str]:
    """Return path to the project's venv Python executable if present.

    On Windows this is `.venv\Scripts\python.exe`. On *nix it's `.venv/bin/python`.
    """
    # Prefer an activated venv if available
    venv = os.environ.get('VIRTUAL_ENV')
    if venv:
        candidate = os.path.join(venv, 'Scripts' if os.name == 'nt' else 'bin', 'python.exe' if os.name == 'nt' else 'python')
        if os.path.isfile(candidate):
            return candidate

    # Fall back to a .venv folder inside the repo
    dotvenv = os.path.join(REPO_ROOT, '.venv')
    if os.path.isdir(dotvenv):
        candidate = os.path.join(dotvenv, 'Scripts' if os.name == 'nt' else 'bin', 'python.exe' if os.name == 'nt' else 'python')
        if os.path.isfile(candidate):
            return candidate

    return None


def find_logdirs(explicit_logdir: Optional[str] = None) -> List[str]:
    """Search likely locations for TensorBoard event files and return matching folders.

    If explicit_logdir is supplied it will be inspected (absolute or repo-relative).
    """
    candidates = []
    # Common locations used in this project
    candidates.append(os.path.join(REPO_ROOT, 'geometry_dash_project', 'trained_models', 'logs'))
    candidates.append(os.path.join(REPO_ROOT, 'trained_models', 'logs'))
    candidates.append(os.path.join(REPO_ROOT, 'geometry_dash_ai_v1', 'geometry_dash_project', 'trained_models', 'logs'))
    candidates.append(os.path.join(REPO_ROOT, 'geometry_dash_ai_V1', 'geometry_dash_project', 'trained_models', 'logs'))

    # Include an explicit logdir if provided
    if explicit_logdir:
        p = explicit_logdir if os.path.isabs(explicit_logdir) else os.path.join(REPO_ROOT, explicit_logdir)
        if os.path.exists(p):
            candidates.insert(0, p)

    found = []
    seen = set()

    def _add(path: str):
        path = os.path.abspath(path)
        if path not in seen and os.path.isdir(path):
            seen.add(path)
            found.append(path)

    for c in candidates:
        if not c:
            continue
        if not os.path.isdir(c):
            continue
        # If the folder itself has event files, add it
        try:
            for fname in os.listdir(c):
                if 'tfevents' in fname:
                    _add(c)
                    break
            # Also check subfolders for event files (typical TB layout)
            for child in os.listdir(c):
                child_path = os.path.join(c, child)
                if os.path.isdir(child_path):
                    for fname in os.listdir(child_path):
                        if 'tfevents' in fname:
                            _add(child_path)
                            break
        except PermissionError:
            continue

    return found


def start_tensorboard_subprocess(python_exe: str, logdir: str, port: int, open_browser: bool) -> subprocess.Popen:
    """Start TensorBoard using the specified Python executable in the foreground.

    Returns the subprocess.Popen instance.
    """
    cmd = [python_exe, '-m', 'tensorboard.main', '--logdir', logdir, '--port', str(port)]
    print('Launching:', ' '.join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    url = f'http://localhost:{port}/'
    # Wait briefly for server to start and open browser
    if open_browser:
        # give tensorboard a second to bind the port
        time.sleep(1.0)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    return proc


def tail_process_output(proc: subprocess.Popen):
    """Stream subprocess output to stdout until it exits or user interrupts."""
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='')
    except KeyboardInterrupt:
        print('\nKeyboard interrupt received — terminating TensorBoard...')
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def main():
    parser = argparse.ArgumentParser(description='Find TensorBoard logs and run TensorBoard (uses project venv if available)')
    parser.add_argument('--logdir', type=str, default=None, help='Force a specific logdir (absolute or repo-relative path)')
    parser.add_argument('--port', type=int, default=6006, help='Port for TensorBoard')
    parser.add_argument('--no-browser', action='store_true', help="Don't open a browser automatically")
    parser.add_argument('--no-launch', action='store_true', help="Don't launch TensorBoard; just list found logdirs")
    args = parser.parse_args()

    found = find_logdirs(args.logdir)
    if not found:
        print('No TensorBoard event folders found in the usual places.')
        if args.logdir:
            print('Explicit logdir provided but no events found at:', args.logdir)
        else:
            print('Checked (repo-relative) candidates:')
            print(' -', os.path.join(REPO_ROOT, 'geometry_dash_project', 'trained_models', 'logs'))
            print(' -', os.path.join(REPO_ROOT, 'trained_models', 'logs'))
        sys.exit(1)

    print('Found TensorBoard folders:')
    for d in found:
        print(' -', d)

    if args.no_launch:
        print('Exiting without launching TensorBoard (--no-launch).')
        return

    # If multiple runs found, point TB at their common parent so TB shows separate runs
    chosen = os.path.commonpath(found) if len(found) > 1 else found[0]

    venv_python = find_venv_python()
    if venv_python:
        print('Using venv python:', venv_python)
    else:
        print('No project venv detected; using current Python interpreter:', sys.executable)
        venv_python = sys.executable

    print(f'Starting TensorBoard for: {chosen} on port {args.port} ...')
    proc = start_tensorboard_subprocess(venv_python, chosen, args.port, open_browser=not args.no_browser)

    # Stream output until process exits or user interrupts
    tail_process_output(proc)


if __name__ == '__main__':
    main()
