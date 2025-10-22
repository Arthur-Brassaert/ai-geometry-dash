"""monitor_tensorboard.py

Small helper to find TensorBoard event folders in this repo and start TensorBoard
using the venv Python / TensorBoard API. Useful on Windows when `tensorboard` may
not be on PATH.

Usage:
  python monitor_tensorboard.py                          # auto-detect logdirs and start TB on port 6006
  python monitor_tensorboard.py --port 6007 --no-launch  # show found logdirs but don't start
  python monitor_tensorboard.py --logdir "path/to/logs"  # force a specific logdir
  python monitor_tensorboard.py --no-browser             # don't open browser automatically
  python monitor_tensorboard.py --help                   # show this help message

"""
import argparse
import os
import webbrowser
import sys
import importlib
import subprocess
import shutil
import time
# import the project's logging config to find the canonical LOG_ROOT
try:
    from logging_config import LOG_ROOT
except Exception:
    LOG_ROOT = None


REPO_ROOT = os.path.dirname(os.path.realpath(__file__))

# Candidate relative paths to search for event files
CANDIDATES = [
    # prefer the canonical LOG_ROOT from logging_config when available
    os.path.abspath(LOG_ROOT) if LOG_ROOT else None,
    os.path.join('geometry_dash_ai_V1', 'geometry dash', 'geometry_dash_project', 'tensorboard_log'),
    os.path.join('geometry_dash_ai_V1', 'geometry dash', 'geometry_dash_project', 'training_logs'),
    os.path.join('geometry_dash_ai_V1', 'geometry dash', 'geometry_dash_project', 'tensorboard_logs'),
]


def find_logdirs(explicit_logdir=None):
    dirs = []
    seen = set()

    def add_path(d):
        d = os.path.abspath(d)
        if d not in seen:
            seen.add(d)
            dirs.append(d)

    # If user supplied a specific logdir, inspect it for events (either
    # directly or in subfolders) and return only matching folders.
    if explicit_logdir:
        p = explicit_logdir if os.path.isabs(explicit_logdir) else os.path.join(REPO_ROOT, explicit_logdir)
        if not os.path.exists(p):
            return []
        try:
            # Check direct tfevents in p
            for f in os.listdir(p):
                if 'tfevents' in f:
                    add_path(p)
                    return dirs
            # Otherwise look for subfolders that contain tfevents
            for child in os.listdir(p):
                child_path = os.path.join(p, child)
                if os.path.isdir(child_path):
                    for f in os.listdir(child_path):
                        if 'tfevents' in f:
                            add_path(child_path)
                            break
            return dirs
        except PermissionError:
            return []

    # No explicit logdir: search the candidate locations
    for rel in CANDIDATES:
        if not rel:
            continue
        # allow absolute LOG_ROOT values or relative entries
        p = rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)
        if not os.path.isdir(p):
            continue
        try:
            # First, add any child folders that contain tfevents
            for child in os.listdir(p):
                child_path = os.path.join(p, child)
                if os.path.isdir(child_path):
                    for f in os.listdir(child_path):
                        if 'tfevents' in f:
                            add_path(child_path)
                            break
            # If the parent itself contains tfevents, include it as well
            for f in os.listdir(p):
                if 'tfevents' in f:
                    add_path(p)
                    break
        except PermissionError:
            continue

    return dirs
def start_tensorboard(logdir, port=6006, open_browser=True):
    """
    Try to use the TensorBoard Python API if available; otherwise fall back to
    launching the 'tensorboard' CLI via subprocess. Returns the TensorBoard
    object (if API used) or the subprocess.Popen object (if CLI used).
    """
    # First, try to import the tensorboard.program module dynamically.
    try:
        tb_module = importlib.import_module('tensorboard.program')
        tb = tb_module.TensorBoard()
        tb.configure(argv=[None, '--logdir', logdir, '--port', str(port)])
        url = tb.launch()
        print(f"TensorBoard serving {logdir} at: {url}")
        if open_browser:
            webbrowser.open(url)
        return tb
    except (ModuleNotFoundError, ImportError):
        # Fallback: try to find a tensorboard executable on PATH.
        tb_cmd = shutil.which('tensorboard')
        if not tb_cmd:
            raise RuntimeError(
                "TensorBoard package not found and 'tensorboard' command not on PATH. "
                "Install tensorboard in your environment (e.g. 'pip install tensorboard') "
                "or activate the virtual environment that contains it."
            )
        cmd = [tb_cmd, "--logdir", logdir, "--port", str(port)]
        # Start TensorBoard as a subprocess and don't block the caller.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        url = f"http://localhost:{port}/"
        print(f"Started tensorboard process for {logdir} at: {url} (PID {proc.pid})")
        if open_browser:
            # Give the server a moment to start before opening the browser.
            time.sleep(1.0)
            webbrowser.open(url)
        return proc
    if open_browser:
        webbrowser.open(url)
    return tb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default=None, help='Force a logdir to use')
    parser.add_argument('--port', type=int, default=6006)
    parser.add_argument('--no-launch', action='store_true', help="Don't start TensorBoard, just list found logdirs")
    parser.add_argument('--no-browser', action='store_true', help="Don't open the browser automatically")
    args = parser.parse_args()

    found = find_logdirs(args.logdir)
    if not found:
        print('No TensorBoard event folders found in the usual places.')
        if args.logdir:
            print(f"Explicit logdir '{args.logdir}' not found or contains no events.")
        else:
            print('Checked these candidate locations:')
            for c in CANDIDATES:
                if not c:
                    continue
                # show absolute path that was checked
                p = c if os.path.isabs(c) else os.path.join(REPO_ROOT, c)
                print(' -', os.path.abspath(p))
        sys.exit(1)

    print('Found TensorBoard folders:')
    for d in found:
        print(' -', d)

    if args.no_launch:
        print('Exiting without launching TensorBoard (--no-launch).')
        return

    # If multiple logdirs found, point TB at the parent folder that contains them
    # so TensorBoard shows them as separate runs. Otherwise use the single folder.
    if len(found) > 1:
        # find common parent
        common = os.path.commonpath(found)
        chosen = common
    else:
        chosen = found[0]

    print(f'Starting TensorBoard for: {chosen} on port {args.port} ...')
    start_tensorboard(chosen, port=args.port, open_browser=not args.no_browser)


if __name__ == '__main__':
    main()
