import argparse
import subprocess
import sys
import socket
import urllib.request
import webbrowser
import time
from pathlib import Path
from datetime import datetime

# Log directory relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "gd_tensorboard"
DEFAULT_PORT = 6006

def timestamp_now():
    """Return timestamp in format 'HH:MM / DD/MM/YY'."""
    return datetime.now().strftime("%H:%M / %d/%m/%y")

def find_event_files(base_dir: Path):
    if not base_dir.exists():
        return []
    return list(base_dir.rglob('events.out.tfevents.*'))

def is_port_open(port: int):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=0.5):
            return True
    except Exception:
        return False

def find_free_port(start: int = DEFAULT_PORT, max_tries: int = 20):
    for p in range(start, start + max_tries):
        if not is_port_open(p):
            return p
    return None

def wait_for_http(url: str, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                return r.getcode() == 200
        except Exception:
            time.sleep(0.3)
    return False

def get_pid_on_port(port: int):
    try:
        out = subprocess.check_output(["netstat", "-ano"], encoding='utf-8', errors='ignore')
        for line in out.splitlines():
            if f':{port} ' in line and 'LISTEN' in line.upper():
                parts = line.split()
                if parts:
                    pid_str = parts[-1]
                    try:
                        return int(pid_str)
                    except Exception:
                        continue
        return None
    except Exception:
        return None

def kill_pid(pid: int):
    if pid is None:
        return False
    if sys.platform.startswith('win'):
        try:
            subprocess.check_call(["taskkill", "/PID", str(pid), "/F"])
            return True
        except Exception:
            return False
    else:
        import os, signal
        try:
            os.kill(pid, signal.SIGTERM)
            return True
        except Exception:
            return False

def start_tensorboard(logdir: Path, preferred_port: int = DEFAULT_PORT):
    print(f"TensorBoard launcher started — timestamp: {timestamp_now()}")

    # Check for event files
    events = find_event_files(logdir)
    if not events:
        print(f"No TensorBoard events found in '{logdir}'. Ensure training logs are written to {logdir}.")
        return False

    # Use relative display path
    try:
        display = logdir.relative_to(PROJECT_ROOT)
    except Exception:
        display = logdir
    print(f"TensorBoard logdir: {display} (found {len(events)} event file(s))")

    # Find a free port
    port = find_free_port(preferred_port)
    if port is None:
        print("Could not find a free port to start TensorBoard.")
        return False

    url = f"http://127.0.0.1:{port}"

    # Start TensorBoard
    cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir), "--port", str(port), "--bind_all"]
    print(f"Starting TensorBoard with: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait until server is reachable or process fails
    if not wait_for_http(url, timeout=15.0):
        try:
            stderr = proc.stderr.read().decode(errors='ignore')
        except Exception:
            stderr = '<no stderr available>'
        print("TensorBoard could not start; process error:\n", stderr)
        return False

    print(f"TensorBoard started on {url} (PID {proc.pid}). Opening browser...")
    webbrowser.open(url)
    return True

if __name__ == '__main__':
    # Automatically start TensorBoard with the relative gd_tensorboard directory
    if not start_tensorboard(LOG_DIR, DEFAULT_PORT):
        print("TensorBoard failed to start — check logs and installation.")
        sys.exit(1)