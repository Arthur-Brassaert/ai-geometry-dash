import subprocess
import sys
import socket
import urllib.request
import time
from pathlib import Path
from datetime import datetime

# Log directory & vaste poort
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "gd_tensorboard"
PORT = 6022


def timestamp_now():
    return datetime.now().strftime("%H:%M / %d/%m/%y")


def find_event_files(base_dir: Path):
    if not base_dir.exists():
        return []
    return list(base_dir.rglob('events.out.tfevents.*'))


def wait_for_http(url: str, timeout: float = 10.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                return r.getcode() == 200
        except Exception:
            time.sleep(0.3)
    return False


def start_tensorboard():
    print(f"TensorBoard launcher started — timestamp: {timestamp_now()}")

    events = find_event_files(LOG_DIR)
    if not events:
        print(f"No TensorBoard events found in '{LOG_DIR}'.")
        return False

    print(f"TensorBoard logdir: {LOG_DIR.name} (found {len(events)} event file(s))")

    url = f"http://127.0.0.1:{PORT}"

    # Start TensorBoard
    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", str(LOG_DIR),
        "--port", str(PORT),
        "--bind_all"
    ]

    print(f"Starting TensorBoard with: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wachten tot TensorBoard draait
    if not wait_for_http(url, timeout=15.0):
        print("TensorBoard did not become reachable.")
        return False

    print(f"TensorBoard started on {url} (PID {proc.pid}).")
    return True


if __name__ == '__main__':
    if not start_tensorboard():
        print("TensorBoard failed to start.")
        sys.exit(1)
