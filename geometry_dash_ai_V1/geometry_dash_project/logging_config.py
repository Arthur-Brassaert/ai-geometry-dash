import os
from datetime import datetime

# Central log root used by all trainers in this project.
# Trainers should use get_tb_log_root() for SB3's `tensorboard_log`
# and get_run_dir() for custom SummaryWriter runs.
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
# Place TensorBoard logs next to trained models for easier grouping
LOG_ROOT = os.path.join(REPO_DIR, 'trained_models', 'logs')


def ensure_log_root():
    """Create the canonical log root directory if it doesn't exist."""
    os.makedirs(LOG_ROOT, exist_ok=True)


def get_tb_log_root():
    """Return the absolute path to the TensorBoard root folder.

    This is intended to be passed as `tensorboard_log` to SB3 (it will
    create per-run subfolders there when tb_log_name is provided).
    """
    ensure_log_root()
    return LOG_ROOT


def get_run_dir(model_name: str | None = None, timestamp: bool = True) -> str:
    """Return a unique subfolder inside LOG_ROOT for custom writers.

    Example: get_run_dir('my_model') -> '<...>/training_logs/my_model_20250101_120000'
    """
    ensure_log_root()
    base = model_name or 'run'
    if timestamp:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(LOG_ROOT, f"{base}_{ts}")
    return os.path.join(LOG_ROOT, base)
