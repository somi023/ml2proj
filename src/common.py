"""Helpers every run uses: fix random seeds, name a run, and log it to results/runs.csv."""

import csv
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
RUNS_CSV = ROOT / "results" / "runs.csv"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every GPU) so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device
    # Use the same cuDNN kernels every time instead of benchmarking for the fastest one.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _git(repo: Path, *args: str) -> str | None:
    """Run a git command in `repo` and return its output, or None if git or the repo is missing."""
    try:
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_commit(repo: Path = ROOT) -> str:
    """Full hash of the current commit, for the runs log."""
    return _git(repo, "rev-parse", "HEAD") or "nogit"


def code_version(repo: Path = ROOT) -> str:
    """Short hash of the committed src/ folder, plus '-dirty' if src/ or configs/ has
    uncommitted edits.

    Unlike the commit hash, it stays the same when only notes or docs are committed,
    so a crashed run can resume under the same run_id.
    """
    tree = _git(repo, "rev-parse", "HEAD:src")
    if tree is None:
        return "nogit"
    dirty = _git(repo, "status", "--porcelain", "--", "src", "configs")
    return tree[:12] + ("-dirty" if dirty else "")


def run_id(cfg: dict, seed: int, code: str | None = None) -> str:
    """12-character name for one run: the same config, seed and code always give the same id."""
    if code is None:
        code = code_version()
    payload = json.dumps({"cfg": cfg, "seed": seed, "code": code}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def append_run(row: dict, path: Path = RUNS_CSV) -> None:
    """Append one row to the runs log, writing the header first if the file is new.

    The lock stops two jobs that finish at the same time from mixing their lines.
    Rows must use the columns of the first row: a missing column is left blank,
    an unknown column raises ValueError.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(f"{path}.lock", timeout=60):
        is_new = not path.exists() or path.stat().st_size == 0
        if is_new:
            columns = list(row)
        else:
            with path.open(newline="") as f:
                columns = next(csv.reader(f))
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
            if is_new:
                writer.writeheader()
            writer.writerow(row)
