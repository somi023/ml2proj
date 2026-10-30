"""Tests for src/common.py: seeding, run names, the code fingerprint and the runs log."""

import csv
import multiprocessing
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
import torch

from src.common import append_run, code_version, run_id, set_seed

CFG = {"method": "lora", "rank": 4, "lr": 3e-4}


def draw_numbers():
    return random.random(), np.random.rand(), torch.rand(1).item()


def test_same_seed_gives_same_numbers():
    set_seed(3)
    first = draw_numbers()
    set_seed(3)
    assert draw_numbers() == first


def test_different_seed_gives_different_numbers():
    set_seed(3)
    first = draw_numbers()
    set_seed(4)
    for a, b in zip(draw_numbers(), first, strict=True):
        assert a != b


def test_run_id_is_stable_and_ignores_key_order():
    reordered = {"lr": 3e-4, "rank": 4, "method": "lora"}
    assert run_id(CFG, 0, code="abc") == run_id(reordered, 0, code="abc")
    assert len(run_id(CFG, 0, code="abc")) == 12


def test_run_id_changes_with_seed_config_or_code():
    base = run_id(CFG, 0, code="abc")
    assert run_id(CFG, 1, code="abc") != base
    assert run_id({**CFG, "rank": 64}, 0, code="abc") != base
    assert run_id(CFG, 0, code="abd") != base


def git(repo, *args):
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
    subprocess.run(["git", "-C", str(repo), *identity, *args], check=True, capture_output=True)


def test_code_version_changes_only_with_code(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "code")
    v1 = code_version(tmp_path)

    # Committing a notes file does not change the fingerprint.
    (tmp_path / "notes.md").write_text("log\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "notes")
    assert code_version(tmp_path) == v1

    # An uncommitted code edit marks it dirty.
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    assert code_version(tmp_path) == v1 + "-dirty"

    # Committing the edit gives a new, clean fingerprint.
    git(tmp_path, "commit", "-qam", "code change")
    v2 = code_version(tmp_path)
    assert v2 != v1
    assert not v2.endswith("-dirty")


def test_code_version_outside_git(tmp_path):
    assert code_version(tmp_path) == "nogit"


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_append_run_writes_header_once(tmp_path):
    path = tmp_path / "runs.csv"
    append_run({"run_id": "a", "acc": 0.9}, path)
    append_run({"run_id": "b", "acc": 0.8}, path)
    assert read_rows(path) == [["run_id", "acc"], ["a", "0.9"], ["b", "0.8"]]


def test_append_run_blanks_missing_and_rejects_unknown_columns(tmp_path):
    path = tmp_path / "runs.csv"
    append_run({"run_id": "a", "acc": 0.9}, path)
    append_run({"run_id": "b"}, path)
    with pytest.raises(ValueError):
        append_run({"run_id": "c", "loss": 1.0}, path)
    assert read_rows(path) == [["run_id", "acc"], ["a", "0.9"], ["b", ""]]


def append_25_rows(path, worker, start_line):
    start_line.wait()  # all workers start writing at the same instant
    for i in range(25):
        append_run({"worker": worker, "i": i}, path)


def test_parallel_appends_keep_every_row_whole(tmp_path):
    path = tmp_path / "runs.csv"
    with multiprocessing.Manager() as manager, ProcessPoolExecutor(max_workers=4) as pool:
        start_line = manager.Barrier(4)
        list(pool.map(append_25_rows, [path] * 4, range(4), [start_line] * 4))
    rows = read_rows(path)
    assert rows[0] == ["worker", "i"]
    expected = sorted([str(w), str(i)] for w in range(4) for i in range(25))
    assert sorted(rows[1:]) == expected
