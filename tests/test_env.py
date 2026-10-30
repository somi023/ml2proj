"""Installed packages must match requirements.txt: pin before the first sweep, never upgrade."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"


def _pins():
    pins = []
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            name, _, pinned = line.partition("==")
            pins.append((name, pinned))
    return pins


def test_every_requirement_is_pinned_exactly():
    assert _pins(), "requirements.txt is empty"
    unpinned = [name for name, pinned in _pins() if not pinned]
    assert not unpinned, f"not pinned with ==: {unpinned}"


@pytest.mark.parametrize(("name", "pinned"), _pins())
def test_installed_version_matches_pin(name, pinned):
    try:
        installed = version(name)
    except PackageNotFoundError:
        pytest.fail(f"{name} is pinned but not installed; run make setup")
    # Ignore local build tags such as +cu128 from the PyTorch wheel index.
    assert installed.split("+")[0] == pinned, f"{name}: installed {installed}, pinned {pinned}"
