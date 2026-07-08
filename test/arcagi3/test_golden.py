"""Tier-2 golden-trace tests for the ARC-AGI-3 runtime subset.

Replays the recorded action traces (see ``scripts/generate_arcagi3_golden.py``)
and asserts the current implementation reproduces the stored frames and per-step
state exactly. This locks behavioral parity without needing the official
ARCEngine in CI. If these fail after an intentional engine change, review the
diff and regenerate with the script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jaxarc.arcagi3 import make_arcagi3

from .golden_common import build_traces, run_trace

GOLDEN_ROOT = Path(__file__).parent / "golden"
GAMES = ["simple_maze", "complex_maze"]


def _trace_names(game_id: str) -> list[str]:
    game_dir = GOLDEN_ROOT / game_id
    return sorted(
        p.name[: -len("_actions.npy")] for p in game_dir.glob("*_actions.npy")
    )


def _load_frames(game_dir: Path, trace: str) -> np.ndarray:
    """Load a trace's stored frames from the compressed ``.npz`` fixture."""
    return np.load(game_dir / f"{trace}_frames.npz")["frames"]


def _cases() -> list[tuple[str, str]]:
    return [(g, name) for g in GAMES for name in _trace_names(g)]


def test_fixtures_exist():
    for game_id in GAMES:
        names = _trace_names(game_id)
        assert names, (
            f"no golden traces for {game_id}; run scripts/generate_arcagi3_golden.py"
        )


@pytest.mark.parametrize(("game_id", "trace"), _cases())
def test_golden_frames_match(game_id, trace):
    _env, params = make_arcagi3(game_id)
    game_dir = GOLDEN_ROOT / game_id

    actions = np.load(game_dir / f"{trace}_actions.npy").tolist()
    expected_frames = _load_frames(game_dir, trace)

    frames, _ = run_trace(params, actions)
    assert frames.shape == expected_frames.shape
    np.testing.assert_array_equal(frames, expected_frames)


@pytest.mark.parametrize(("game_id", "trace"), _cases())
def test_golden_states_match(game_id, trace):
    _env, params = make_arcagi3(game_id)
    game_dir = GOLDEN_ROOT / game_id

    actions = np.load(game_dir / f"{trace}_actions.npy").tolist()
    expected = json.loads((game_dir / f"{trace}_states.json").read_text())

    _, records = run_trace(params, actions)
    assert records == expected


@pytest.mark.parametrize(("game_id", "trace"), _cases())
def test_golden_frames_in_palette(game_id, trace):
    # Every stored frame must be within the ARC-AGI-3 0..15 value contract.
    frames = _load_frames(GOLDEN_ROOT / game_id, trace)
    assert frames.min() >= 0
    assert frames.max() <= 15


def test_traces_are_current(maze_env):
    # The action traces the generator would produce today match what's on disk,
    # so a stale generator can't silently diverge from the committed fixtures.
    _env, params = maze_env
    traces = build_traces(params)
    for name, actions in traces.items():
        stored = np.load(GOLDEN_ROOT / "simple_maze" / f"{name}_actions.npy").tolist()
        assert actions == stored, f"{name} action trace drifted; regenerate goldens"
