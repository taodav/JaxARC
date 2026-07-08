"""Tier-3 parity test: JAX ComplexMaze vs. the official ARCEngine.

Skipped by default. Runs only when BOTH:
- ``RUN_ARCAGI3_PARITY=1`` is set, and
- the optional ``arcengine`` package is importable.

Drives the real ``arcengine`` ``ComplexMaze`` and the pure-JAX port (all 5
levels, including the moving-maze level 5) through the same committed action
trace and asserts they agree on the final rendered frame after each action and on
the shared per-step state fields (``game_state``, ``levels_completed``) across the
**entire** trace, byte-for-byte.

The committed trace clears levels 1-4 and enters the moving-maze level 5. Level 5
is not solvable by pure ACTION1-4 movement (the exit sits behind a pushable block
against a wall, and the maze itself translates) — and crucially the **official
engine behaves identically**, so both end the trace at ``levels_completed=4`` /
``NOT_FINISHED``. The value of this test is the byte-exact agreement on the
moving-maze mechanic, blocks, invisible walls, and energy overlay throughout.

Run with::

    pip install arcengine
    RUN_ARCAGI3_PARITY=1 pytest test/arcagi3/test_parity_complex_maze.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from jaxarc.arcagi3 import make_arcagi3
from jaxarc.arcagi3.parity.official_adapter import (
    arcengine_available,
    run_official_trace,
)
from jaxarc.arcagi3.parity.trace_runner import run_trace

_ENABLED = os.environ.get("RUN_ARCAGI3_PARITY") == "1"
_HAVE_ENGINE = arcengine_available()

pytestmark = [
    pytest.mark.skipif(
        not _ENABLED, reason="set RUN_ARCAGI3_PARITY=1 to run official parity tests"
    ),
    pytest.mark.skipif(
        not _HAVE_ENGINE, reason="optional `arcengine` package not installed"
    ),
]

_REFERENCE = Path(__file__).parent / "reference" / "complex_maze_reference.py"
_GOLDEN = Path(__file__).parent / "golden" / "complex_maze"


def _load_reference_game():
    """Import the vendored reference module and instantiate ``ComplexMaze``."""
    spec = importlib.util.spec_from_file_location(
        "arcagi3_ref_complex_maze", _REFERENCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ComplexMaze()


def _trace_items() -> list[tuple[str, list[int]]]:
    # Use the committed winning trace (all 5 levels) to avoid re-running the solver.
    actions = np.load(_GOLDEN / "trace_000_actions.npy").tolist()
    return [("trace_000", actions)]


@pytest.mark.parametrize(("trace_name", "actions"), _trace_items())
def test_parity_final_frames(trace_name, actions):
    _env, params = make_arcagi3("complex_maze")
    jax_frames, _ = run_trace(params, actions)

    game = _load_reference_game()
    official_frames, _ = run_official_trace(game, actions)

    assert jax_frames.shape == official_frames.shape, (
        f"{trace_name}: frame-stack length differs "
        f"({jax_frames.shape} vs {official_frames.shape})"
    )
    for i in range(jax_frames.shape[0]):
        np.testing.assert_array_equal(
            jax_frames[i],
            official_frames[i],
            err_msg=f"{trace_name}: frame {i} differs from official engine",
        )


@pytest.mark.parametrize(("trace_name", "actions"), _trace_items())
def test_parity_state_fields(trace_name, actions):
    _env, params = make_arcagi3("complex_maze")
    _, jax_records = run_trace(params, actions)

    game = _load_reference_game()
    _, official_records = run_official_trace(game, actions)

    assert len(jax_records) == len(official_records)
    for i, (jr, orr) in enumerate(zip(jax_records, official_records)):
        assert jr["game_state"] == orr["game_state"], (
            f"{trace_name}: step {i} game_state {jr['game_state']} "
            f"!= official {orr['game_state']}"
        )
        assert jr["levels_completed"] == orr["levels_completed"], (
            f"{trace_name}: step {i} levels_completed {jr['levels_completed']} "
            f"!= official {orr['levels_completed']}"
        )
