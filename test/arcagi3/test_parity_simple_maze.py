"""Tier-3 parity test: JAX SimpleMaze vs. the official ARCEngine.

Skipped by default. Runs only when BOTH:
- ``RUN_ARCAGI3_PARITY=1`` is set, and
- the optional ``arcengine`` package is importable.

For each canonical trace, this drives the real ``arcengine`` ``SimpleMaze`` and
the pure-JAX port through the same action ids and asserts they agree on:
- the **final** rendered frame after each action (64x64, values 0..15), and
- the shared per-step state fields (``game_state``, ``levels_completed``).

Frame note (documented v1 divergence): the official ``FrameData.frame`` is a
*stack* of frames per action; v1 compares the final frame only (animation
sub-frames are a later milestone).

Install the optional dependency and run with, e.g.::

    pip install arcengine
    RUN_ARCAGI3_PARITY=1 pytest test/arcagi3/test_parity_simple_maze.py
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

from .golden_common import build_traces

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

_REFERENCE = Path(__file__).parent / "reference" / "simple_maze_reference.py"


def _load_reference_game():
    """Import the vendored reference module and instantiate ``SimpleMaze``.

    Mirrors how the official ``LocalEnvironmentWrapper`` exec-loads a game file,
    but via a normal import of the committed reference source.
    """
    spec = importlib.util.spec_from_file_location("arcagi3_ref_simple_maze", _REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SimpleMaze()


def _trace_items() -> list[tuple[str, list[int]]]:
    _env, params = make_arcagi3("simple_maze")
    return list(build_traces(params).items())


@pytest.mark.parametrize(("trace_name", "actions"), _trace_items())
def test_parity_final_frames(trace_name, actions):
    _env, params = make_arcagi3("simple_maze")
    jax_frames, _ = run_trace(params, actions)

    game = _load_reference_game()
    official_frames, _ = run_official_trace(game, actions)

    assert jax_frames.shape == official_frames.shape, (
        f"{trace_name}: frame-stack length differs "
        f"({jax_frames.shape} vs {official_frames.shape})"
    )
    # Compare the final frame after each action (index 0 == reset).
    for i in range(jax_frames.shape[0]):
        np.testing.assert_array_equal(
            jax_frames[i],
            official_frames[i],
            err_msg=f"{trace_name}: frame {i} differs from official engine",
        )


@pytest.mark.parametrize(("trace_name", "actions"), _trace_items())
def test_parity_state_fields(trace_name, actions):
    _env, params = make_arcagi3("simple_maze")
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
