"""Shared fixtures/helpers for ARC-AGI-3 runtime tests."""

from __future__ import annotations

import pytest

from jaxarc.arcagi3 import make_arcagi3


@pytest.fixture
def maze_env():
    """A fresh SimpleMaze (env, params) for tests asserting its layout/scoring."""
    return make_arcagi3("simple_maze")


@pytest.fixture(params=["simple_maze", "complex_maze"])
def game_env(request):
    """A fresh (env, params) for each built-in movement game (parametrized).

    Both games are fully solvable by their committed traces (see load_solution).
    """
    return make_arcagi3(request.param)
