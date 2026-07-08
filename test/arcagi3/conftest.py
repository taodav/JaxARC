"""Shared fixtures/helpers for ARC-AGI-3 runtime tests."""

from __future__ import annotations

import pytest

from jaxarc.arcagi3 import make_arcagi3


@pytest.fixture
def maze_env():
    """A fresh SimpleMaze (env, params) for tests asserting its layout/scoring."""
    return make_arcagi3("simple_maze")
