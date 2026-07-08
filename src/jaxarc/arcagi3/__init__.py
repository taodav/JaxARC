"""Experimental pure-JAX ARC-AGI-3 runtime subset.

A JIT/vmap-compatible reimplementation of the ARC-AGI-3 environment *contract*
(64x64 frames, values 0-15, the RESET/ACTION1..ACTION7 action enum,
available-action masks, sparse game-completion reward, level progression) plus a
subset of the ARCEngine mechanics needed for small-action-space movement games
(sprites, layered rendering, transparency, bounding-box / pixel-perfect
collision, reset, next-level, win).

This package is intentionally decoupled from the static-ARC environment
(``jaxarc.envs``) and its dataset registry: ARC-AGI-3 needs entity/sprite/
game-state fields and a 16-color 64x64 grid, not ARC grid-editing state.

Quickstart::

    import jax
    from jaxarc.arcagi3 import make_arcagi3

    env, params = make_arcagi3("simple_maze")
    state, ts = env.reset(jax.random.PRNGKey(0))
    state, ts = env.step(state, 4)  # ACTION4 = move right

Non-goals in v1: no runtime ``arc_agi``/``arcengine`` dependency, no
``jax.pure_callback``, no ACTION6 coordinate clicks, no animation sub-frames, no
full public-suite parity, and reward is 1.0 only on game completion.
"""

from __future__ import annotations

from typing import Any

from .constants import (
    ACTION1,
    ACTION2,
    ACTION3,
    ACTION4,
    ACTION5,
    ACTION6,
    ACTION7,
    GAME_OVER,
    GRID_SIZE,
    NOT_FINISHED,
    NOT_PLAYED,
    NUM_ACTIONS,
    NUM_COLORS,
    RESET,
    WIN,
)
from .env import ArcAgi3Environment, reset, step
from .types import ArcAgi3Params, ArcAgi3State
from .wrappers import DiscreteMovementWrapper


def make_arcagi3(
    game_id: str, **kwargs: Any
) -> tuple[ArcAgi3Environment, ArcAgi3Params]:
    """Create an ARC-AGI-3 environment and its params for a built-in game.

    Args:
        game_id: A registered game id (e.g. ``"simple_maze"``). An
            ``"arcagi3:"`` prefix is accepted and stripped.
        **kwargs: Forwarded to the game's param builder (e.g. ``max_steps``).

    Returns:
        ``(env, params)`` — an :class:`ArcAgi3Environment` and its
        :class:`ArcAgi3Params`, mirroring the ``(env, params)`` return of
        ``jaxarc.registration.make``.

    Raises:
        ValueError: If ``game_id`` is not a known built-in game.
    """
    from .games import GAME_BUILDERS

    key = game_id.split(":", 1)[1] if game_id.startswith("arcagi3:") else game_id
    if key not in GAME_BUILDERS:
        available = ", ".join(sorted(GAME_BUILDERS))
        msg = f"Unknown ARC-AGI-3 game '{game_id}'. Available: {available}"
        raise ValueError(msg)

    params = GAME_BUILDERS[key](**kwargs)
    return ArcAgi3Environment(params), params


__all__ = [
    "ACTION1",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION5",
    "ACTION6",
    "ACTION7",
    "GAME_OVER",
    "GRID_SIZE",
    "NOT_FINISHED",
    "NOT_PLAYED",
    "NUM_ACTIONS",
    "NUM_COLORS",
    "RESET",
    "WIN",
    "ArcAgi3Environment",
    "ArcAgi3Params",
    "ArcAgi3State",
    "DiscreteMovementWrapper",
    "make_arcagi3",
    "reset",
    "step",
]
