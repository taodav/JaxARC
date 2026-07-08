"""Shared transition logic for movement-only ARC-AGI-3 games.

``transition_movement`` is a generic, pure ``(params, state, action) -> state``
function that reproduces the control flow of a SimpleMaze-style
``ARCBaseGame.step`` + ``perform_action`` (verified against ARCEngine ``main``):

- ``RESET`` -> ``handle_reset`` semantics (full vs. level reset).
- When the game is already terminal (``WIN``/``GAME_OVER``) a non-RESET action is
  a no-op (ARCEngine returns an empty frame and leaves state untouched).
- ``ACTION1..ACTION4`` -> move the player one cell via :func:`try_move_player`
  (reverted on collision); any non-RESET action increments ``action_count``,
  matching ``_set_action``.
- Colliding with a goal-kind sprite triggers :func:`advance_level` (next level, or
  WIN on the last level).

Games whose entire mechanic is "move the player, reach the goal, advance" need no
bespoke transition — they register :func:`transition_movement` directly and only
supply their own :class:`ArcAgi3Params`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int

from ..constants import (
    ACTION1,
    ACTION2,
    ACTION3,
    ACTION4,
    RESET,
    WIN,
)
from ..engine import (
    advance_level,
    eqx_replace,
    reset_full,
    reset_to_level,
    try_move_player,
)
from ..types import ArcAgi3Params, ArcAgi3State


def action_delta(action: Int[Array, ""]) -> tuple[Int[Array, ""], Int[Array, ""]]:
    """Map an action id to a (dx, dy) movement delta; (0, 0) for non-movement.

    ACTION1 up (dy=-1), ACTION2 down (dy=+1), ACTION3 left (dx=-1),
    ACTION4 right (dx=+1). Reserved actions (5/6/7) do not move in v1.
    """
    dx = jnp.where(action == ACTION3, -1, jnp.where(action == ACTION4, 1, 0))
    dy = jnp.where(action == ACTION1, -1, jnp.where(action == ACTION2, 1, 0))
    return dx.astype(jnp.int32), dy.astype(jnp.int32)


def apply_reset(params: ArcAgi3Params, state: ArcAgi3State) -> ArcAgi3State:
    """Port of ``ARCBaseGame.handle_reset`` (default, non-ONLY_RESET_LEVELS path).

    ``params.reset_mode`` is static, so the top-level choice is a Python branch:
    - "full": always a full reset.
    - "level": full reset when no actions have been taken yet or the game was
      already won, otherwise a level reset.
    """
    if params.reset_mode == "full":
        return reset_full(state, params)

    do_full = (state.action_count == 0) | (state.game_state == WIN)
    return jax.lax.cond(
        do_full,
        lambda s: reset_full(s, params),
        lambda s: reset_to_level(s, params),
        state,
    )


def _movement_action(
    params: ArcAgi3Params, state: ArcAgi3State, action: Int[Array, ""]
) -> ArcAgi3State:
    """Apply a single non-RESET action: move (if directional), then check the goal."""
    dx, dy = action_delta(action)
    moved, collisions = try_move_player(state, params, dx, dy)

    # Any non-RESET action counts (mirrors _set_action incrementing action_count).
    moved = eqx_replace(moved, action_count=moved.action_count + 1)

    goal_mask = moved.sprite_active & (moved.sprite_kind == params.goal_kind)
    hit_goal = jnp.any(collisions & goal_mask)

    return jax.lax.cond(
        hit_goal,
        lambda s: advance_level(s, params),
        lambda s: s,
        moved,
    )


def transition_movement(
    params: ArcAgi3Params, state: ArcAgi3State, action: Int[Array, ""]
) -> ArcAgi3State:
    """Generic movement-game transition (see module docstring)."""

    def on_reset(s: ArcAgi3State) -> ArcAgi3State:
        return apply_reset(params, s)

    def on_action(s: ArcAgi3State) -> ArcAgi3State:
        # Terminal games ignore non-RESET actions (empty frame in ARCEngine).
        return jax.lax.cond(
            s.done,
            lambda x: x,
            lambda x: _movement_action(params, x, action),
            s,
        )

    return jax.lax.cond(action == RESET, on_reset, on_action, state)


__all__ = ["action_delta", "apply_reset", "transition_movement"]
