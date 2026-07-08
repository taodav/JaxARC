"""Canonical JAX-side trace runner shared by golden and parity tests.

Replays a list of action ids through the pure-JAX ARC-AGI-3 environment and
records, per step, the rendered frame plus scalar state/timestep metadata. This
is the single source of truth for "what the JAX implementation produces for a
trace", used by both:

- Tier-2 golden tests (compare against committed fixtures), and
- Tier-3 parity tests (compare against the official ``arcengine``).

Kept in ``src`` (not ``test``) so it can be imported without a ``test`` -> ``src``
dependency and so downstream users can reproduce traces.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

from ..env import reset as env_reset
from ..env import step as env_step

# SimpleMaze reset is deterministic and ignores the key; fixed for reproducibility.
DEFAULT_KEY = jax.random.PRNGKey(0)

PLAYER_KIND = 0


def player_xy(state) -> tuple[int, int]:
    """(x, y) of the first player-kind sprite, matching engine.player_index."""
    kinds = [int(k) for k in state.sprite_kind]
    p = kinds.index(PLAYER_KIND)
    return int(state.sprite_x[p]), int(state.sprite_y[p])


def state_record(action: int | None, state, ts) -> dict[str, Any]:
    """Scalar snapshot of one step (JSON-serializable)."""
    px, py = player_xy(state)
    return {
        "action": None if action is None else int(action),
        "game_state": int(state.game_state),
        "level_index": int(state.level_index),
        "levels_completed": int(state.levels_completed),
        "action_count": int(state.action_count),
        "step_count": int(state.step_count),
        "reward": float(ts.reward),
        "step_type": int(ts.step_type),
        "player_x": px,
        "player_y": py,
    }


def run_trace(
    params, actions: list[int], *, key=DEFAULT_KEY
) -> tuple[np.ndarray, list[dict]]:
    """Replay ``actions`` and return ``(frames, records)``.

    Frames: ``int32[T+1, 64, 64]`` — ``frames[0]`` is the reset observation and
    ``frames[i]`` is the observation after ``actions[i-1]``.
    Records: list of ``T+1`` scalar-state dicts (see :func:`state_record`).
    """
    state, ts = env_reset(params, key)

    def observe(ts) -> np.ndarray:
        return np.asarray(ts.observation[..., 0], dtype=np.int32)

    frames = [observe(ts)]
    records = [state_record(None, state, ts)]
    for a in actions:
        state, ts = env_step(params, state, a)
        frames.append(observe(ts))
        records.append(state_record(a, state, ts))

    return np.stack(frames, axis=0), records


__all__ = ["DEFAULT_KEY", "player_xy", "run_trace", "state_record"]
