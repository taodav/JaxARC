"""Adapter that drives the official ARCEngine and mirrors the JAX trace schema.

**Test-only / optional.** ``arcengine`` is never a runtime dependency of JaxARC;
it is imported lazily here and only used when a parity test explicitly opts in
(``RUN_ARCAGI3_PARITY=1`` and ``arcengine`` installed). See
``test/arcagi3/test_parity_simple_maze.py``.

The adapter replays a trace of ARC-AGI-3 action ids against a real
``ARCBaseGame`` instance via ``perform_action(..., raw=True)`` and returns frames
and per-step records in the **same shape/keys** as
:func:`jaxarc.arcagi3.parity.trace_runner.run_trace`, so the two can be compared
field-by-field.

Frame extraction (documented v1 divergence): ``FrameDataRaw.frame`` is a *stack*
of 64x64 grids (1..N animation frames per action). v1 compares the **final**
frame of each action, so we take ``frame[-1]``. RESET's initial frame is taken
from the wrapper's reset return.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def arcengine_available() -> bool:
    """Whether the optional ``arcengine`` package can be imported."""
    import importlib.util

    return importlib.util.find_spec("arcengine") is not None


def _game_state_to_int(state: Any) -> int:
    """Map ``arcengine.GameState`` (str enum) to our int codes in constants.py."""
    from ..constants import GAME_OVER, NOT_FINISHED, NOT_PLAYED, WIN

    name = state.value if hasattr(state, "value") else str(state)
    return {
        "NOT_PLAYED": NOT_PLAYED,
        "NOT_FINISHED": NOT_FINISHED,
        "WIN": WIN,
        "GAME_OVER": GAME_OVER,
    }[name]


def _final_frame(frame_stack: list) -> np.ndarray:
    """Take the last 64x64 grid of an action's frame stack as an int32 array."""
    return np.asarray(frame_stack[-1], dtype=np.int32)


def run_official_trace(game, actions: list[int]) -> tuple[np.ndarray, list[dict]]:
    """Replay ``actions`` against an ARCEngine game; return ``(frames, records)``.

    Args:
        game: An instantiated ``ARCBaseGame`` subclass (e.g. ``SimpleMaze()``).
        actions: ARC-AGI-3 action ids (0=RESET, 1..7=ACTIONn).

    Returns:
        ``frames``: ``int32[T+1, 64, 64]`` — index 0 is the post-reset frame.
        ``records``: ``T+1`` dicts with the subset of fields both engines expose:
        ``action, game_state, levels_completed, step_type``.
    """
    from arcengine import ActionInput, GameAction

    def record(action: int | None, frame_obj) -> dict:
        return {
            "action": None if action is None else int(action),
            "game_state": _game_state_to_int(frame_obj.state),
            "levels_completed": int(frame_obj.levels_completed),
        }

    # RESET to obtain the initial frame/state.
    reset_frame = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    frames = [_final_frame(reset_frame.frame)]
    records = [record(None, reset_frame)]

    for a in actions:
        action_input = ActionInput(id=GameAction.from_id(int(a)))
        frame_obj = game.perform_action(action_input, raw=True)
        # A terminal engine returns an empty frame stack for non-RESET actions;
        # reuse the previous frame so indices stay aligned with the JAX trace.
        if len(frame_obj.frame) == 0:
            frames.append(frames[-1])
        else:
            frames.append(_final_frame(frame_obj.frame))
        records.append(record(a, frame_obj))

    return np.stack(frames, axis=0), records


__all__ = ["arcengine_available", "run_official_trace"]
