"""Named constants for the ARC-AGI-3 runtime subset.

These are intentionally **separate** from ``jaxarc.constants`` (which defines
static-ARC values: 10 colors, 30x30 grids, cells in ``[-1, 9]``). ARC-AGI-3 uses
16 colors on a 64x64 grid, so reusing the static-ARC constants or the
``jaxarc.types.Grid`` wrapper (which validates cells into ``[-1, 9]``) would be
incorrect. This module is dependency-free so it can be imported anywhere in the
``arcagi3`` package without circular imports.

References (verified against ``main``):
- Grid size / value range: ARC-AGI-3 docs game-schema (64x64, values 0-15).
- Action enum / GameState: ``arcengine/enums.py``.
- Transparent pixel sentinel ``-1``: ``arcengine`` sprites/camera use ``-1`` for
  transparency internally; rendered frames are clamped to ``0..15``.
"""

from __future__ import annotations

from typing import Final

# --- Grid / color contract -------------------------------------------------
GRID_SIZE: Final[int] = 64
"""ARC-AGI-3 frames are always rendered to 64x64 (see ``arcengine`` Camera)."""

NUM_COLORS: Final[int] = 16
"""Cell values are integers ``0..15`` (game-schema)."""

TRANSPARENT: Final[int] = -1
"""Fully-transparent sprite pixel: not drawn and not collidable.

Matches ``arcengine``: collision uses ``pixel != -1`` and rendering paints
``pixel >= 0``, so ``-1`` is skipped by both. The final rendered frame never
contains ``-1`` (transparent regions fall through to the camera background).
"""

SOLID_INVISIBLE: Final[int] = -2
"""Invisible-wall sprite pixel: solid for collision but not drawn.

ARCEngine collides where ``pixel != -1`` (so ``-2`` blocks) but paints only
``pixel >= 0`` (so ``-2`` is not rendered). Used by complex_maze's mazes.
"""

# --- Action enum (matches arcengine.GameAction ids) ------------------------
RESET: Final[int] = 0
ACTION1: Final[int] = 1  # semantically "up"
ACTION2: Final[int] = 2  # semantically "down"
ACTION3: Final[int] = 3  # semantically "left"
ACTION4: Final[int] = 4  # semantically "right"
ACTION5: Final[int] = 5  # interact / select / rotate (reserved in v1)
ACTION6: Final[int] = 6  # complex action, x,y in 0..63 (reserved in v1)
ACTION7: Final[int] = 7  # undo (reserved in v1)

NUM_ACTIONS: Final[int] = 8
"""Total size of the ARC-AGI-3 action enum (RESET + ACTION1..ACTION7)."""

# --- GameState enum (matches arcengine.GameState, encoded as int) ----------
# ``arcengine.GameState`` is a str enum; these int codes are a JaxARC-local
# encoding for JAX compatibility. Order is stable and documented.
NOT_PLAYED: Final[int] = 0
NOT_FINISHED: Final[int] = 1
WIN: Final[int] = 2
GAME_OVER: Final[int] = 3

GAME_STATE_NAMES: Final[dict[int, str]] = {
    NOT_PLAYED: "NOT_PLAYED",
    NOT_FINISHED: "NOT_FINISHED",
    WIN: "WIN",
    GAME_OVER: "GAME_OVER",
}

# --- Collision / blocking modes (matches arcengine.BlockingMode) -----------
# arcengine uses ``enum.auto()`` starting at 1; we re-encode from 0 for use as
# static ints in the JAX engine.
NOT_BLOCKED: Final[int] = 0
BOUNDING_BOX: Final[int] = 1
PIXEL_PERFECT: Final[int] = 2

# --- Movement deltas for ACTION1..ACTION4 ----------------------------------
# (dx, dy) with origin (0,0) at top-left, +y downward. Matches the SimpleMaze
# mapping: ACTION1 up (dy=-1), ACTION2 down (dy=+1), ACTION3 left (dx=-1),
# ACTION4 right (dx=+1).
MOVE_DELTAS: Final[dict[int, tuple[int, int]]] = {
    ACTION1: (0, -1),
    ACTION2: (0, 1),
    ACTION3: (-1, 0),
    ACTION4: (1, 0),
}


__all__ = [
    "ACTION1",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION5",
    "ACTION6",
    "ACTION7",
    "BOUNDING_BOX",
    "GAME_OVER",
    "GAME_STATE_NAMES",
    "GRID_SIZE",
    "MOVE_DELTAS",
    "NOT_BLOCKED",
    "NOT_FINISHED",
    "NOT_PLAYED",
    "NUM_ACTIONS",
    "NUM_COLORS",
    "PIXEL_PERFECT",
    "RESET",
    "TRANSPARENT",
    "WIN",
]
