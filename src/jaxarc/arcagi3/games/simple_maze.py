"""SimpleMaze — first ARC-AGI-3 movement-game port.

Direct translation of ``examples/simple_maze.py`` from ``arcprize/ARCEngine``
(MIT-licensed, public), verified against ``main``. The player navigates a maze to
reach the exit; touching the exit advances to the next level, and clearing the
last level wins.

Original sprite data (x = column, y = row; origin top-left):
- ``player``: pixels ``[[8]]``, ``BOUNDING_BOX``, layer 0.
- ``exit``:   pixels ``[[9]]``, ``BOUNDING_BOX``, layer 0.
- ``maze_1``: 8x8 walls (color 5, ``-1`` = open), ``PIXEL_PERFECT``, layer -1.
- ``maze_2``: 12x12 walls, ``PIXEL_PERFECT``, layer -1.
- Level 1 (grid 8x8):  maze_1, player @ (1,1), exit @ (6,6).
- Level 2 (grid 12x12): maze_2, player @ (1,1), exit @ (10,10).
- Camera: background 0, letter-box (padding) 3.

The mechanic is pure move-and-reach-goal, so this game reuses
:func:`jaxarc.arcagi3.games.base.transition_movement` unchanged; only the static
:class:`ArcAgi3Params` differ.

Rendering note (documented divergence): v1 composites sprites at their native
coordinates onto a 64x64 canvas (no per-level camera upscaling/letterboxing).
Official-frame parity via :func:`jaxarc.arcagi3.rendering.scale_and_letterbox` is
deferred to the camera-parity milestone.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..constants import (
    ACTION1,
    ACTION2,
    ACTION3,
    ACTION4,
    BOUNDING_BOX,
    NUM_ACTIONS,
    PIXEL_PERFECT,
    RESET,
    TRANSPARENT,
)
from ..types import ArcAgi3Params

# --- Sprite kinds ----------------------------------------------------------
KIND_PLAYER = 0
KIND_EXIT = 1
KIND_MAZE1 = 2
KIND_MAZE2 = 3
NUM_KINDS = 4

SPRITE_H = 12
SPRITE_W = 12

_MAZE_1 = [
    [5, 5, 5, 5, 5, 5, 5, 5],
    [5, -1, -1, -1, 5, -1, -1, 5],
    [5, -1, 5, -1, 5, -1, 5, 5],
    [5, -1, 5, -1, -1, -1, -1, 5],
    [5, -1, 5, 5, 5, 5, -1, 5],
    [5, -1, -1, -1, -1, 5, -1, 5],
    [5, 5, 5, 5, -1, -1, -1, 5],
    [5, 5, 5, 5, 5, 5, 5, 5],
]

_MAZE_2 = [
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, 5],
    [5, -1, 5, -1, 5, -1, 5, 5, 5, 5, -1, 5],
    [5, -1, 5, -1, -1, -1, -1, -1, -1, 5, -1, 5],
    [5, -1, 5, 5, 5, 5, 5, 5, -1, 5, -1, 5],
    [5, -1, -1, -1, -1, -1, -1, 5, -1, 5, -1, 5],
    [5, 5, 5, 5, 5, 5, -1, 5, -1, 5, -1, 5],
    [5, -1, -1, -1, -1, 5, -1, 5, -1, 5, -1, 5],
    [5, -1, 5, 5, -1, 5, -1, 5, -1, 5, -1, 5],
    [5, -1, 5, -1, -1, 5, -1, -1, -1, 5, -1, 5],
    [5, -1, -1, -1, 5, 5, 5, 5, 5, 5, -1, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
]

BACKGROUND_COLOR = 0
PADDING_COLOR = 3


def _pad_tile(rows: list[list[int]]) -> np.ndarray:
    """Embed a (h, w) pixel grid at the top-left of a (SPRITE_H, SPRITE_W) tile."""
    tile = np.full((SPRITE_H, SPRITE_W), TRANSPARENT, dtype=np.int32)
    arr = np.array(rows, dtype=np.int32)
    h, w = arr.shape
    tile[:h, :w] = arr
    return tile


def _build_sprite_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-kind (pixels, layer, blocking) tables."""
    pixels = np.stack(
        [
            _pad_tile([[8]]),  # KIND_PLAYER
            _pad_tile([[9]]),  # KIND_EXIT
            _pad_tile(_MAZE_1),  # KIND_MAZE1
            _pad_tile(_MAZE_2),  # KIND_MAZE2
        ],
        axis=0,
    )
    layer = np.array([0, 0, -1, -1], dtype=np.int32)
    blocking = np.array(
        [BOUNDING_BOX, BOUNDING_BOX, PIXEL_PERFECT, PIXEL_PERFECT], dtype=np.int32
    )
    return pixels, layer, blocking


def _build_levels() -> dict[str, np.ndarray]:
    """Per-level initial sprite layout (shape [max_levels, max_sprites]).

    Slot order per level: [maze, player, exit]. All three slots are active.
    """
    max_levels = 2
    max_sprites = 3

    x = np.zeros((max_levels, max_sprites), dtype=np.int32)
    y = np.zeros((max_levels, max_sprites), dtype=np.int32)
    kind = np.zeros((max_levels, max_sprites), dtype=np.int32)
    active = np.ones((max_levels, max_sprites), dtype=bool)
    visible = np.ones((max_levels, max_sprites), dtype=bool)
    collidable = np.ones((max_levels, max_sprites), dtype=bool)

    # Level 0: maze_1 @ (0,0), player @ (1,1), exit @ (6,6)
    kind[0] = [KIND_MAZE1, KIND_PLAYER, KIND_EXIT]
    x[0] = [0, 1, 6]
    y[0] = [0, 1, 6]

    # Level 1: maze_2 @ (0,0), player @ (1,1), exit @ (10,10)
    kind[1] = [KIND_MAZE2, KIND_PLAYER, KIND_EXIT]
    x[1] = [0, 1, 10]
    y[1] = [0, 1, 10]

    return {
        "x": x,
        "y": y,
        "kind": kind,
        "active": active,
        "visible": visible,
        "collidable": collidable,
    }


def make_params(*, transition_id: int = 0, max_steps: int = 512) -> ArcAgi3Params:
    """Build :class:`ArcAgi3Params` for SimpleMaze.

    Args:
        transition_id: Index of this game's transition fn in ``games.TRANSITIONS``.
            SimpleMaze uses the generic movement transition (registered first).
        max_steps: Episode truncation limit (RL convenience; not an engine concept).
    """
    pixels, layer, blocking = _build_sprite_tables()
    levels = _build_levels()

    # Available actions: RESET + movement (ACTION1..ACTION4). No ACTION5/6/7.
    avail = np.zeros((NUM_ACTIONS,), dtype=bool)
    for a in (RESET, ACTION1, ACTION2, ACTION3, ACTION4):
        avail[a] = True

    return ArcAgi3Params(
        height=64,
        width=64,
        max_steps=max_steps,
        max_sprites=3,
        max_levels=2,
        sprite_h=SPRITE_H,
        sprite_w=SPRITE_W,
        player_kind=KIND_PLAYER,
        goal_kind=KIND_EXIT,
        background=BACKGROUND_COLOR,
        letter_box=PADDING_COLOR,
        reset_mode="level",
        # Camera resizes to each level's grid_size = (width, height):
        # level 0 = 8x8, level 1 = 12x12 (see the original Level(grid_size=...)).
        level_cam_w=jnp.asarray([8, 12], dtype=jnp.int32),
        level_cam_h=jnp.asarray([8, 12], dtype=jnp.int32),
        init_sprite_x=jnp.asarray(levels["x"]),
        init_sprite_y=jnp.asarray(levels["y"]),
        init_sprite_kind=jnp.asarray(levels["kind"]),
        init_sprite_active=jnp.asarray(levels["active"]),
        init_sprite_visible=jnp.asarray(levels["visible"]),
        init_sprite_collidable=jnp.asarray(levels["collidable"]),
        sprite_pixels=jnp.asarray(pixels),
        sprite_layer=jnp.asarray(layer),
        sprite_blocking=jnp.asarray(blocking),
        available_actions=jnp.asarray(avail),
        game_id="simple_maze",
        transition_id=transition_id,
    )


__all__ = ["KIND_EXIT", "KIND_MAZE1", "KIND_MAZE2", "KIND_PLAYER", "make_params"]
