"""Merge — third ARC-AGI-3 movement-game port.

Ports ``examples/merge.py`` from ``arcprize/ARCEngine`` (MIT, public), verified
against ``main``. The player (a single pixel) navigates a 16x16 room and absorbs
``"merge"``-tagged sprites on contact, growing into a composite shape. The level
is won when the merged player's rendered pixels exactly match the ``"target"``
sprite's; it is lost if only one merge-sprite remains without a win.

New mechanics vs. the maze games (built on the per-sprite pixel buffer added in
``ArcAgi3State.sprite_pixels``):

- **Runtime sprite merging**: colliding with a merge sprite composites it into the
  player's slot (``engine.merge_into``) and moves the player one more cell.
- **Pixel-equality win**: ``rendering.region_equal(player, target)`` — the merged
  player's raw-camera region must equal the target's (same extent AND content).

Faithful reproduction of ``Merge.step``:

    others = try_move("player", dx, dy)
    for collide in others:
        if "merge" in collide.tags:
            player = player.merge(collide); remove both; add merged; player.move(dx,dy)
    if check_win_condition(): next_level()
    elif count(merge sprites) <= 1: lose()

`merge_detach` / `ACTION5` is a separate, later increment.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int

from ..constants import ACTION1, ACTION2, ACTION3, ACTION4, NUM_ACTIONS, RESET
from ..engine import (
    eqx_replace,
    lose_game,
    merge_into,
    next_level_or_win,
    player_index,
    try_move_player,
)
from ..rendering import region_equal
from ..types import ArcAgi3Params, ArcAgi3State
from .base import action_delta, apply_reset

# --- Sprite kinds ----------------------------------------------------------
KIND_PLAYER = 0  # [[9]], tag "merge" (the absorber)
KIND_WALLS = 1  # sprite-1: 16x16 room walls, untagged (blocks, never merged)
KIND_S2 = 2  # sprite-2, merge
KIND_S3 = 3  # sprite-3, merge
KIND_S4 = 4  # sprite-4, target
KIND_S5 = 5  # sprite-5, merge
KIND_S6 = 6  # sprite-6, target
KIND_S7 = 7  # sprite-7, target
NUM_KINDS = 8

# sprite_h/w = grid size (16): a guaranteed upper bound on any merged bbox, so
# merges never clip (see the build-plan Phase 0 findings).
SPRITE_H = 16
SPRITE_W = 16

BACKGROUND_COLOR = 1
PADDING_COLOR = 3

# Merge- vs target-tagged kinds (from the ARCEngine source).
MERGE_KINDS = (KIND_PLAYER, KIND_S2, KIND_S3, KIND_S5)
TARGET_KINDS = (KIND_S4, KIND_S6, KIND_S7)

_T = -1  # TRANSPARENT
_W = 5  # wall color

# Room walls (sprite-1): 16x16 border-ish maze; unrotated form.
_WALLS = [
    [_W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _T, _T, _T, _T, _T, _T],
    [_W, _T, _T, _T, _T, _T, _T, _T, _T, _W, _T, _T, _T, _T, _T, _T],
    [_W, _T, _T, _T, _T, _T, _T, _T, _T, _W, _T, _T, _T, _T, _T, _T],
    [_W, _T, _T, _T, _T, _T, _T, _T, _T, _W, _T, _T, _T, _T, _T, _T],
    [_W, _T, _T, _T, _T, _T, _T, _T, _T, _W, _T, _T, _T, _T, _T, _T],
    [_W, _T, _T, _T, _T, _T, _T, _T, _T, _W, _T, _T, _T, _T, _T, _T],
    [_W, _T, _T, _T, _T, _T, _T, _T, _T, _W, _W, _W, _W, _W, _W, _W],
    [_W, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _W],
    [_W, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _W],
    [_W, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _W],
    [_W, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _W],
    [_W, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _W],
    [_W, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _W],
    [_W, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _W],
    [_W, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _W],
    [_W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W],
]

_PIXELS = {
    KIND_PLAYER: [[9]],
    KIND_WALLS: _WALLS,
    KIND_S2: [[14, 14], [14, 14]],
    KIND_S3: [[8, 8], [_T, 8]],
    KIND_S4: [[8, 8], [9, 8]],
    KIND_S5: [[11], [11], [11]],
    KIND_S6: [[11, 8, 8], [11, 9, 8], [11, _T, _T]],
    KIND_S7: [[_T, 8, 8, 11], [14, 14, 8, 11], [14, 14, 9, 11]],
}

# Per-level: (camera w/h, [(kind, x, y, rotation180), ...]). Slot order matches the
# ARCEngine source's sprite list per level (player first).
_LEVELS = [
    (
        16,
        16,
        [
            (KIND_PLAYER, 3, 10, False),
            (KIND_WALLS, 0, 0, False),
            (KIND_S3, 4, 5, False),
            (KIND_S4, 12, 2, False),
        ],
    ),
    (
        16,
        16,
        [
            (KIND_PLAYER, 3, 12, False),
            (KIND_WALLS, 0, 0, False),
            (KIND_S3, 7, 9, False),
            (KIND_S5, 2, 3, False),
            (KIND_S6, 11, 1, False),
        ],
    ),
    (
        16,
        16,
        [
            (KIND_PLAYER, 12, 9, False),
            (KIND_WALLS, 0, 0, True),  # rotated 180 in level 3
            (KIND_S2, 12, 3, False),
            (KIND_S3, 8, 5, False),
            (KIND_S5, 4, 2, False),
            (KIND_S7, 1, 11, False),
        ],
    ),
]

MAX_SPRITES = 6  # level 3 has 6 sprites


# ---------------------------------------------------------------------------
# Transition logic
# ---------------------------------------------------------------------------


def _target_index(state: ArcAgi3State, params: ArcAgi3Params) -> Int[Array, ""]:
    """Index of the (first) active target-kind sprite."""
    is_target = state.sprite_active & params.sprite_is_target[state.sprite_kind]
    return jnp.argmax(is_target).astype(jnp.int32)


def _merge_move(
    params: ArcAgi3Params, state: ArcAgi3State, dx: Int[Array, ""], dy: Int[Array, ""]
) -> ArcAgi3State:
    """Move the player; absorb every merge sprite it collides with (port of step)."""
    moved, collisions = try_move_player(state, params, dx, dy)
    is_merge = params.sprite_is_merge[moved.sprite_kind]

    def body(state: ArcAgi3State, i: Int[Array, ""]) -> tuple[ArcAgi3State, None]:
        # Absorb sprite i if the player collided with it and it's a merge sprite.
        do = collisions[i] & is_merge[i] & state.sprite_active[i]

        def absorb(s: ArcAgi3State) -> ArcAgi3State:
            p = player_index(s, params)
            s = merge_into(s, p, i)
            # ARCEngine moves the merged player one more cell after each absorb.
            m, _ = try_move_player(s, params, dx, dy)
            return m

        return jax.lax.cond(do, absorb, lambda s: s, state), None

    idxs = jnp.arange(state.sprite_x.shape[0])
    moved, _ = jax.lax.scan(body, moved, idxs)
    return moved


def _count_merge_sprites(params: ArcAgi3Params, state: ArcAgi3State) -> Int[Array, ""]:
    """Number of active merge-tagged sprites (incl. the player)."""
    is_merge = params.sprite_is_merge[state.sprite_kind]
    return jnp.sum((state.sprite_active & is_merge).astype(jnp.int32))


def transition_merge(
    params: ArcAgi3Params, state: ArcAgi3State, action: Int[Array, ""]
) -> ArcAgi3State:
    """Merge transition (see module docstring)."""

    def on_reset(s: ArcAgi3State) -> ArcAgi3State:
        return apply_reset(params, s)

    def on_action(s: ArcAgi3State) -> ArcAgi3State:
        def live(s: ArcAgi3State) -> ArcAgi3State:
            s = eqx_replace(s, action_count=s.action_count + 1)
            dx, dy = action_delta(action)
            s = _merge_move(params, s, dx, dy)

            # Win if the merged player's region equals the target's; else lose when
            # only the player (<=1 merge sprite) remains. Checked in this order.
            p = player_index(s, params)
            t = _target_index(s, params)
            won = region_equal(params, s, p, t)

            def on_win(s: ArcAgi3State) -> ArcAgi3State:
                return next_level_or_win(s, params)

            def on_not_win(s: ArcAgi3State) -> ArcAgi3State:
                lose = _count_merge_sprites(params, s) <= 1
                return jax.lax.cond(
                    lose, lambda x: lose_game(x, params), lambda x: x, s
                )

            return jax.lax.cond(won, on_win, on_not_win, s)

        return jax.lax.cond(s.done, lambda x: x, live, s)

    return jax.lax.cond(action == RESET, on_reset, on_action, state)


# ---------------------------------------------------------------------------
# Params builder
# ---------------------------------------------------------------------------


def _pad_tile(rows: list[list[int]], *, rotate180: bool = False) -> np.ndarray:
    """Embed a pixel grid at the top-left of a (SPRITE_H, SPRITE_W) tile."""
    arr = np.array(rows, dtype=np.int32)
    if rotate180:
        arr = np.rot90(arr, 2)
    tile = np.full((SPRITE_H, SPRITE_W), _T, dtype=np.int32)
    h, w = arr.shape
    tile[:h, :w] = arr
    return tile


def _build_sprite_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-kind (pixels, layer, blocking). All PIXEL_PERFECT (ARCEngine default).

    Note: level 3 rotates the walls 180; that is baked per-sprite at reset (the
    walls kind's tile here is the unrotated form, and the level layout flags the
    rotation — handled in _build_levels by storing rotated pixels per level).
    """
    from ..constants import PIXEL_PERFECT

    pixels = np.stack([_pad_tile(_PIXELS[k]) for k in range(NUM_KINDS)], axis=0)
    layer = np.zeros((NUM_KINDS,), dtype=np.int32)  # merge has no layering
    blocking = np.full((NUM_KINDS,), PIXEL_PERFECT, dtype=np.int32)
    return pixels, layer, blocking


def _build_kind_flags() -> dict[str, np.ndarray]:
    is_merge = np.zeros((NUM_KINDS,), dtype=bool)
    is_target = np.zeros((NUM_KINDS,), dtype=bool)
    for k in MERGE_KINDS:
        is_merge[k] = True
    for k in TARGET_KINDS:
        is_target[k] = True
    return {"is_merge": is_merge, "is_target": is_target}


def _build_levels() -> dict[str, np.ndarray]:
    """Per-level initial sprite layout, incl. per-sprite rotated pixel buffers."""
    n = len(_LEVELS)
    x = np.zeros((n, MAX_SPRITES), dtype=np.int32)
    y = np.zeros((n, MAX_SPRITES), dtype=np.int32)
    kind = np.zeros((n, MAX_SPRITES), dtype=np.int32)
    active = np.zeros((n, MAX_SPRITES), dtype=bool)
    visible = np.zeros((n, MAX_SPRITES), dtype=bool)
    collidable = np.zeros((n, MAX_SPRITES), dtype=bool)
    # Per-(level, slot) initial pixels: needed because level 3 rotates the walls,
    # so a slot's initial buffer can differ from its kind's default tile.
    init_pixels = np.tile(
        np.full((SPRITE_H, SPRITE_W), _T, dtype=np.int32), (n, MAX_SPRITES, 1, 1)
    )

    for li, (_cw, _ch, sprites) in enumerate(_LEVELS):
        for si, (k, sx, sy, rot) in enumerate(sprites):
            kind[li, si] = k
            x[li, si] = sx
            y[li, si] = sy
            active[li, si] = True
            visible[li, si] = True
            collidable[li, si] = True
            init_pixels[li, si] = _pad_tile(_PIXELS[k], rotate180=rot)

    return {
        "x": x,
        "y": y,
        "kind": kind,
        "active": active,
        "visible": visible,
        "collidable": collidable,
        "init_pixels": init_pixels,
    }


def make_params(*, transition_id: int, max_steps: int = 512) -> ArcAgi3Params:
    """Build :class:`ArcAgi3Params` for the merge game (3 levels)."""
    pixels, layer, blocking = _build_sprite_tables()
    flags = _build_kind_flags()
    levels = _build_levels()
    cam_w = np.array([lv[0] for lv in _LEVELS], dtype=np.int32)
    cam_h = np.array([lv[1] for lv in _LEVELS], dtype=np.int32)

    avail = np.zeros((NUM_ACTIONS,), dtype=bool)
    for a in (RESET, ACTION1, ACTION2, ACTION3, ACTION4):
        avail[a] = True

    return ArcAgi3Params(
        height=64,
        width=64,
        max_steps=max_steps,
        max_sprites=MAX_SPRITES,
        max_levels=len(_LEVELS),
        sprite_h=SPRITE_H,
        sprite_w=SPRITE_W,
        player_kind=KIND_PLAYER,
        goal_kind=KIND_S4,  # unused by merge (win is pixel-equality), set to a target
        background=BACKGROUND_COLOR,
        letter_box=PADDING_COLOR,
        reset_mode="level",
        level_cam_w=jnp.asarray(cam_w),
        level_cam_h=jnp.asarray(cam_h),
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
        game_id="merge",
        transition_id=transition_id,
        sprite_is_merge=jnp.asarray(flags["is_merge"]),
        sprite_is_target=jnp.asarray(flags["is_target"]),
        # Per-(level, slot) initial pixels — level 3 rotates the room walls 180.
        init_sprite_pixels=jnp.asarray(levels["init_pixels"]),
    )


__all__ = ["make_params", "transition_merge"]
