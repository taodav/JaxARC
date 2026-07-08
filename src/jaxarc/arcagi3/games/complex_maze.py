"""ComplexMaze — second ARC-AGI-3 movement-game port (Increment A: levels 1-4).

Ports ``examples/complex_maze.py`` from ``arcprize/ARCEngine`` (MIT, public),
verified against ``main``. Extends SimpleMaze with:

- **Invisible walls** (pixel value ``-2``): solid for collision, not drawn.
- **Pushable blocks**: moving into a ``block`` pushes it one cell if free; the
  player then follows into the vacated cell. Two same-kind blocks that collide
  during a push **annihilate** (both removed).
- **Energy budget** (``lose()`` -> GAME_OVER): a per-level move budget rendered as
  a border of 63 pills; exceeding it loses the game.

Increment A implements levels 1-4 (no moving maze). The moving-maze mechanic of
level 5 is deferred to Increment B; this port registers 4 levels.

Faithful reproduction of ``ComplexMaze.step`` / ``_try_pushing_move``:

    _try_pushing_move(dx, dy):
        collided = try_move("player", dx, dy)
        if exit in collided:            next_level()          # score += 1 always
        for sprite in collided:
            if sprite is a block:
                pushed = try_move_sprite(sprite, dx, dy)
                if pushed collided w/ same-kind block:  remove both
                else:                                   try_move("player", dx, dy)
    step():
        _try_pushing_move(...)
        if action != RESET and energy exhausted:  lose()
        complete_action()

The order matters: the goal check uses the *first* player-move collisions; block
pushing then possibly frees a cell so the player's *second* move succeeds. Note
``next_level()`` defers the actual level load via a flag, so the goal-reaching
step renders on the old level (a discarded non-final frame) and the final frame
is the freshly-loaded next level with ``action_count`` reset to 0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int

from ..constants import (
    ACTION1,
    ACTION2,
    ACTION3,
    ACTION4,
    NUM_ACTIONS,
    PIXEL_PERFECT,
    RESET,
    SOLID_INVISIBLE,
    TRANSPARENT,
)
from ..engine import (
    eqx_replace,
    lose_game,
    next_level_or_win,
    set_sprite_removed,
    try_move_player,
    try_move_sprite,
)
from ..types import ArcAgi3Params, ArcAgi3State
from .base import action_delta, apply_reset

# --- Sprite kinds ----------------------------------------------------------
KIND_PLAYER = 0
KIND_EXIT = 1
KIND_BLOCK = 2  # "block_orange": fixed-tagged, moves with the maze
KIND_BLOCK_FLEX = 3  # "block_orange_flex": floating, pushable but not maze-linked
KIND_MAZE1 = 4
KIND_MAZE2 = 5
KIND_MAZE3 = 6
KIND_MAZE4 = 7
KIND_MAZE5 = 8
NUM_KINDS = 9
MAZE_KINDS = (KIND_MAZE1, KIND_MAZE2, KIND_MAZE3, KIND_MAZE4, KIND_MAZE5)

SPRITE_H = 12
SPRITE_W = 12

# Colors (from the ARCEngine source).
PLAYER_COLOR = 8
EXIT_COLOR = 9
BLOCK_COLOR = 12
BACKGROUND_COLOR = 0
PADDING_COLOR = 0
PILL_ON_COLOR = 6
PILL_OFF_COLOR = 3

# W = wall (5), O = open (-1), I = invisible wall (-2, solid but not drawn).
_W, _O, _I = 5, TRANSPARENT, SOLID_INVISIBLE

_MAZE_1 = [
    [_W, _W, _W, _W, _W, _W, _W, _W],
    [_W, _O, _O, _O, _W, _O, _O, _W],
    [_W, _O, _W, _O, _W, _O, _W, _W],
    [_W, _O, _W, _O, _O, _O, _O, _W],
    [_W, _O, _W, _W, _W, _W, _O, _W],
    [_W, _O, _O, _O, _O, _W, _O, _W],
    [_W, _W, _W, _W, _O, _W, _O, _W],
    [_W, _W, _W, _W, _W, _W, _W, _W],
]

_MAZE_2 = [
    [_W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W],
    [_W, _O, _O, _O, _W, _O, _O, _O, _O, _O, _O, _W],
    [_W, _O, _W, _O, _W, _O, _W, _W, _W, _W, _O, _W],
    [_W, _O, _W, _O, _I, _O, _W, _W, _O, _W, _O, _W],
    [_W, _O, _W, _O, _I, _O, _W, _W, _O, _W, _O, _W],
    [_W, _O, _W, _O, _I, _O, _W, _W, _O, _W, _O, _W],
    [_W, _O, _W, _O, _I, _O, _W, _W, _O, _W, _O, _W],
    [_W, _O, _W, _O, _O, _O, _W, _W, _O, _W, _O, _W],
    [_W, _O, _W, _W, _W, _W, _W, _W, _O, _W, _O, _W],
    [_W, _O, _W, _W, _W, _W, 4, 4, _O, _W, _O, _W],
    [_W, _O, _O, _O, _O, _O, _O, _O, _O, _W, _O, _W],
    [_W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W],
]

_MAZE_3 = [
    [_W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W],
    [_W, _O, _O, _O, _O, _I, _O, _O, _O, _O, _O, _W],
    [_W, _W, _W, _W, _O, _W, _O, _W, _W, _W, _O, _W],
    [_W, _W, _W, _W, _O, _W, _O, _W, _W, _W, _O, _W],
    [_W, _W, _W, _W, _O, _W, _O, _W, _W, _W, _O, _W],
    [_W, _W, _W, _W, _O, _W, _O, _W, _W, _W, _O, _W],
    [_W, _W, _W, _W, _O, _I, _O, _W, _W, _W, _O, _W],
    [_W, _W, _W, _W, _O, _I, _O, _W, _W, _W, _O, _W],
    [_W, _W, _W, _W, _O, _I, _O, _W, _W, _W, _O, _W],
    [_W, _W, _W, _W, _O, _O, _O, _W, _O, _O, _O, _W],
    [_W, _W, _W, _W, _O, _I, _I, _W, _W, _W, _O, _W],
    [_W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W],
]

_MAZE_4 = [
    [_W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W],
    [_W, _O, _W, _O, _O, _O, _O, _O, _O, _W, _O, _W],
    [_W, _O, _W, _W, _W, _W, _W, _O, _W, _W, _O, _W],
    [_W, _O, _W, _W, _W, _W, _W, _O, _W, _W, _O, _W],
    [_W, _O, _W, _W, _W, _W, _W, _O, _I, _I, _O, _W],
    [_W, _O, _W, _W, _W, _W, _W, _O, _W, _W, _O, _W],
    [_W, _O, _O, _O, _W, _W, _W, _O, _W, _W, _O, _W],
    [_W, _O, _O, _O, _W, _W, _W, _O, _W, _W, _O, _W],
    [_W, _O, _O, _O, _O, _O, _O, _O, _O, _O, _O, _W],
    [_W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W],
    [_W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W],
    [_W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W],
]

_MAZE_5 = [
    [_W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W],
    [_W, _O, _W, _W, _W, _W, _W, _W, _W, _W, _O, _W],
    [_W, _O, _W, _W, _W, _W, _W, _W, _W, _W, _O, _W],
    [_W, _O, _W, _W, _W, _W, _W, _W, _W, _W, _O, _W],
    [_W, _O, _W, _W, _W, _W, _W, _W, _W, _W, _O, _W],
    [_W, _O, _W, _W, _W, _W, _W, _W, _W, _W, _O, _W],
    [_W, _O, _O, _O, _O, _O, _O, _O, _O, _O, _O, _W],
    [_W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _O, _W],
    [_W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _O, _W],
    [_W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _O, _W],
    [_W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _O, _W],
    [_W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W, _W],
]

# Per-level: (camera w/h, [(kind, x, y), ...], move_maze, max_dx, max_dy). Slot
# order matches the ARCEngine source's sprite list per level.
_LEVELS = [
    (8, 8, [(KIND_EXIT, 6, 6), (KIND_MAZE1, 0, 0), (KIND_PLAYER, 1, 1)], False, 0, 0),
    (
        12,
        12,
        [(KIND_EXIT, 10, 10), (KIND_MAZE2, 0, 0), (KIND_PLAYER, 1, 1)],
        False,
        0,
        0,
    ),
    (
        12,
        12,
        [
            (KIND_BLOCK, 4, 2),
            (KIND_BLOCK, 10, 2),
            (KIND_EXIT, 8, 9),
            (KIND_MAZE3, 0, 0),
            (KIND_PLAYER, 1, 1),
        ],
        False,
        0,
        0,
    ),
    (
        12,
        12,
        [
            (KIND_BLOCK, 3, 8),
            (KIND_BLOCK, 10, 8),
            (KIND_EXIT, 10, 1),
            (KIND_MAZE4, 0, 0),
            (KIND_PLAYER, 1, 1),
        ],
        False,
        0,
        0,
    ),
    # Level 5: moving maze. block_orange (fixed) + block_orange_flex (floating).
    (
        12,
        12,
        [
            (KIND_BLOCK, 10, 6),
            (KIND_BLOCK_FLEX, 5, 8),
            (KIND_EXIT, 10, 1),
            (KIND_MAZE5, 0, 0),
            (KIND_PLAYER, 1, 1),
        ],
        True,
        0,
        2,
    ),
]

MAX_SPRITES = 5  # up to 2 blocks + exit + maze + player per level
MAX_ENERGY = 63  # 32 top-row pills + 31 right-column pills


# ---------------------------------------------------------------------------
# Transition logic
# ---------------------------------------------------------------------------


def _handle_collided_sprites(
    params: ArcAgi3Params,
    state: ArcAgi3State,
    collisions: Array,
    dx: Int[Array, ""],
    dy: Int[Array, ""],
) -> ArcAgi3State:
    """Port of the per-collided-sprite loop in ``_try_pushing_move``.

    For each sprite the player's first move collided with (fixed-index scan over
    the small, static sprite array), in order:

    - **Pushable block**: push it one cell. If it collides with a same-kind block,
      both annihilate; otherwise the player follows into the vacated cell.
    - **Maze on a move-maze level**: if the maze's absolute displacement stays
      within ``max_delta``, translate the maze and every "fixed"-tagged sprite by
      (dx, dy). Compiled out entirely when ``params.has_moving_maze`` is False.
    """
    pushable = params.sprite_pushable[state.sprite_kind]  # (max_sprites,) bool
    is_maze = params.sprite_is_maze[state.sprite_kind]

    def body(state: ArcAgi3State, i: Int[Array, ""]) -> tuple[ArcAgi3State, None]:
        active_i = state.sprite_active[i]

        def push_block(state: ArcAgi3State) -> ArcAgi3State:
            pushed, block_collisions = try_move_sprite(state, params, i, dx, dy)
            same_kind = (
                block_collisions
                & (pushed.sprite_kind == pushed.sprite_kind[i])
                & pushed.sprite_active
            )
            hit_same = jnp.any(same_kind)

            def annihilate(s: ArcAgi3State) -> ArcAgi3State:
                partner = jnp.argmax(same_kind)
                s = set_sprite_removed(s, i)
                return set_sprite_removed(s, partner)

            def follow(s: ArcAgi3State) -> ArcAgi3State:
                moved, _ = try_move_player(s, params, dx, dy)
                return moved

            return jax.lax.cond(hit_same, annihilate, follow, pushed)

        def maybe_push_block(state: ArcAgi3State) -> ArcAgi3State:
            do_block = collisions[i] & pushable[i] & active_i
            return jax.lax.cond(do_block, push_block, lambda s: s, state)

        state = maybe_push_block(state)

        if params.has_moving_maze:
            state = _maybe_move_maze(params, state, collisions, i, is_maze[i], dx, dy)

        return state, None

    idxs = jnp.arange(state.sprite_x.shape[0])
    state, _ = jax.lax.scan(body, state, idxs)
    return state


def _maybe_move_maze(
    params: ArcAgi3Params,
    state: ArcAgi3State,
    collisions: Array,
    i: Int[Array, ""],
    is_maze_i: Array,
    dx: Int[Array, ""],
    dy: Int[Array, ""],
) -> ArcAgi3State:
    """Move the maze + all fixed sprites if sprite ``i`` is the collided maze.

    Port of the ``move_maze`` branch of ``_try_pushing_move``: only on a
    ``move_maze`` level, only for a maze the player collided with, and only while
    the maze's absolute displacement from the origin stays within ``max_delta``.
    """
    move_maze = params.level_move_maze[state.level_index]
    max_dx = params.level_max_delta_x[state.level_index]
    max_dy = params.level_max_delta_y[state.level_index]

    def do_move(state: ArcAgi3State) -> ArcAgi3State:
        # ARCEngine bounds the *absolute* displacement: abs(maze.x + dx) and
        # abs(maze.y + dy) must stay within max_delta (maze starts at origin).
        within = (jnp.abs(state.sprite_x[i] + dx) <= max_dx) & (
            jnp.abs(state.sprite_y[i] + dy) <= max_dy
        )

        def shift(state: ArcAgi3State) -> ArcAgi3State:
            # Move the maze and every "fixed"-tagged, active sprite together.
            fixed = params.sprite_fixed[state.sprite_kind] & state.sprite_active
            move_mask = fixed.at[i].set(True)  # include the maze itself
            new_x = jnp.where(move_mask, state.sprite_x + dx, state.sprite_x)
            new_y = jnp.where(move_mask, state.sprite_y + dy, state.sprite_y)
            return eqx_replace(state, sprite_x=new_x, sprite_y=new_y)

        return jax.lax.cond(within, shift, lambda s: s, state)

    trigger = move_maze & collisions[i] & is_maze_i & state.sprite_active[i]
    return jax.lax.cond(trigger, do_move, lambda s: s, state)


def _pushing_move(
    params: ArcAgi3Params, state: ArcAgi3State, dx: Int[Array, ""], dy: Int[Array, ""]
) -> ArcAgi3State:
    """Port of ``_try_pushing_move``: player move, goal check, then per-sprite handling."""
    moved, collisions = try_move_player(state, params, dx, dy)

    goal_mask = moved.sprite_active & (moved.sprite_kind == params.goal_kind)
    hit_goal = jnp.any(collisions & goal_mask)

    # next_level() is called on the goal before block handling in the original.
    # ComplexMaze uses next_level() uniformly (score increments even on the last
    # level), unlike SimpleMaze's direct win().
    moved = jax.lax.cond(
        hit_goal, lambda s: next_level_or_win(s, params), lambda s: s, moved
    )

    # Per-sprite handling operates on the pre-advance collisions. If the goal was
    # hit (level changed / game won), skip it — those collided sprites belong to
    # the old level's layout, which has just been replaced.
    return jax.lax.cond(
        hit_goal,
        lambda s: s,
        lambda s: _handle_collided_sprites(params, s, collisions, dx, dy),
        moved,
    )


def _energy_exhausted(params: ArcAgi3Params, state: ArcAgi3State) -> Array:
    """True once the per-level move budget is spent (``action_count > max_energy``).

    ARCEngine disables one pill per non-RESET action and loses when none remain;
    with ``max_energy`` pills that is exactly ``action_count > max_energy``.
    """
    return state.action_count > jnp.asarray(params.max_energy, dtype=jnp.int32)


def transition_complex_maze(
    params: ArcAgi3Params, state: ArcAgi3State, action: Int[Array, ""]
) -> ArcAgi3State:
    """ComplexMaze transition (see module docstring)."""

    def on_reset(s: ArcAgi3State) -> ArcAgi3State:
        return apply_reset(params, s)

    def on_action(s: ArcAgi3State) -> ArcAgi3State:
        def live(s: ArcAgi3State) -> ArcAgi3State:
            # ARCEngine increments action_count in _set_action BEFORE step(); if
            # the move then reaches a goal, set_level resets action_count to 0 (and
            # re-enables all energy pills). So increment first, then move.
            s = eqx_replace(s, action_count=s.action_count + 1)
            dx, dy = action_delta(action)
            s = _pushing_move(params, s, dx, dy)
            # Lose if energy exhausted — but only if the move didn't already end
            # the game (reaching the goal advances/wins before the lose check, and
            # resets action_count so energy is never "exhausted" on a fresh level).
            lose = _energy_exhausted(params, s) & ~s.done
            return jax.lax.cond(lose, lambda x: lose_game(x, params), lambda x: x, s)

        # Terminal games ignore non-RESET actions (empty frame in ARCEngine).
        return jax.lax.cond(s.done, lambda x: x, live, s)

    return jax.lax.cond(action == RESET, on_reset, on_action, state)


# ---------------------------------------------------------------------------
# Params builder
# ---------------------------------------------------------------------------


def _pad_tile(rows: list[list[int]]) -> np.ndarray:
    """Embed a (h, w) pixel grid at the top-left of a (SPRITE_H, SPRITE_W) tile."""
    tile = np.full((SPRITE_H, SPRITE_W), TRANSPARENT, dtype=np.int32)
    arr = np.array(rows, dtype=np.int32)
    h, w = arr.shape
    tile[:h, :w] = arr
    return tile


def _build_sprite_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-kind (pixels, layer, blocking). All sprites use PIXEL_PERFECT (ARCEngine default)."""
    tiles_by_kind = {
        KIND_PLAYER: _pad_tile([[PLAYER_COLOR]]),
        KIND_EXIT: _pad_tile([[EXIT_COLOR]]),
        KIND_BLOCK: _pad_tile([[BLOCK_COLOR]]),
        KIND_BLOCK_FLEX: _pad_tile([[BLOCK_COLOR]]),  # same appearance as block_orange
        KIND_MAZE1: _pad_tile(_MAZE_1),
        KIND_MAZE2: _pad_tile(_MAZE_2),
        KIND_MAZE3: _pad_tile(_MAZE_3),
        KIND_MAZE4: _pad_tile(_MAZE_4),
        KIND_MAZE5: _pad_tile(_MAZE_5),
    }
    pixels = np.stack([tiles_by_kind[k] for k in range(NUM_KINDS)], axis=0)
    # Mazes render below player/exit/blocks (layer -1); everything else layer 0.
    layer = np.array(
        [0 if k not in MAZE_KINDS else -1 for k in range(NUM_KINDS)], dtype=np.int32
    )
    blocking = np.full((NUM_KINDS,), PIXEL_PERFECT, dtype=np.int32)
    return pixels, layer, blocking


def _build_kind_flags() -> dict[str, np.ndarray]:
    """Per-kind push/maze/fixed flags (length NUM_KINDS).

    - pushable: both block kinds (block_orange and block_orange_flex).
    - is_maze : all maze kinds.
    - fixed   : "fixed"-tagged kinds that translate with the maze — player, exit,
      and the fixed block_orange. The floating block_orange_flex is NOT fixed.
    """
    pushable = np.zeros((NUM_KINDS,), dtype=bool)
    is_maze = np.zeros((NUM_KINDS,), dtype=bool)
    fixed = np.zeros((NUM_KINDS,), dtype=bool)
    pushable[KIND_BLOCK] = pushable[KIND_BLOCK_FLEX] = True
    for k in MAZE_KINDS:
        is_maze[k] = True
    fixed[KIND_PLAYER] = fixed[KIND_EXIT] = fixed[KIND_BLOCK] = True
    return {"pushable": pushable, "is_maze": is_maze, "fixed": fixed}


def _build_levels() -> dict[str, np.ndarray]:
    """Per-level initial sprite layout + move-maze config.

    Sprite arrays have shape [max_levels, MAX_SPRITES]; unused slots are
    inactive/invisible/non-collidable and default to kind 0.
    """
    n = len(_LEVELS)
    x = np.zeros((n, MAX_SPRITES), dtype=np.int32)
    y = np.zeros((n, MAX_SPRITES), dtype=np.int32)
    kind = np.zeros((n, MAX_SPRITES), dtype=np.int32)
    active = np.zeros((n, MAX_SPRITES), dtype=bool)
    visible = np.zeros((n, MAX_SPRITES), dtype=bool)
    collidable = np.zeros((n, MAX_SPRITES), dtype=bool)
    move_maze = np.zeros((n,), dtype=bool)
    max_dx = np.zeros((n,), dtype=np.int32)
    max_dy = np.zeros((n,), dtype=np.int32)

    for li, (_cw, _ch, sprites, mm, mdx, mdy) in enumerate(_LEVELS):
        for si, (k, sx, sy) in enumerate(sprites):
            kind[li, si] = k
            x[li, si] = sx
            y[li, si] = sy
            active[li, si] = True
            visible[li, si] = True
            collidable[li, si] = True
        move_maze[li] = mm
        max_dx[li] = mdx
        max_dy[li] = mdy

    return {
        "x": x,
        "y": y,
        "kind": kind,
        "active": active,
        "visible": visible,
        "collidable": collidable,
        "move_maze": move_maze,
        "max_dx": max_dx,
        "max_dy": max_dy,
    }


def _build_pill_positions() -> tuple[np.ndarray, np.ndarray]:
    """Display coordinates of the 63 energy pills, in consumption (list) order.

    32 across the top row (x = i*2, y = 0), then 31 down the right column
    (x = 62, y = i*2 + 2), matching the ARCEngine ``ComplexMaze.__init__`` order.
    """
    xs: list[int] = []
    ys: list[int] = []
    for i in range(32):
        xs.append(i * 2)
        ys.append(0)
    for i in range(31):
        xs.append(62)
        ys.append(i * 2 + 2)
    return np.array(xs, dtype=np.int32), np.array(ys, dtype=np.int32)


def make_params(*, transition_id: int, max_steps: int = 512) -> ArcAgi3Params:
    """Build :class:`ArcAgi3Params` for ComplexMaze (all 5 levels).

    Args:
        transition_id: Index of this game's transition fn in ``games.TRANSITIONS``.
        max_steps: Episode truncation limit (RL convenience; not an engine concept).
    """
    pixels, layer, blocking = _build_sprite_tables()
    flags = _build_kind_flags()
    levels = _build_levels()
    cam_w = np.array([lv[0] for lv in _LEVELS], dtype=np.int32)
    cam_h = np.array([lv[1] for lv in _LEVELS], dtype=np.int32)
    pill_x, pill_y = _build_pill_positions()

    # Available actions: RESET + movement (ACTION1..ACTION4). No ACTION5/6/7.
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
        goal_kind=KIND_EXIT,
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
        game_id="complex_maze",
        transition_id=transition_id,
        # Pushable blocks + moving maze.
        has_moving_maze=True,
        sprite_pushable=jnp.asarray(flags["pushable"]),
        sprite_is_maze=jnp.asarray(flags["is_maze"]),
        sprite_fixed=jnp.asarray(flags["fixed"]),
        level_move_maze=jnp.asarray(levels["move_maze"]),
        level_max_delta_x=jnp.asarray(levels["max_dx"]),
        level_max_delta_y=jnp.asarray(levels["max_dy"]),
        # Energy budget + pill overlay.
        max_energy=MAX_ENERGY,
        num_ui_pills=MAX_ENERGY,
        ui_pill_size=2,
        ui_pill_on_color=PILL_ON_COLOR,
        ui_pill_off_color=PILL_OFF_COLOR,
        ui_pill_x=jnp.asarray(pill_x),
        ui_pill_y=jnp.asarray(pill_y),
    )


__all__ = [
    "KIND_BLOCK",
    "KIND_BLOCK_FLEX",
    "KIND_EXIT",
    "KIND_PLAYER",
    "make_params",
    "transition_complex_maze",
]
