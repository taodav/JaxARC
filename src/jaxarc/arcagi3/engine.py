"""Core engine mechanics for the ARC-AGI-3 runtime subset.

Pure JAX reimplementation of the ARCEngine mechanics needed by movement games
(verified against ``arcprize/ARCEngine`` ``main``):

- ``try_move_player`` — port of ``ARCBaseGame.try_move`` / ``try_move_sprite``:
  tentatively move the player, test collisions against all other active &
  collidable sprites, revert on any collision, and report which sprites were hit.
- ``reset_to_level`` / ``reset_full`` — port of ``level_reset`` / ``full_reset``.
- ``advance_level`` — port of ``next_level`` + ``is_last_level`` + ``win``:
  increments ``levels_completed``; if on the last level, sets ``WIN``; otherwise
  loads the next level's initial layout.

Per-game logic is a pure function ``(params, state, action) -> state`` selected by
``params.transition_id`` via :func:`step_engine`. The generic engine helpers here
are shared by all such transition functions.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from .collisions import pair_collides
from .constants import GAME_OVER, NOT_FINISHED, WIN
from .types import ArcAgi3Params, ArcAgi3State


def initial_sprites_for_level(
    params: ArcAgi3Params, level_index: Int[Array, ""]
) -> dict:
    """Gather the initial sprite component arrays for a level as a dict of arrays."""
    kinds = params.init_sprite_kind[level_index]
    # Per-slot initial pixels: use the explicit per-(level, slot) override when the
    # game provides one (e.g. merge's rotated walls), else gather from the kind
    # tile. The `.size` check is static (shapes are known at trace time).
    if params.init_sprite_pixels.size > 0:
        init_pixels = params.init_sprite_pixels[level_index]
    else:
        init_pixels = params.sprite_pixels[kinds]
    return {
        "sprite_x": params.init_sprite_x[level_index],
        "sprite_y": params.init_sprite_y[level_index],
        "sprite_kind": kinds,
        "sprite_active": params.init_sprite_active[level_index],
        "sprite_visible": params.init_sprite_visible[level_index],
        "sprite_collidable": params.init_sprite_collidable[level_index],
        # Each slot's initial pixels; games that reshape sprites at runtime (merge)
        # mutate these afterwards.
        "sprite_pixels": init_pixels,
    }


def load_level(
    state: ArcAgi3State, params: ArcAgi3Params, level_index: Int[Array, ""]
) -> ArcAgi3State:
    """Reset sprite layout to a level's initial state and zero the action count."""
    init = initial_sprites_for_level(params, level_index)
    return eqx_replace(
        state,
        level_index=level_index,
        action_count=jnp.asarray(0, dtype=jnp.int32),
        sprite_x=init["sprite_x"],
        sprite_y=init["sprite_y"],
        sprite_kind=init["sprite_kind"],
        sprite_active=init["sprite_active"],
        sprite_visible=init["sprite_visible"],
        sprite_collidable=init["sprite_collidable"],
        sprite_pixels=init["sprite_pixels"],
    )


def reset_to_level(state: ArcAgi3State, params: ArcAgi3Params) -> ArcAgi3State:
    """Port of ``ARCBaseGame.level_reset``: reload the current level only."""
    state = load_level(state, params, state.level_index)
    return eqx_replace(
        state,
        game_state=jnp.asarray(NOT_FINISHED, dtype=jnp.int32),
        done=jnp.asarray(False),
    )


def reset_full(state: ArcAgi3State, params: ArcAgi3Params) -> ArcAgi3State:
    """Port of ``ARCBaseGame.full_reset``: level 0, zero score, NOT_FINISHED."""
    zero = jnp.asarray(0, dtype=jnp.int32)
    state = load_level(state, params, zero)
    return eqx_replace(
        state,
        levels_completed=zero,
        game_state=jnp.asarray(NOT_FINISHED, dtype=jnp.int32),
        done=jnp.asarray(False),
    )


def player_index(state: ArcAgi3State, params: ArcAgi3Params) -> Int[Array, ""]:
    """Index of the (first) active sprite whose kind is the player kind.

    ARCEngine ``try_move`` moves the first sprite matching the name; we mirror
    that with the first active sprite of ``player_kind``.
    """
    is_player = state.sprite_active & (state.sprite_kind == params.player_kind)
    # argmax returns first True; falls back to 0 if none (guarded by callers).
    return jnp.argmax(is_player).astype(jnp.int32)


def try_move_sprite(
    state: ArcAgi3State,
    params: ArcAgi3Params,
    mover: Int[Array, ""],
    dx: Int[Array, ""],
    dy: Int[Array, ""],
) -> tuple[ArcAgi3State, Bool[Array, " max_sprites"]]:
    """Attempt to move sprite ``mover`` by (dx, dy); revert on collision.

    Generic port of ``ARCBaseGame.try_move_sprite``: tentatively move the sprite,
    test collisions against all other active & collidable sprites, and revert the
    move if any collide. Returns the (possibly updated) state and a per-sprite
    boolean vector marking which sprites ``mover`` collided with (all False if the
    move succeeded), mirroring the returned collided-sprite list.

    A non-collidable mover collides with nothing (ARCEngine requires both sprites
    to be collidable), so its move always commits.
    """
    mx, my = state.sprite_x[mover], state.sprite_y[mover]
    nx, ny = mx + dx, my + dy

    mover_pixels = state.sprite_pixels[mover]
    mover_blocking = params.sprite_blocking[state.sprite_kind[mover]]

    def check_other(idx: Int[Array, ""]) -> Bool[Array, ""]:
        other_kind = state.sprite_kind[idx]
        other_pixels = state.sprite_pixels[idx]
        other_blocking = params.sprite_blocking[other_kind]
        # Exclude self, inactive, and non-collidable others (ARCEngine rules).
        eligible = (
            (idx != mover)
            & state.sprite_active[idx]
            & state.sprite_collidable[idx]
            & state.sprite_collidable[mover]
        )
        hit = pair_collides(
            mover_pixels,
            nx,
            ny,
            mover_blocking,
            other_pixels,
            state.sprite_x[idx],
            state.sprite_y[idx],
            other_blocking,
        )
        return hit & eligible

    idxs = jnp.arange(state.sprite_x.shape[0])
    collisions = jax.vmap(check_other)(idxs)  # (max_sprites,) bool
    blocked = jnp.any(collisions)

    # Commit the move only if unblocked (ARCEngine reverts on collision).
    new_x = state.sprite_x.at[mover].set(jnp.where(blocked, mx, nx))
    new_y = state.sprite_y.at[mover].set(jnp.where(blocked, my, ny))
    return eqx_replace(state, sprite_x=new_x, sprite_y=new_y), collisions


def try_move_player(
    state: ArcAgi3State, params: ArcAgi3Params, dx: Int[Array, ""], dy: Int[Array, ""]
) -> tuple[ArcAgi3State, Bool[Array, " max_sprites"]]:
    """Attempt to move the player by (dx, dy); revert on collision.

    Thin wrapper over :func:`try_move_sprite` on the player sprite (ARCEngine's
    ``try_move("player", ...)`` resolves the name then calls ``try_move_sprite``).
    """
    return try_move_sprite(state, params, player_index(state, params), dx, dy)


def is_last_level(state: ArcAgi3State, params: ArcAgi3Params) -> Bool[Array, ""]:
    """Port of ``ARCBaseGame.is_last_level`` (index == number of levels - 1).

    ``max_levels`` is the true level count for a game (no padding levels).
    """
    return state.level_index == (params.max_levels - 1)


def win_game(state: ArcAgi3State, _params: ArcAgi3Params) -> ArcAgi3State:
    """Set the game to WIN without advancing the score.

    Mirrors SimpleMaze's ``step``, which calls ``ARCBaseGame.win()`` **directly**
    on the last level rather than ``next_level()``. So the engine's ``_score``
    (our ``levels_completed``) is only incremented for non-final levels;
    ``win_levels`` (== ``max_levels``) is the count needed to have "cleared" all.
    """
    return eqx_replace(
        state,
        game_state=jnp.asarray(WIN, dtype=jnp.int32),
        done=jnp.asarray(True),
    )


def go_to_next_level(state: ArcAgi3State, params: ArcAgi3Params) -> ArcAgi3State:
    """Port of ``ARCBaseGame.next_level`` for the non-final case.

    Increments ``levels_completed`` (the engine's ``_score``) and loads the next
    level's initial layout.
    """
    completed = state.levels_completed + 1
    state = load_level(state, params, state.level_index + 1)
    return eqx_replace(state, levels_completed=completed)


def advance_level(state: ArcAgi3State, params: ArcAgi3Params) -> ArcAgi3State:
    """Resolve reaching the goal: win on the last level, else advance.

    Matches SimpleMaze's control flow exactly (``if is_last_level(): win() else:
    next_level()``): the final goal wins **without** bumping ``levels_completed``.
    """
    return jax.lax.cond(
        is_last_level(state, params),
        lambda s: win_game(s, params),
        lambda s: go_to_next_level(s, params),
        state,
    )


def next_level_or_win(state: ArcAgi3State, params: ArcAgi3Params) -> ArcAgi3State:
    """Full port of ``ARCBaseGame.next_level``: always ``_score += 1``, then win/advance.

    Unlike SimpleMaze (which calls ``win()`` directly on the last level),
    ComplexMaze reaches the goal via ``next_level()`` on **every** level, so the
    score is incremented on the final level too and ``levels_completed`` reaches
    ``max_levels`` at WIN. Use this for games that call ``next_level()`` uniformly.
    """
    completed = state.levels_completed + 1

    def do_win(s: ArcAgi3State) -> ArcAgi3State:
        return eqx_replace(
            s,
            levels_completed=completed,
            game_state=jnp.asarray(WIN, dtype=jnp.int32),
            done=jnp.asarray(True),
        )

    def do_next(s: ArcAgi3State) -> ArcAgi3State:
        s = load_level(s, params, s.level_index + 1)
        return eqx_replace(s, levels_completed=completed)

    return jax.lax.cond(is_last_level(state, params), do_win, do_next, state)


def lose_game(state: ArcAgi3State, _params: ArcAgi3Params) -> ArcAgi3State:
    """Port of ``ARCBaseGame.lose``: set GAME_OVER.

    ARCEngine's ``lose()`` comment notes it "will auto win if last level" — but
    that auto-win lives in game logic that calls ``next_level`` before checking
    the lose condition, not in ``lose()`` itself. ``lose()`` simply sets
    GAME_OVER; callers that want last-level leniency handle it explicitly.
    """
    return eqx_replace(
        state,
        game_state=jnp.asarray(GAME_OVER, dtype=jnp.int32),
        done=jnp.asarray(True),
    )


def set_sprite_removed(state: ArcAgi3State, idx: Int[Array, ""]) -> ArcAgi3State:
    """Remove a sprite from play (``InteractionMode.REMOVED``): inactive, hidden, non-collidable."""
    return eqx_replace(
        state,
        sprite_active=state.sprite_active.at[idx].set(False),
        sprite_visible=state.sprite_visible.at[idx].set(False),
        sprite_collidable=state.sprite_collidable.at[idx].set(False),
    )


def move_sprite_unchecked(
    state: ArcAgi3State, idx: Int[Array, ""], dx: Int[Array, ""], dy: Int[Array, ""]
) -> ArcAgi3State:
    """Move a sprite by (dx, dy) without collision checks (port of ``Sprite.move``)."""
    return eqx_replace(
        state,
        sprite_x=state.sprite_x.at[idx].add(dx),
        sprite_y=state.sprite_y.at[idx].add(dy),
    )


def _shift_tile(
    tile: Int[Array, "sh sw"], dy: Int[Array, ""], dx: Int[Array, ""]
) -> Int[Array, "sh sw"]:
    """Shift a tile's content down by ``dy`` and right by ``dx`` (both >= 0).

    Exposed cells are filled with TRANSPARENT; content shifted past the bottom/
    right edge is dropped (non-wrapping). ``out[r, c] = tile[r - dy, c - dx]``.
    """
    from .constants import TRANSPARENT

    sh, sw = tile.shape
    r = jnp.arange(sh)[:, None]
    c = jnp.arange(sw)[None, :]
    src_r = r - dy
    src_c = c - dx
    valid = (src_r >= 0) & (src_r < sh) & (src_c >= 0) & (src_c < sw)
    gathered = tile[jnp.clip(src_r, 0, sh - 1), jnp.clip(src_c, 0, sw - 1)]
    return jnp.where(valid, gathered, TRANSPARENT)


def merge_into(
    state: ArcAgi3State,
    p: Int[Array, ""],
    c: Int[Array, ""],
) -> ArcAgi3State:
    """Merge sprite ``c`` into sprite ``p`` (port of ``Sprite.merge`` + level swap).

    Reproduces ARCEngine exactly: the merged sprite spans the combined bounding
    box of both sprites; ``c``'s non-transparent pixels are painted first, then
    ``p``'s over the top (``p`` wins ties); the result anchors at the bbox min
    corner. Here we write the composited pixels into ``p``'s slot buffer, move
    ``p`` to ``(min_x, min_y)``, and deactivate ``c``.

    Fixed-shape/JIT-safe: because the anchor is the min corner, the merged content
    is top-left-aligned in the ``sprite_h x sprite_w`` buffer (chosen large enough
    to bound any merge — 16x16 for merge), so a non-wrapping shift + composite
    matches ARCEngine's variable-size merged array when rendered at ``(min_x,
    min_y)``.
    """
    from .constants import TRANSPARENT

    px, py = state.sprite_x[p], state.sprite_y[p]
    cx, cy = state.sprite_x[c], state.sprite_y[c]

    min_x = jnp.minimum(px, cx)
    min_y = jnp.minimum(py, cy)

    # Shift each sprite's content so the bbox min corner maps to buffer (0, 0).
    shifted_c = _shift_tile(state.sprite_pixels[c], cy - min_y, cx - min_x)
    shifted_p = _shift_tile(state.sprite_pixels[p], py - min_y, px - min_x)

    # Paint c first, then p's non-transparent pixels over the top (p wins ties).
    merged = jnp.where(shifted_p != TRANSPARENT, shifted_p, shifted_c)

    state = eqx_replace(
        state,
        sprite_x=state.sprite_x.at[p].set(min_x),
        sprite_y=state.sprite_y.at[p].set(min_y),
        sprite_pixels=state.sprite_pixels.at[p].set(merged),
    )
    return set_sprite_removed(state, c)


def eqx_replace(state: ArcAgi3State, **changes) -> ArcAgi3State:
    """Small helper: functional update of an Equinox module via ``tree_at``.

    Centralizes the (field-name -> new-value) update pattern used throughout the
    engine.
    """
    if not changes:
        return state
    names = list(changes.keys())
    values = [changes[n] for n in names]
    return eqx.tree_at(lambda s: [getattr(s, n) for n in names], state, values)


# ---------------------------------------------------------------------------
# Per-game transition dispatch
# ---------------------------------------------------------------------------


def step_engine(
    params: ArcAgi3Params, state: ArcAgi3State, action: Int[Array, ""]
) -> ArcAgi3State:
    """Dispatch to the per-game transition function selected by ``transition_id``.

    ``transition_id`` is a **static** field (baked into the JIT trace), so this is
    a plain Python index into the ordered registry rather than ``jax.lax.switch``.
    That matters: ``lax.switch`` traces *every* branch, which would run each game's
    transition against the wrong game's ``Params`` (e.g. gathering into another
    game's empty optional arrays). Python dispatch traces only the active game.

    Registered transitions are collected lazily from ``arcagi3.games`` to avoid a
    circular import (games import engine helpers).
    """
    from .games import TRANSITIONS

    return TRANSITIONS[params.transition_id](params, state, action)


__all__ = [
    "advance_level",
    "eqx_replace",
    "go_to_next_level",
    "is_last_level",
    "load_level",
    "lose_game",
    "merge_into",
    "move_sprite_unchecked",
    "next_level_or_win",
    "player_index",
    "reset_full",
    "reset_to_level",
    "set_sprite_removed",
    "step_engine",
    "try_move_player",
    "try_move_sprite",
    "win_game",
]
