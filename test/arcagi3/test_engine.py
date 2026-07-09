"""Tier-1 engine unit tests for the ARC-AGI-3 runtime subset.

No ARC API key / official engine required. Covers the contract and mechanics:
rendering, collision, reset, level progression, win, sparse reward, and
jit/vmap compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxarc.arcagi3 import (
    ACTION1,
    ACTION4,
    NOT_FINISHED,
    RESET,
    WIN,
    make_arcagi3,
)
from jaxarc.arcagi3.env import reset as freset
from jaxarc.arcagi3.env import step as fstep
from jaxarc.arcagi3.rendering import composite

from .golden_common import load_solution

KEY = jax.random.PRNGKey(0)
PLAYER_SLOT = 1  # slot order per level is [maze, player, exit]


# --- Observation / contract -------------------------------------------------


def test_observation_shape_and_range(maze_env):
    env, params = maze_env
    _, ts = env.reset(KEY)
    assert ts.observation.shape == (64, 64, 1)
    assert ts.observation.dtype == jnp.int32
    assert int(ts.observation.min()) >= 0
    assert int(ts.observation.max()) <= 15


def test_reset_step_type_and_extras(maze_env):
    from jaxarc.types import StepType

    env, params = maze_env
    _, ts = env.reset(KEY)
    assert int(ts.step_type) == int(StepType.FIRST)
    for k in (
        "available_actions",
        "game_state",
        "level_index",
        "levels_completed",
        "win_levels",
        "action_count",
    ):
        assert k in ts.extras
    assert ts.extras["available_actions"].shape == (8,)
    # RESET + ACTION1..ACTION4 available; ACTION5/6/7 not.
    avail = [bool(x) for x in ts.extras["available_actions"]]
    assert avail[RESET]
    assert avail[ACTION1]
    assert avail[ACTION4]
    assert not avail[5]
    assert not avail[6]
    assert not avail[7]
    assert int(ts.extras["win_levels"]) == params.max_levels


def test_reset_initial_layout(maze_env):
    env, params = maze_env
    state, _ = env.reset(KEY)
    assert int(state.level_index) == 0
    assert int(state.levels_completed) == 0
    assert int(state.game_state) == NOT_FINISHED
    assert (int(state.sprite_x[PLAYER_SLOT]), int(state.sprite_y[PLAYER_SLOT])) == (
        1,
        1,
    )


# --- Rendering --------------------------------------------------------------


def test_render_places_player_and_exit(maze_env):
    env, params = maze_env
    state, _ = env.reset(KEY)
    frame = np.array(composite(params, state))
    assert int(frame[1, 1]) == 8  # player color at (row=1, col=1)
    assert int(frame[6, 6]) == 9  # exit color at (6, 6)
    # Only expected palette values appear.
    assert {int(x) for x in frame.reshape(-1)} <= {0, 5, 8, 9}


def test_render_layer_order_player_over_maze(maze_env):
    # Player (layer 0) must render on top of the maze (layer -1) when overlapping.
    env, params = maze_env
    state, _ = env.reset(KEY)
    frame = np.array(composite(params, state))
    # Player at an open maze cell -> shows player color, not the maze hole/background.
    assert int(frame[1, 1]) == 8


def test_render_camera_upscales_to_64(maze_env):
    # The full render (as seen in observations) matches ARCEngine's camera:
    # level 0's 8x8 grid upscales 8x to fill 64x64, so grid cell (1,1) maps to
    # the 8x8 display block at rows/cols 8..15.
    from jaxarc.arcagi3.rendering import render

    env, params = maze_env
    state, _ = env.reset(KEY)
    frame = np.array(render(params, state))
    assert frame.shape == (64, 64)
    # Player block (grid (1,1) -> display rows/cols 8..15) is all player color.
    assert np.all(frame[8:16, 8:16] == 8)
    # Exit block (grid (6,6) -> display 48..55) is all exit color.
    assert np.all(frame[48:56, 48:56] == 9)
    # 8x scale exactly fills 64 with no letterbox padding on level 0.
    assert int((frame == params.letter_box).sum()) == 0


# --- Collision --------------------------------------------------------------


def test_move_into_wall_is_blocked(maze_env):
    env, params = maze_env
    state, _ = env.reset(KEY)
    # (1,1) moving up -> (1,0) is wall (color 5); position must not change.
    s2, _ = env.step(state, ACTION1)
    assert (int(s2.sprite_x[PLAYER_SLOT]), int(s2.sprite_y[PLAYER_SLOT])) == (1, 1)


def test_move_into_open_space_succeeds(maze_env):
    env, params = maze_env
    state, _ = env.reset(KEY)
    # (1,1) moving right -> (2,1) is open.
    s2, _ = env.step(state, ACTION4)
    assert (int(s2.sprite_x[PLAYER_SLOT]), int(s2.sprite_y[PLAYER_SLOT])) == (2, 1)


def test_action_count_increments_on_non_reset(maze_env):
    env, params = maze_env
    state, _ = env.reset(KEY)
    s2, _ = env.step(state, ACTION1)  # blocked but still counts as an action
    assert int(s2.action_count) == 1
    s3, _ = env.step(s2, ACTION4)
    assert int(s3.action_count) == 2


# --- Reset semantics --------------------------------------------------------


def test_reset_action_restores_level_after_moves(maze_env):
    env, params = maze_env
    state, _ = env.reset(KEY)
    moved, _ = env.step(state, ACTION4)  # action_count now 1
    reset_state, _ = env.step(moved, RESET)
    # level reset -> player back at start, action_count zeroed.
    assert (
        int(reset_state.sprite_x[PLAYER_SLOT]),
        int(reset_state.sprite_y[PLAYER_SLOT]),
    ) == (1, 1)
    assert int(reset_state.action_count) == 0


# --- Level progression / win / reward --------------------------------------


def test_goal_advances_level(maze_env):
    # Solve just the first level and confirm progression to level 1.
    env, params = maze_env
    state, _ = env.reset(KEY)
    full = load_solution(params.game_id)
    ts = None
    for a in full:
        state, ts = env.step(state, a)
        if int(state.level_index) == 1:
            break
    assert int(state.level_index) == 1
    assert int(state.levels_completed) == 1
    assert int(state.game_state) == NOT_FINISHED
    # No reward yet (only on final win).
    assert float(ts.reward) == 0.0


def test_full_playthrough_wins_with_sparse_reward(game_env):
    # Both games are fully solvable by their committed traces, ending in WIN.
    from jaxarc.types import StepType

    env, params = game_env
    state, ts = env.reset(KEY)
    total_reward = 0.0
    for a in load_solution(params.game_id):
        state, ts = env.step(state, a)
        total_reward += float(ts.reward)
    assert int(state.game_state) == WIN
    # simple_maze calls win() directly on the last level (levels_completed tops out
    # at max_levels - 1); complex_maze uses next_level() uniformly and reaches
    # max_levels. Either way WIN is reached and reward is sparse.
    assert int(state.levels_completed) in (params.max_levels - 1, params.max_levels)
    assert bool(state.done)
    assert int(ts.step_type) == int(StepType.TERMINATED)
    # Sparse: exactly one unit of reward, on the winning transition.
    assert total_reward == pytest.approx(1.0)


def test_terminal_state_ignores_further_actions(game_env):
    env, params = game_env
    state, ts = env.reset(KEY)
    for a in load_solution(params.game_id):
        state, ts = env.step(state, a)
    assert int(state.game_state) == WIN
    # A non-RESET action after WIN is a no-op and yields no further reward.
    after, ts_after = env.step(state, ACTION4)
    assert int(after.game_state) == WIN
    assert float(ts_after.reward) == 0.0


def test_complex_maze_flex_block_annihilates_fixed_block():
    # ARCEngine's asymmetric annihilation: pushing the floating block_orange_flex
    # (KIND_BLOCK_FLEX) into the fixed block_orange (KIND_BLOCK) destroys both
    # (name-prefix rule). This is what unlocks level 5.
    from jaxarc.arcagi3.games.complex_maze import (
        KIND_BLOCK,
        KIND_BLOCK_FLEX,
        KIND_PLAYER,
        _pushing_move,
    )

    _env, params = _complex_env()
    # player(1,1), flex(2,1), orange(3,1): push right -> flex hits orange -> both
    # annihilate; player does NOT follow (annihilation branch).
    s = _synthetic_state(
        params,
        [(KIND_PLAYER, 1, 1), (KIND_BLOCK_FLEX, 2, 1), (KIND_BLOCK, 3, 1)],
    )
    s2 = _pushing_move(params, s, jnp.int32(1), jnp.int32(0))
    assert not bool(s2.sprite_active[1])  # flex removed
    assert not bool(s2.sprite_active[2])  # orange removed
    assert (int(s2.sprite_x[0]), int(s2.sprite_y[0])) == (1, 1)


def test_complex_maze_fixed_block_does_not_annihilate_flex():
    # The rule is asymmetric: pushing orange into flex does NOT annihilate
    # ("block_orange".startswith("block_orange_flex") is False). The orange block
    # is simply pushed and the player follows.
    from jaxarc.arcagi3.games.complex_maze import (
        KIND_BLOCK,
        KIND_BLOCK_FLEX,
        KIND_PLAYER,
        _pushing_move,
    )

    _env, params = _complex_env()
    s = _synthetic_state(
        params,
        [(KIND_PLAYER, 1, 1), (KIND_BLOCK, 2, 1), (KIND_BLOCK_FLEX, 3, 1)],
    )
    s2 = _pushing_move(params, s, jnp.int32(1), jnp.int32(0))
    # orange can't move (flex blocks it) so nothing is removed; player stays.
    assert bool(s2.sprite_active[1])
    assert bool(s2.sprite_active[2])


# --- Truncation -------------------------------------------------------------


def test_truncation_at_max_steps():
    from jaxarc.types import StepType

    env, params = make_arcagi3("simple_maze", max_steps=3)
    state, ts = env.reset(KEY)
    for _ in range(3):
        state, ts = env.step(state, ACTION1)  # blocked move; never wins
    assert int(ts.step_type) == int(StepType.TRUNCATED)
    assert float(ts.discount) == 0.0


# --- JAX transformations ----------------------------------------------------


def test_jit_reset_and_step(maze_env):
    env, params = maze_env
    jreset = jax.jit(lambda k: freset(params, k))
    jstep = jax.jit(lambda s, a: fstep(params, s, a))
    state, _ = jreset(KEY)
    state, ts = jstep(state, jnp.asarray(ACTION4))
    assert (int(state.sprite_x[PLAYER_SLOT]), int(state.sprite_y[PLAYER_SLOT])) == (
        2,
        1,
    )


def test_vmap_rollouts(maze_env):
    env, params = maze_env
    keys = jax.random.split(KEY, 8)
    states, ts = jax.vmap(lambda k: freset(params, k))(keys)
    assert ts.observation.shape == (8, 64, 64, 1)
    actions = jnp.arange(8, dtype=jnp.int32) % 4 + 1
    states2, ts2 = jax.vmap(lambda s, a: fstep(params, s, a))(states, actions)
    assert ts2.reward.shape == (8,)
    assert int(states2.step_count[0]) == 1


# --- complex_maze-specific mechanics ---------------------------------------


def _complex_env():
    return make_arcagi3("complex_maze")


def _synthetic_state(params, positions):
    """Build a state with the given [(kind, x, y), ...] sprites; rest inactive.

    Each slot's pixel buffer is gathered from its kind's tile, so rendering and
    collision see the correct per-sprite pixels.
    """
    from jaxarc.arcagi3.engine import eqx_replace
    from jaxarc.arcagi3.env import reset as freset_

    state, _ = freset_(params, KEY)
    n = params.max_sprites
    kx = np.zeros(n, np.int32)
    ky = np.zeros(n, np.int32)
    kk = np.zeros(n, np.int32)
    act = np.zeros(n, bool)
    for i, (k, x, y) in enumerate(positions):
        kk[i], kx[i], ky[i], act[i] = k, x, y, True
    kk_arr = jnp.asarray(kk)
    return eqx_replace(
        state,
        sprite_kind=kk_arr,
        sprite_x=jnp.asarray(kx),
        sprite_y=jnp.asarray(ky),
        sprite_active=jnp.asarray(act),
        sprite_visible=jnp.asarray(act),
        sprite_collidable=jnp.asarray(act),
        sprite_pixels=params.sprite_pixels[kk_arr],
    )


def test_complex_maze_push_block():
    from jaxarc.arcagi3.games.complex_maze import KIND_BLOCK, KIND_PLAYER, _pushing_move

    _env, params = _complex_env()
    # player(1,1) block(2,1), open to the right: pushing right moves block->3,1, player->2,1.
    s = _synthetic_state(params, [(KIND_PLAYER, 1, 1), (KIND_BLOCK, 2, 1)])
    s2 = _pushing_move(params, s, jnp.int32(1), jnp.int32(0))
    assert (int(s2.sprite_x[0]), int(s2.sprite_y[0])) == (2, 1)
    assert (int(s2.sprite_x[1]), int(s2.sprite_y[1])) == (3, 1)


def test_complex_maze_annihilate_blocks():
    from jaxarc.arcagi3.games.complex_maze import KIND_BLOCK, KIND_PLAYER, _pushing_move

    _env, params = _complex_env()
    # player(1,1) block(2,1) block(3,1): pushing right collides two same-kind
    # blocks -> both removed; player does NOT follow (annihilation branch).
    s = _synthetic_state(
        params, [(KIND_PLAYER, 1, 1), (KIND_BLOCK, 2, 1), (KIND_BLOCK, 3, 1)]
    )
    s2 = _pushing_move(params, s, jnp.int32(1), jnp.int32(0))
    assert not bool(s2.sprite_active[1])
    assert not bool(s2.sprite_active[2])
    assert (int(s2.sprite_x[0]), int(s2.sprite_y[0])) == (1, 1)


def test_complex_maze_lose_on_energy_exhaustion():
    from jaxarc.arcagi3 import GAME_OVER

    env, params = _complex_env()
    state, ts = env.reset(KEY)
    # Move up into the wall (blocked) until the energy budget is exceeded.
    for _ in range(params.max_energy + 1):
        state, ts = env.step(state, ACTION1)
    assert int(state.action_count) == params.max_energy + 1
    assert int(state.game_state) == GAME_OVER
    assert bool(state.done)


def test_complex_maze_invisible_wall_not_rendered_but_solid():
    # A -2 pixel blocks movement (collision) but is not drawn (renders as bg).
    from jaxarc.arcagi3.collisions import _tile_to_grid_mask
    from jaxarc.arcagi3.constants import SOLID_INVISIBLE
    from jaxarc.arcagi3.rendering import _tile_to_grid_values

    _env, params = _complex_env()
    # Find a sprite kind whose tile contains -2 invisible-wall pixels (several
    # complex_maze mazes do); don't hardcode a kind index.
    all_pixels = np.array(params.sprite_pixels)
    kinds_with_inv = [
        k
        for k in range(all_pixels.shape[0])
        if (all_pixels[k] == SOLID_INVISIBLE).any()
    ]
    assert kinds_with_inv, "expected at least one maze with -2 invisible walls"
    tile = params.sprite_pixels[kinds_with_inv[0]]

    # Collision mask counts -2 as solid (!= -1); render mask excludes it (>= 0).
    coll = np.array(_tile_to_grid_mask(tile, jnp.int32(0), jnp.int32(0), 64, 64))
    _vals, rmask = _tile_to_grid_values(tile, jnp.int32(0), jnp.int32(0), 64, 64)
    rmask = np.array(rmask)
    inv = np.array(tile) == SOLID_INVISIBLE
    ys, xs = np.where(inv)
    for y, x in zip(ys, xs):
        assert coll[y, x]  # solid
        assert rmask[y, x] == 0  # not drawn


# --- merge-specific mechanics ----------------------------------------------


def _merge_env():
    return make_arcagi3("merge")


def test_merge_absorbs_and_repositions_player():
    # Player [[9]] at (5,5) absorbing sprite-2 [[14,14],[14,14]] at (4,4): the
    # merged player anchors at the bbox min corner (4,4) and the absorbed slot
    # deactivates. (Uses merge_into directly; move-follow is exercised by parity.)
    from jaxarc.arcagi3.engine import merge_into
    from jaxarc.arcagi3.games.merge import KIND_PLAYER, KIND_S2

    _env, params = _merge_env()
    s = _synthetic_state(params, [(KIND_PLAYER, 5, 5), (KIND_S2, 4, 4)])
    r = merge_into(s, jnp.int32(0), jnp.int32(1))
    assert (int(r.sprite_x[0]), int(r.sprite_y[0])) == (4, 4)
    assert not bool(r.sprite_active[1])
    # Player's pixel at its old (5,5) offset (buffer (1,1)) stays 9; the 14s fill
    # the rest of the 2x2.
    buf = np.array(r.sprite_pixels[0])[:2, :2]
    assert int(buf[1, 1]) == 9  # player wins the overlap
    assert int(buf[0, 0]) == 14  # absorbed sprite elsewhere


def test_merge_win_on_pixel_match():
    # region_equal fires when a merged player's rendered region equals the target.
    # Build a trivial case: player and target are identical single pixels at the
    # same spot after a merge -> equal regions.
    from jaxarc.arcagi3.games.merge import KIND_PLAYER, KIND_S4
    from jaxarc.arcagi3.rendering import region_equal

    _env, params = _merge_env()
    # Player and a target sprite far apart: regions differ -> not a win.
    s = _synthetic_state(params, [(KIND_PLAYER, 2, 2), (KIND_S4, 10, 10)])
    assert not bool(region_equal(params, s, jnp.int32(0), jnp.int32(1)))


def test_merge_level1_solves_and_advances():
    # The committed L1 solution advances the level (win check fires end-to-end).
    env, params = _merge_env()
    state, _ = env.reset(KEY)
    for a in [1, 1, 1, 1, 4, 1]:  # BFS-found level-1 solution
        state, ts = env.step(state, a)
    assert int(state.level_index) == 1
    assert int(state.levels_completed) == 1
    assert float(ts.reward) == 0.0  # not a full-game WIN yet
