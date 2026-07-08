"""Stoa/Stoix compatibility tests for the ARC-AGI-3 runtime subset.

Verifies the environment is a genuine Stoa ``Environment`` with correct spaces,
the movement wrapper exposes ``Discrete(4)`` with the right action remapping, and
the environment composes with the Stoix core wrapper chain under jit/vmap.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import stoa.environment
from stoa import AddRNGKey, AutoResetWrapper, RecordEpisodeMetrics
from stoa.spaces import BoundedArraySpace, DiscreteSpace

from jaxarc.arcagi3 import (
    ACTION1,
    ACTION4,
    NUM_ACTIONS,
    NUM_COLORS,
    DiscreteMovementWrapper,
    make_arcagi3,
)

KEY = jax.random.PRNGKey(0)
PLAYER_SLOT = 1


def test_env_is_stoa_environment(maze_env):
    env, _params = maze_env
    assert isinstance(env, stoa.environment.Environment)


def test_spaces(maze_env):
    env, params = maze_env

    obs_space = env.observation_space()
    assert isinstance(obs_space, BoundedArraySpace)
    assert obs_space.shape == (64, 64, 1)
    assert int(obs_space.maximum) == NUM_COLORS - 1

    act_space = env.action_space()
    assert isinstance(act_space, DiscreteSpace)
    assert act_space.num_values == NUM_ACTIONS

    # state_space and environment_space build without error.
    env.state_space()
    env.environment_space()


def test_reward_space_is_unit_interval(maze_env):
    env, _params = maze_env
    rs = env.reward_space()
    assert float(rs.minimum) == 0.0
    assert float(rs.maximum) == 1.0


def test_movement_wrapper_discrete4(maze_env):
    env, _params = maze_env
    wenv = DiscreteMovementWrapper(env)
    space = wenv.action_space()
    assert isinstance(space, DiscreteSpace)
    assert space.num_values == 4


def test_movement_wrapper_remaps_actions(maze_env):
    # Agent action a maps to internal id a + ACTION1.
    env, _params = maze_env
    wenv = DiscreteMovementWrapper(env)
    state, _ = wenv.reset(KEY)

    # Agent action 3 -> ACTION4 (right): (1,1) -> (2,1).
    s_right, _ = wenv.step(state, jnp.asarray(3))
    assert (int(s_right.sprite_x[PLAYER_SLOT]), int(s_right.sprite_y[PLAYER_SLOT])) == (
        2,
        1,
    )

    # Agent action 0 -> ACTION1 (up into wall): stays at (1,1).
    s_up, _ = wenv.step(state, jnp.asarray(0))
    assert (int(s_up.sprite_x[PLAYER_SLOT]), int(s_up.sprite_y[PLAYER_SLOT])) == (1, 1)

    # Sanity: ACTION1/ACTION4 constants are the remap endpoints.
    assert ACTION1 == 1
    assert ACTION4 == 4


def _stoix_chain(game_id="simple_maze", max_steps=64):
    env, params = make_arcagi3(game_id, max_steps=max_steps)
    env = DiscreteMovementWrapper(env)
    env = AddRNGKey(env)
    env = RecordEpisodeMetrics(env)
    env = AutoResetWrapper(env)
    return env, params


def test_stoix_chain_reset_and_step():
    env, _params = _stoix_chain()
    state, ts = env.reset(KEY)
    assert "episode_metrics" in ts.extras
    state, ts = env.step(state, jnp.asarray(3))
    assert ts.reward.shape == ()


def test_stoix_chain_jit_and_vmap():
    env, _params = _stoix_chain()

    def rollout(key):
        reset_key, scan_key = jax.random.split(key)
        state, _ = env.reset(reset_key)

        def body(carry, k):
            action = jax.random.randint(k, (), 0, 4)
            next_state, ts = env.step(carry, action)
            return next_state, ts.reward

        return jax.lax.scan(body, state, jax.random.split(scan_key, 16))

    # jit a single rollout
    _, rewards = jax.jit(rollout)(KEY)
    assert rewards.shape == (16,)

    # vmap a batch of rollouts
    keys = jax.random.split(KEY, 8)
    _, batch_rewards = jax.jit(jax.vmap(rollout))(keys)
    assert batch_rewards.shape == (8, 16)
