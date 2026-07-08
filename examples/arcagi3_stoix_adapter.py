"""Stoix-compatible setup for the ARC-AGI-3 SimpleMaze environment.

Shows the first-class RL integration path for the pure-JAX ARC-AGI-3 runtime:
the environment is a real Stoa ``Environment``, so it drops into the standard
Stoix wrapper chain exactly like the static-ARC env in
``jaxarc/stoix_adapter.py``.

This example:
- builds the env via ``make_arcagi3`` and wraps it to a ``Discrete(4)`` movement
  action space (the small-action-space setting these games target),
- applies the Stoix core wrapper chain (``AddRNGKey`` -> ``RecordEpisodeMetrics``
  -> ``AutoResetWrapper``) and vmaps a batch of environments,
- runs a short random-policy rollout and reports episode metrics.

Run::

    python examples/arcagi3_stoix_adapter.py

Note: this is a self-contained illustration, not a training script. In a real
Stoix config you would instead point ``make_*_env`` at ``make_arcagi3`` and let
Stoix apply its own wrapper chain / vmap in the standard order:
``AddRNGKey -> RecordEpisodeMetrics -> AutoResetWrapper -> VmapWrapper``.
"""

from __future__ import annotations

import jax
from stoa import (
    AddRNGKey,
    AutoResetWrapper,
    RecordEpisodeMetrics,
    get_final_step_metrics,
)

from jaxarc.arcagi3 import DiscreteMovementWrapper, make_arcagi3
from jaxarc.arcagi3.wrappers import NUM_MOVEMENT_ACTIONS


def make_stoix_arcagi3(game_id: str = "simple_maze", *, max_steps: int = 128):
    """Build a Stoix-ready ARC-AGI-3 environment.

    Returns ``(env, params)`` where ``env`` is the base environment wrapped with
    the Stoix core chain over a ``Discrete(4)`` movement action space. Mirrors the
    ordering used by ``jaxarc.stoix_adapter.make_jaxarc_env`` /
    Stoix's ``make_*_env``.
    """
    env, params = make_arcagi3(game_id, max_steps=max_steps)
    env = DiscreteMovementWrapper(env)  # Discrete(4) over ACTION1..ACTION4
    env = AddRNGKey(env)  # threads a PRNGKey through state
    env = RecordEpisodeMetrics(env)  # adds episode_return / episode_length
    env = AutoResetWrapper(env)  # resets on episode boundaries
    return env, params


def make_rollout_fn(env, num_steps: int):
    """Return a ``rollout(key)`` closure over the wrapped env (env is not traced).

    Stoix wrappers are plain Python objects, not JAX pytrees, so the environment
    is captured by closure rather than passed as a jit argument.
    """

    def rollout(key):
        reset_key, scan_key = jax.random.split(key)
        state, _ = env.reset(reset_key)

        def body(carry, step_key):
            state = carry
            action = jax.random.randint(step_key, (), 0, NUM_MOVEMENT_ACTIONS)
            next_state, ts = env.step(state, action)
            return next_state, ts

        step_keys = jax.random.split(scan_key, num_steps)
        final_state, traj = jax.lax.scan(body, state, step_keys)
        return final_state, traj

    return rollout


def main() -> None:
    env, _params = make_stoix_arcagi3("simple_maze", max_steps=128)
    rollout = make_rollout_fn(env, num_steps=256)

    # Single rollout.
    _, traj = jax.jit(rollout)(jax.random.PRNGKey(0))
    print("Single random rollout (256 steps):")
    print(f"  total reward : {float(traj.reward.sum())}")

    # Batched rollout over 128 parallel environments — the JAX-native throughput.
    keys = jax.random.split(jax.random.PRNGKey(1), 128)
    _, batch_traj = jax.jit(jax.vmap(rollout))(keys)

    # RecordEpisodeMetrics tags each step; episode_return/length are only valid on
    # terminal steps. `get_final_step_metrics` extracts those (the Stoix logging
    # pattern) and reports whether any episode actually finished.
    metrics = batch_traj.extras["episode_metrics"]
    final_metrics, had_final = get_final_step_metrics(metrics)
    print("\nBatched rollout (128 envs x 256 steps):")
    print(f"  completed episodes : {had_final}")
    if had_final:
        print(
            f"  mean episode_return : {float(final_metrics['episode_return'].mean()):.4f}"
        )
        print(
            f"  mean episode_length : {float(final_metrics['episode_length'].mean()):.2f}"
        )
    print(f"  total reward (all)  : {float(batch_traj.reward.sum())}")


if __name__ == "__main__":
    main()
