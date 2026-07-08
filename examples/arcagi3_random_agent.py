"""Random-agent rollout on the ARC-AGI-3 SimpleMaze environment.

Demonstrates the pure-JAX ARC-AGI-3 runtime subset:
- ``make_arcagi3`` to build an environment + params,
- the Stoa-style ``(state, timestep)`` reset/step interface,
- sampling only from the *available* movement actions,
- a ``jax.lax.scan`` rollout that is fully ``jit``-compatible,
- a ``jax.vmap`` batch of independent rollouts.

Run::

    python examples/arcagi3_random_agent.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxarc.arcagi3 import make_arcagi3
from jaxarc.arcagi3.env import reset as env_reset
from jaxarc.arcagi3.env import step as env_step


def rollout(params, key, num_steps: int):
    """Run a single random rollout of ``num_steps`` and return per-step rewards.

    Actions are sampled uniformly from the movement actions ACTION1..ACTION4
    (the available actions for SimpleMaze, excluding RESET). The environment is
    *not* auto-reset on termination here; rewards after a win are simply 0.
    """
    reset_key, scan_key = jax.random.split(key)
    state, _ = env_reset(params, reset_key)

    # SimpleMaze exposes RESET + ACTION1..ACTION4; sample from movement only.
    movement_actions = jnp.array([1, 2, 3, 4], dtype=jnp.int32)

    def body(carry, step_key):
        state = carry
        action = movement_actions[jax.random.randint(step_key, (), 0, 4)]
        next_state, ts = env_step(params, state, action)
        return next_state, ts.reward

    step_keys = jax.random.split(scan_key, num_steps)
    final_state, rewards = jax.lax.scan(body, state, step_keys)
    return final_state, rewards


def main() -> None:
    _env, params = make_arcagi3("simple_maze")

    # Single JIT-compiled rollout.
    key = jax.random.PRNGKey(0)
    final_state, rewards = jax.jit(rollout, static_argnums=2)(params, key, 200)
    print("Single rollout (200 steps):")
    print(f"  total reward     : {float(rewards.sum())}")
    print(f"  levels_completed : {int(final_state.levels_completed)}")
    print(f"  game_state       : {int(final_state.game_state)} (2 == WIN)")

    # Batch of 512 independent rollouts via vmap — the JAX-native throughput win.
    keys = jax.random.split(jax.random.PRNGKey(1), 512)
    batched = jax.jit(jax.vmap(rollout, in_axes=(None, 0, None)), static_argnums=2)
    final_states, batch_rewards = batched(params, keys, 200)
    solved = (final_states.game_state == 2).sum()
    print("\nBatched rollouts (512 envs x 200 steps):")
    print(f"  solved by random policy : {int(solved)} / 512")
    print(f"  mean total reward       : {float(batch_rewards.sum(axis=1).mean()):.4f}")


if __name__ == "__main__":
    main()
