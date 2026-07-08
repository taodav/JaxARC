"""Action-space wrappers for the ARC-AGI-3 runtime subset.

The base :class:`jaxarc.arcagi3.env.ArcAgi3Environment` uses the full ARC-AGI-3
action enum (a scalar id in ``0..7``). For small-action-space RL (the exploration
setting these games target), :class:`DiscreteMovementWrapper` exposes a compact
``Discrete(4)`` over the four movement actions ACTION1..ACTION4, remapping agent
action ``0..3`` to internal action ids ``1..4``.

This mirrors the static-ARC ``jaxarc.wrappers`` pattern: a thin Stoa
``Wrapper`` that translates the action and overrides ``action_space``, delegating
everything else to the wrapped environment.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Int
from stoa.core_wrappers.wrapper import Wrapper
from stoa.spaces import DiscreteSpace, Space

from jaxarc.types import TimeStep

from .constants import ACTION1
from .types import ArcAgi3Params, ArcAgi3State

# Number of movement actions (ACTION1..ACTION4).
NUM_MOVEMENT_ACTIONS = 4


class DiscreteMovementWrapper(Wrapper):
    """Expose ``Discrete(4)`` over ACTION1..ACTION4.

    Agent action ``a in {0,1,2,3}`` maps to internal id ``a + ACTION1`` (i.e.
    ``1..4`` = up/down/left/right). RESET and ACTION5..7 are not reachable through
    this wrapper — appropriate for the movement-only exploration setting.
    """

    def step(
        self,
        state: ArcAgi3State,
        action: Int[Array, ""],
        env_params: ArcAgi3Params | None = None,
    ) -> tuple[ArcAgi3State, TimeStep]:
        internal = jnp.asarray(action, dtype=jnp.int32) + ACTION1
        return self._env.step(state, internal, env_params)

    def action_space(self, env_params: ArcAgi3Params | None = None) -> Space:
        return DiscreteSpace(NUM_MOVEMENT_ACTIONS, dtype=jnp.int32, name="movement")


__all__ = ["NUM_MOVEMENT_ACTIONS", "DiscreteMovementWrapper"]
