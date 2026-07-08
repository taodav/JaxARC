"""Functional environment API for the ARC-AGI-3 runtime subset.

Provides pure ``reset(params, key) -> (state, timestep)`` and
``step(params, state, action) -> (state, timestep)`` functions plus a thin
Stoa-compatible :class:`ArcAgi3Environment` wrapper, mirroring the conventions of
``jaxarc.envs.functional`` / ``jaxarc.envs.environment``.

Contract (v1):
- Observation: ``int32[64, 64, 1]`` with values ``0..15`` (final composited frame
  plus JaxARC's trailing channel dim).
- Reward: sparse ``1.0`` **only** on the transition that newly enters ``WIN``,
  ``0.0`` otherwise (RHAE / action efficiency is scored separately, not as reward).
- ``TimeStep.extras`` carries the ARC-AGI-3 metadata: available actions, game
  state, level index, levels completed, win level count, action count.
- Termination: ``TERMINATED`` on ``WIN``/``GAME_OVER``; ``TRUNCATED`` at
  ``max_steps``; ``MID`` otherwise.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import stoa.environment
from jaxtyping import Array, Int
from stoa.spaces import BoundedArraySpace, DictSpace, DiscreteSpace, Space

from jaxarc.types import StepType, TimeStep

from .constants import GAME_OVER, GRID_SIZE, NOT_FINISHED, NUM_ACTIONS, NUM_COLORS, WIN
from .engine import reset_full, step_engine
from .types import ArcAgi3Params, ArcAgi3State


def _empty_state(params: ArcAgi3Params, key: Array) -> ArcAgi3State:
    """Allocate a zeroed state with the correct shapes; ``reset_full`` fills it."""
    zeros_s = jnp.zeros((params.max_sprites,), dtype=jnp.int32)
    false_s = jnp.zeros((params.max_sprites,), dtype=jnp.bool_)
    return ArcAgi3State(
        step_count=jnp.asarray(0, dtype=jnp.int32),
        action_count=jnp.asarray(0, dtype=jnp.int32),
        level_index=jnp.asarray(0, dtype=jnp.int32),
        levels_completed=jnp.asarray(0, dtype=jnp.int32),
        game_state=jnp.asarray(NOT_FINISHED, dtype=jnp.int32),
        done=jnp.asarray(False),
        sprite_x=zeros_s,
        sprite_y=zeros_s,
        sprite_kind=zeros_s,
        sprite_active=false_s,
        sprite_visible=false_s,
        sprite_collidable=false_s,
        key=key.astype(jnp.uint32),
    )


def _observation(params: ArcAgi3Params, state: ArcAgi3State) -> Int[Array, "64 64 1"]:
    from .rendering import render

    frame = render(params, state)
    return jnp.expand_dims(frame, axis=-1)


def _extras(params: ArcAgi3Params, state: ArcAgi3State) -> dict[str, Any]:
    """ARC-AGI-3 metadata mirror of ``FrameData`` fields (JAX-friendly)."""
    return {
        "available_actions": params.available_actions,  # bool[8]
        "game_state": state.game_state,  # int32[]
        "level_index": state.level_index,  # int32[]
        "levels_completed": state.levels_completed,  # int32[] (== engine _score)
        "win_levels": jnp.asarray(params.max_levels, dtype=jnp.int32),
        "action_count": state.action_count,  # int32[]
    }


@eqx.filter_jit
def reset(params: ArcAgi3Params, key: Array) -> tuple[ArcAgi3State, TimeStep]:
    """Reset to level 0 (full reset), producing the initial ``FIRST`` timestep."""
    state = _empty_state(params, key)
    state = reset_full(state, params)

    timestep = TimeStep(
        step_type=StepType.FIRST,
        reward=jnp.asarray(0.0, dtype=jnp.float32),
        discount=jnp.asarray(1.0, dtype=jnp.float32),
        observation=_observation(params, state),
        extras=_extras(params, state),
    )
    return state, timestep


@eqx.filter_jit
def step(
    params: ArcAgi3Params, state: ArcAgi3State, action: Int[Array, ""]
) -> tuple[ArcAgi3State, TimeStep]:
    """Advance one action; sparse reward on newly entering WIN."""
    action = jnp.asarray(action, dtype=jnp.int32)

    was_win = state.game_state == WIN
    next_state = step_engine(params, state, action)
    next_state = eqx.tree_at(lambda s: s.step_count, next_state, state.step_count + 1)

    newly_won = (next_state.game_state == WIN) & (~was_win)
    reward = jnp.where(newly_won, 1.0, 0.0).astype(jnp.float32)

    is_terminal = (next_state.game_state == WIN) | (next_state.game_state == GAME_OVER)
    is_truncated = next_state.step_count >= jnp.asarray(params.max_steps)

    step_type = jnp.where(
        is_terminal,
        StepType.TERMINATED,
        jnp.where(is_truncated, StepType.TRUNCATED, StepType.MID),
    )
    discount = jnp.where(
        is_terminal | is_truncated,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )

    timestep = TimeStep(
        step_type=step_type,
        reward=reward,
        discount=discount,
        observation=_observation(params, next_state),
        extras=_extras(params, next_state),
    )
    return next_state, timestep


class ArcAgi3Environment(stoa.environment.Environment):
    """Stoa-compatible ARC-AGI-3 environment (delegates to the functional API).

    Implements the Stoa ``Environment`` interface so it composes with Stoa/Stoix
    wrappers (``AddRNGKey``, ``RecordEpisodeMetrics``, ``AutoResetWrapper``,
    ``VmapWrapper``, ...). ``reset``/``step`` accept an optional per-call
    ``env_params`` override for Meta-RL, matching the Stoa signature.

    The native internal action is the full ARC-AGI-3 enum (a scalar id in
    ``0..7``). For RL with a small action space, wrap with
    :class:`jaxarc.arcagi3.wrappers.DiscreteMovementWrapper` to expose
    ``Discrete(4)`` over ACTION1..ACTION4.
    """

    def __init__(self, params: ArcAgi3Params):
        self.params = params

    def reset(
        self, rng_key: Array, env_params: ArcAgi3Params | None = None
    ) -> tuple[ArcAgi3State, TimeStep]:
        return reset(self.params if env_params is None else env_params, rng_key)

    def step(
        self,
        state: ArcAgi3State,
        action: Int[Array, ""],
        env_params: ArcAgi3Params | None = None,
    ) -> tuple[ArcAgi3State, TimeStep]:
        return step(self.params if env_params is None else env_params, state, action)

    def observation_shape(self) -> tuple[int, int, int]:
        return (self.params.height, self.params.width, 1)

    # --- Stoa spaces -------------------------------------------------------

    def observation_space(self, env_params: ArcAgi3Params | None = None) -> Space:
        p = self.params if env_params is None else env_params
        return BoundedArraySpace(
            shape=(p.height, p.width, 1),
            dtype=jnp.int32,
            minimum=0,
            maximum=NUM_COLORS - 1,
            name="observation",
        )

    def action_space(self, env_params: ArcAgi3Params | None = None) -> Space:
        # Full internal ARC-AGI-3 action enum (RESET + ACTION1..ACTION7).
        return DiscreteSpace(NUM_ACTIONS, dtype=jnp.int32, name="action")

    def state_space(self, env_params: ArcAgi3Params | None = None) -> Space:
        p = self.params if env_params is None else env_params
        max_i32 = int(jnp.iinfo(jnp.int32).max)

        def scalar(minimum: int = 0, maximum: int = max_i32) -> BoundedArraySpace:
            return BoundedArraySpace(
                shape=(), dtype=jnp.int32, minimum=minimum, maximum=maximum
            )

        def sprite_vec(dtype, minimum, maximum) -> BoundedArraySpace:
            return BoundedArraySpace(
                shape=(p.max_sprites,), dtype=dtype, minimum=minimum, maximum=maximum
            )

        return DictSpace(
            {
                "step_count": scalar(0, p.max_steps),
                "action_count": scalar(),
                "level_index": scalar(0, max(p.max_levels - 1, 0)),
                "levels_completed": scalar(0, p.max_levels),
                "game_state": scalar(0, 3),
                "done": BoundedArraySpace(
                    shape=(), dtype=jnp.bool_, minimum=False, maximum=True
                ),
                "sprite_x": sprite_vec(jnp.int32, -GRID_SIZE, GRID_SIZE),
                "sprite_y": sprite_vec(jnp.int32, -GRID_SIZE, GRID_SIZE),
                "sprite_kind": sprite_vec(jnp.int32, 0, max_i32),
                "sprite_active": sprite_vec(jnp.bool_, False, True),
                "sprite_visible": sprite_vec(jnp.bool_, False, True),
                "sprite_collidable": sprite_vec(jnp.bool_, False, True),
                "key": BoundedArraySpace(
                    shape=(2,),
                    dtype=jnp.uint32,
                    minimum=0,
                    maximum=int(jnp.iinfo(jnp.uint32).max),
                ),
            }
        )

    def reward_space(
        self, env_params: ArcAgi3Params | None = None
    ) -> BoundedArraySpace:
        # Sparse completion reward in {0, 1}.
        return BoundedArraySpace(shape=(), dtype=jnp.float32, minimum=0.0, maximum=1.0)

    @property
    def unwrapped(self) -> ArcAgi3Environment:
        return self

    def render(
        self, state: ArcAgi3State, env_params: ArcAgi3Params | None = None
    ) -> Any:
        from .rendering import render as render_frame

        p = self.params if env_params is None else env_params
        return render_frame(p, state)


__all__ = ["ArcAgi3Environment", "reset", "step"]
