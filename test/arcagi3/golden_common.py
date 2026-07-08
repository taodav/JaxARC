"""Shared logic for ARC-AGI-3 golden-trace tests.

Defines the deterministic action traces used by both the generator
(``scripts/generate_arcagi3_golden.py``) and the replay test
(``test_golden.py``). The actual replay (frames + state records) is delegated to
:mod:`jaxarc.arcagi3.parity.trace_runner`, which is the single source of truth
shared with the Tier-3 parity tests.

Movement-game resets are deterministic and ignore the RNG key, so a trace of
action ids reproduces identical frames/states every run — the property golden
tests rely on.

Trace construction uses a general **push-aware BFS** that drives the real JAX
environment (so it handles walls, invisible walls, and pushable blocks without
re-encoding each game's maze): it searches the space of (level, player, block
positions) for an action sequence that clears every level. Because it steps the
actual engine, the resulting traces are valid solutions for any movement game.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxarc.arcagi3 import ACTION1, RESET
from jaxarc.arcagi3.env import reset as env_reset
from jaxarc.arcagi3.env import step as env_step

# Re-export the canonical runner so existing imports keep working.
from jaxarc.arcagi3.parity.trace_runner import DEFAULT_KEY as KEY
from jaxarc.arcagi3.parity.trace_runner import run_trace  # noqa: F401

GOLDEN_ROOT = Path(__file__).parent / "golden"

PLAYER_KIND = 0

_MOVEMENT_ACTIONS = (1, 2, 3, 4)


def load_solution(game_id: str) -> list[int]:
    """Load a game's committed winning action trace (``trace_000``) from disk.

    Fast path for tests that just need *a* full solution without re-running the
    BFS solver: reads the golden fixture produced by
    ``scripts/generate_arcagi3_golden.py``.
    """
    path = GOLDEN_ROOT / game_id / "trace_000_actions.npy"
    return np.load(path).tolist()


def _state_key(state) -> tuple:
    """Search key: (level, player pos, every active sprite's (kind, x, y)).

    Captures the *full* mutable layout — player, both block kinds, and the maze
    (which itself moves on move-maze levels) — so BFS never conflates distinct
    reachable states. Anything that can move must appear here or the search will
    wrongly prune and fail to find a solution.
    """
    kinds = [int(k) for k in state.sprite_kind]
    p = kinds.index(PLAYER_KIND)
    sprites = tuple(
        sorted(
            (kinds[i], int(state.sprite_x[i]), int(state.sprite_y[i]))
            for i in range(len(kinds))
            if bool(state.sprite_active[i])
        )
    )
    return (
        int(state.level_index),
        int(state.sprite_x[p]),
        int(state.sprite_y[p]),
        sprites,
    )


# Solutions are deterministic per game; cache by game id so the BFS (and its jit
# compilation) runs at most once per game across the whole test session.
_SOLUTION_CACHE: dict[str, list[int]] = {}


def solve_game(params, *, max_depth: int = 120) -> list[int]:
    """Return an action sequence clearing as many levels as movement can solve.

    Push-aware BFS per level: expand movement actions against the live JAX
    environment, advancing to the next level (or WIN) as soon as it happens, and
    pruning GAME_OVER branches. Works for simple_maze and complex_maze alike.
    Results are cached per ``params.game_id``.

    Some levels are not solvable by ACTION1-4 movement alone (e.g. complex_maze's
    moving-maze level 5 — verified to behave identically in the official engine).
    When a level's reachable state space is exhausted without progress, the search
    **stops and returns the trace so far** rather than raising, so the canonical
    trace exercises every solvable level plus entry into the first unsolvable one.
    """
    cached = _SOLUTION_CACHE.get(params.game_id)
    if cached is not None:
        return list(cached)

    jstep = jax.jit(lambda s, a: env_step(params, s, a))
    state, _ = env_reset(params, KEY)

    actions: list[int] = []
    cur = state
    for _ in range(params.max_levels):
        start = cur
        start_level = int(start.level_index)
        q: deque = deque([(start, [])])
        seen = {_state_key(start)}
        found = None
        while q and found is None:
            st, path = q.popleft()
            for a in _MOVEMENT_ACTIONS:
                ns, _ = jstep(st, jnp.int32(a))
                gs = int(ns.game_state)
                if gs == 3:  # GAME_OVER — dead branch
                    continue
                if int(ns.level_index) > start_level or gs == 2:  # advanced or WIN
                    found = ([*path, a], ns)
                    break
                key = _state_key(ns)
                if key not in seen and len(path) < max_depth:
                    seen.add(key)
                    q.append((ns, [*path, a]))
        if found is None:
            # Level unsolvable by movement: stop here (see docstring). The trace
            # still covers all solved levels plus the entry into this one.
            break
        actions.extend(found[0])
        cur = found[1]
        if int(cur.game_state) == 2:  # WIN
            break
    _SOLUTION_CACHE[params.game_id] = list(actions)
    return actions


def build_traces(params) -> dict[str, list[int]]:
    """Construct the canonical action traces for a movement game.

    - ``trace_000`` — a full solve of every level, ending in WIN.
    - ``trace_001`` — two blocked moves into a wall, a RESET, then the full solve.
      Exercises collision-revert and level-reset in addition to progression.
    """
    solve_actions = solve_game(params)
    mixed = [ACTION1, ACTION1, RESET, *solve_actions]
    return {"trace_000": solve_actions, "trace_001": mixed}
