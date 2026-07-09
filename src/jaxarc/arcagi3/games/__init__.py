"""Game ports and the transition registry for the ARC-AGI-3 runtime subset.

``TRANSITIONS`` is the ordered list of per-game transition functions. A game's
``ArcAgi3Params.transition_id`` indexes into this list, and ``engine.step_engine``
dispatches to it by (static) Python index. Append new transitions at the end to
keep existing ``transition_id`` values stable.

Games whose mechanic is pure move-and-reach-goal (like SimpleMaze) reuse the
generic :func:`transition_movement`; they only differ in their ``Params``.
"""

from __future__ import annotations

from .base import transition_movement
from .complex_maze import make_params as make_complex_maze_params
from .complex_maze import transition_complex_maze
from .merge import make_params as make_merge_params
from .merge import transition_merge
from .simple_maze import make_params as make_simple_maze_params

# Ordered transition registry — index == transition_id. Append new transitions
# at the end to keep existing transition_id values (baked into saved params) stable.
# 0: generic movement (SimpleMaze and other move-to-goal games).
# 1: complex maze (pushable blocks, energy budget, invisible walls).
# 2: merge (runtime sprite merging, pixel-equality win).
TRANSITIONS = [transition_movement, transition_complex_maze, transition_merge]

TRANSITION_MOVEMENT = 0
TRANSITION_COMPLEX_MAZE = 1
TRANSITION_MERGE = 2

# Registry of built-in game param builders, keyed by game id.
GAME_BUILDERS = {
    "simple_maze": lambda **kw: make_simple_maze_params(
        transition_id=TRANSITION_MOVEMENT, **kw
    ),
    "complex_maze": lambda **kw: make_complex_maze_params(
        transition_id=TRANSITION_COMPLEX_MAZE, **kw
    ),
    "merge": lambda **kw: make_merge_params(transition_id=TRANSITION_MERGE, **kw),
}


__all__ = [
    "GAME_BUILDERS",
    "TRANSITIONS",
    "TRANSITION_COMPLEX_MAZE",
    "TRANSITION_MERGE",
    "TRANSITION_MOVEMENT",
    "make_complex_maze_params",
    "make_merge_params",
    "make_simple_maze_params",
    "transition_complex_maze",
    "transition_merge",
    "transition_movement",
]
