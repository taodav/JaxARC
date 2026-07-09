# ARC-AGI-3 Environments

JaxARC includes an experimental **pure-JAX ARC-AGI-3 runtime subset**: a
`jax.jit` / `jax.vmap`-compatible reimplementation of the
[ARC-AGI-3](https://arcprize.org/arc-agi/3) environment contract and the subset
of the [ARCEngine](https://github.com/arcprize/ARCEngine) mechanics needed for
small-action-space movement games.

Unlike the static-ARC environment (grid editing), ARC-AGI-3 games are
*interactive*: an embodied agent moves through a level, collides with walls, and
reaches a goal to advance. This makes them well-suited to exploration and
reinforcement-learning research.

```{note}
This runtime lives in the standalone `jaxarc.arcagi3` package and is independent
of the static-ARC environment and its dataset registry. It uses a 16-color 64×64
grid and its own state/params types — it does **not** share `jaxarc.envs`,
`jaxarc.State`, or the dataset registry.
```

## Available games

| Game ID | Levels | Actions | Description |
|---|---|---|---|
| `simple_maze` | 2 | `RESET`, `ACTION1`–`ACTION4` | Navigate a maze to the exit; clearing the last level wins. Ported from ARCEngine's public `examples/simple_maze.py` (MIT). |
| `complex_maze` | 5 | `RESET`, `ACTION1`–`ACTION4` | Maze navigation with **pushable blocks** (colliding blocks annihilate per ARCEngine's name-prefix rule — e.g. the floating block into the fixed one, which clears level 5's exit), **invisible walls** (solid but not drawn), a **moving maze** (level 5: pushing a wall translates the whole maze and its fixed sprites), and a per-level **energy budget** rendered as a pill border — exhausting it loses the game. Fully solvable; ported from ARCEngine's public `examples/complex_maze.py` (MIT). |

More movement games can be added as pure transition functions; see
[Adding a game](#adding-a-game).

```{note}
`complex_maze` adds mechanics beyond `simple_maze`: an energy/lose condition
(`GAME_OVER` when the per-level move budget is spent), pushable blocks, and the
`-2` "invisible wall" pixel (solid for collision, transparent to rendering). Its
energy pill border is drawn as a UI overlay on the final frame, matching the
official engine's camera exactly.
```

## Quick start

```python
import jax
from jaxarc.arcagi3 import make_arcagi3

env, params = make_arcagi3("simple_maze")

state, timestep = env.reset(jax.random.PRNGKey(0))
state, timestep = env.step(state, 4)  # ACTION4 = move right
```

`make_arcagi3` returns `(env, params)`, mirroring `jaxarc.make()`. An
`"arcagi3:"` prefix on the game id is accepted and stripped (e.g.
`make_arcagi3("arcagi3:simple_maze")`).

### The action enum

The internal action space is the full ARC-AGI-3 enum (a scalar id `0–7`):

```python
from jaxarc.arcagi3 import (
    RESET,  # 0  initialize / restart the level or game
    ACTION1,  # 1  up
    ACTION2,  # 2  down
    ACTION3,  # 3  left
    ACTION4,  # 4  right
    ACTION5,  # 5  interact / select / rotate   (reserved in v1)
    ACTION6,  # 6  coordinate click, x,y in 0–63 (reserved in v1)
    ACTION7,  # 7  undo                          (reserved in v1)
)
```

`simple_maze` exposes `RESET` + `ACTION1..ACTION4`. Which actions a game accepts
is reported per step in `timestep.extras["available_actions"]`.

### The observation

`timestep.observation` is an `int32[64, 64, 1]` frame with values `0–15`,
matching the ARC-AGI-3 grid contract (a trailing channel dimension is added, as
in the static-ARC env).

```{note}
**Frame simplification (v1).** The official `FrameData.frame` is a *stack* of
64×64 grids (1–N animation sub-frames per action). v1 keeps only the **final**
frame of each action. Frames are camera-scaled: a level's native grid (e.g. 8×8)
is uniformly upscaled to 64×64 with letter-box padding, exactly matching
`arcengine.Camera.render`.
```

### The `TimeStep`

The environment returns a Stoa `(State, TimeStep)`. `TimeStep.extras` mirrors the
official `FrameData` metadata (as JAX-friendly arrays):

| Extra | Type | Meaning |
|---|---|---|
| `available_actions` | `bool[8]` | Which action ids are valid this step |
| `game_state` | `int32[]` | `0` NOT_PLAYED, `1` NOT_FINISHED, `2` WIN, `3` GAME_OVER |
| `level_index` | `int32[]` | Active level |
| `levels_completed` | `int32[]` | Levels cleared so far (the engine's `_score`) |
| `win_levels` | `int32[]` | Levels needed to win (== level count) |
| `action_count` | `int32[]` | Non-`RESET` actions on the current level |

```{note}
`levels_completed` matches ARCEngine exactly: SimpleMaze calls `win()` directly
on the final level (rather than `next_level()`), so `levels_completed` is only
incremented for non-final levels and tops out at `win_levels − 1` at WIN.
```

### Reward and termination

Reward is a clean **sparse completion signal**: `1.0` on the transition that
newly enters `WIN`, `0.0` otherwise. Action efficiency (the official RHAE score)
is intentionally *not* folded into reward — log `action_count` separately if you
need it.

- `StepType.TERMINATED` on `WIN` / `GAME_OVER`
- `StepType.TRUNCATED` at `max_steps` (an RL convenience, configurable via
  `make_arcagi3("simple_maze", max_steps=...)`)
- `StepType.MID` otherwise

## Reinforcement learning

The environment is a genuine Stoa `Environment`, so it composes with the standard
Stoix wrapper chain. For the small-action-space exploration setting, wrap it to a
`Discrete(4)` movement space:

```python
import jax
from stoa import AddRNGKey, RecordEpisodeMetrics, AutoResetWrapper
from jaxarc.arcagi3 import make_arcagi3, DiscreteMovementWrapper

env, params = make_arcagi3("simple_maze", max_steps=128)
env = DiscreteMovementWrapper(env)  # Discrete(4): agent action a -> ACTION(a+1)
env = AddRNGKey(env)
env = RecordEpisodeMetrics(env)
env = AutoResetWrapper(env)

state, ts = env.reset(jax.random.PRNGKey(0))
state, ts = env.step(state, 3)  # agent action 3 -> ACTION4 (right)
```

`DiscreteMovementWrapper` maps agent action `a ∈ {0,1,2,3}` to internal id
`a + ACTION1` (i.e. up/down/left/right). The base env keeps the full enum, so
later PRs can expose richer action spaces (interact, coordinate clicks) without
breaking this one.

A complete, runnable example lives at
[`examples/arcagi3_stoix_adapter.py`](https://github.com/aadimator/JaxARC/blob/main/examples/arcagi3_stoix_adapter.py),
and a minimal functional-API rollout at
[`examples/arcagi3_random_agent.py`](https://github.com/aadimator/JaxARC/blob/main/examples/arcagi3_random_agent.py).

### Vectorization

`reset` and `step` are `jit`/`vmap`-friendly, so you can run thousands of
independent games in parallel:

```python
from jaxarc.arcagi3.env import reset, step

keys = jax.random.split(jax.random.PRNGKey(0), 4096)
states, timesteps = jax.vmap(lambda k: reset(params, k))(keys)
```

## Functional API

For fine-grained control (and Meta-RL, where params vary per task), use the pure
functions directly instead of the object wrapper:

```python
from jaxarc.arcagi3.env import reset, step

state, ts = reset(params, jax.random.PRNGKey(0))
state, ts = step(params, state, 4)
```

Both are `@eqx.filter_jit`-compiled and take `params` explicitly, so params are
never baked into the state.

## Adding a game

A movement game is defined entirely by its `ArcAgi3Params` — the transition logic
is shared. If your game's mechanic is "move the player, reach the goal, advance",
you reuse the generic `transition_movement` and only supply params:

1. Build sprite/level arrays (see
   [`games/simple_maze.py`](https://github.com/aadimator/JaxARC/blob/main/src/jaxarc/arcagi3/games/simple_maze.py)
   as a template): per-kind pixel tiles, per-level initial sprite layout, camera
   dimensions, and the available-action mask.
2. Register a param builder in `games/__init__.py`'s `GAME_BUILDERS`, pointing at
   the `transition_movement` transition (or a new one if you need custom logic).

Games with genuinely new mechanics add a pure
`transition_<game>(params, state, action) -> state` function to the ordered
`TRANSITIONS` registry; `params.transition_id` selects it via `jax.lax.switch`.

## Parity with the official engine

The port is validated against the real ARCEngine. Tier-1 unit tests and Tier-2
golden-trace tests run in CI with no extra dependencies. Tier-3 parity tests
compare frames and game state against the official `arcengine`, and are skipped
unless you opt in:

```bash
pip install arcengine
RUN_ARCAGI3_PARITY=1 pytest test/arcagi3/test_parity_simple_maze.py
```

`arcengine` is never a runtime dependency of JaxARC — it is used only by the
opt-in parity tests.

## Current scope and non-goals (v1)

**Included:** 64×64 frames (values 0–15), the ARC-AGI-3 action enum,
available-action masks, sparse completion reward, level progression, reset,
sprite rendering with layers and transparency, bounding-box / pixel-perfect
collision, and camera scaling / letterboxing.

**Not yet:** `ACTION5` interactions, `ACTION6` coordinate clicks, `ACTION7`
undo, animation sub-frames, and full 25-game public-suite parity. These are
planned follow-ups and do not affect the movement-game contract above.
```
