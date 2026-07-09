"""Generate golden-trace fixtures for the ARC-AGI-3 runtime subset.

Records deterministic action traces for SimpleMaze along with the frames and
per-step state they produce, writing them under
``test/arcagi3/golden/<game_id>/``. ``test_golden.py`` replays the same traces
and asserts byte-for-byte equality, locking parity without needing the official
ARCEngine in CI.

Regenerate (only when the engine's *intended* behavior changes, and after
reviewing the diff)::

    python scripts/generate_arcagi3_golden.py

Fixtures per trace:
- ``<trace>_actions.npy``  : int32[T]      action ids
- ``<trace>_frames.npy``   : int32[T+1,64,64] observations (index 0 == reset)
- ``<trace>_states.json``  : list[T+1] of scalar state/timestep records
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure the sibling test package is importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jaxarc.arcagi3 import make_arcagi3  # noqa: E402
from test.arcagi3.golden_common import build_traces, run_trace  # noqa: E402

GOLDEN_ROOT = _REPO_ROOT / "test" / "arcagi3" / "golden"

# Games to snapshot; extend as new movement games are ported.
GAMES = ["simple_maze", "complex_maze", "merge"]


def generate_game(game_id: str) -> None:
    _env, params = make_arcagi3(game_id)
    out_dir = GOLDEN_ROOT / game_id
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = build_traces(params)
    for name, actions in traces.items():
        frames, records = run_trace(params, actions)
        np.save(out_dir / f"{name}_actions.npy", np.asarray(actions, dtype=np.int32))
        # Frames are stored compressed: they are large (int32[T,64,64]) but highly
        # repetitive, so np.savez_compressed shrinks them ~150x (keeping each well
        # under pre-commit's large-file limit). Loaded via np.load(...)["frames"].
        np.savez_compressed(out_dir / f"{name}_frames.npz", frames=frames)
        # Trailing newline keeps the pre-commit end-of-file-fixer happy.
        (out_dir / f"{name}_states.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        print(
            f"[{game_id}] {name}: {len(actions)} actions, "
            f"frames {frames.shape}, final state={records[-1]['game_state']}"
        )


def main() -> None:
    for game_id in GAMES:
        generate_game(game_id)
    print(f"\nWrote golden fixtures under {GOLDEN_ROOT}")


if __name__ == "__main__":
    main()
