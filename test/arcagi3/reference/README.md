# ARC-AGI-3 parity reference sources

Vendored, unmodified game sources from the official
[ARCEngine](https://github.com/arcprize/ARCEngine) (MIT, © ARC Prize
Foundation), used **only** by the Tier-3 parity tests to compare the pure-JAX
port against the real engine.

- `simple_maze_reference.py` — verbatim copy of `examples/simple_maze.py`.

These files:

- are **not** part of the JaxARC package and are never imported at runtime;
- require the optional `arcengine` package (not a JaxARC dependency);
- are only loaded when `RUN_ARCAGI3_PARITY=1` and `arcengine` is importable —
  otherwise the parity tests skip.

To refresh, re-download from upstream `main` and keep the attribution header:

```bash
curl -s https://raw.githubusercontent.com/arcprize/ARCEngine/main/examples/simple_maze.py \
  -o test/arcagi3/reference/simple_maze_reference.py
# then re-add the vendored-source header comment
```
