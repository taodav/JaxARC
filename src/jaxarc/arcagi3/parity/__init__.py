"""Parity tooling for the ARC-AGI-3 runtime subset.

- ``trace_runner`` — canonical JAX-side replay (frames + scalar state records).
- ``official_adapter`` — optional adapter that drives the real ``arcengine`` and
  mirrors the same schema for direct comparison (test-only; ``arcengine`` is
  never a runtime dependency).
"""

from __future__ import annotations

from .trace_runner import run_trace, state_record

__all__ = ["run_trace", "state_record"]
