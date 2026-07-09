"""Type definitions for the ARC-AGI-3 runtime subset.

The design intentionally avoids porting ARCEngine's flexible Python object model
(named sprites, dynamic lists, dicts). Instead it uses a compact, fixed-size
**entity-component** representation in static JAX arrays so that ``reset`` and
``step`` are ``jax.jit`` / ``jax.vmap`` compatible.

Two Equinox modules:
- ``ArcAgi3Params`` — static, per-game configuration and level data. Fields that
  change array shapes or control Python-level branching are marked static.
- ``ArcAgi3State`` — purely dynamic per-episode state (sprite positions, level
  index, progress counters, RNG key).

All arrays are plain ``jax.Array``; we deliberately do **not** use
``jaxarc.types.Grid`` (which validates cells into ``[-1, 9]``) — ARC-AGI-3 needs
16 colors on a 64x64 grid.
"""

from __future__ import annotations

import chex
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, UInt


def _empty_int_vec() -> Array:
    """Default for optional int arrays (pill coords, per-level deltas): empty int32."""
    return jnp.zeros((0,), dtype=jnp.int32)


def _empty_bool_vec() -> Array:
    """Default for optional bool arrays (per-kind / per-level flags): empty bool."""
    return jnp.zeros((0,), dtype=jnp.bool_)


def _empty_bool_mat() -> Array:
    """Default for the optional per-kind-pair annihilation matrix: empty (0, 0)."""
    return jnp.zeros((0, 0), dtype=jnp.bool_)


class ArcAgi3State(eqx.Module):
    """Dynamic per-episode state for an ARC-AGI-3 movement game.

    ``max_sprites`` is fixed per game (a static array dimension). Inactive sprite
    slots are marked ``sprite_active == False`` and are ignored by rendering and
    collision.
    """

    # --- Episode / game progress ------------------------------------------
    step_count: Int[Array, ""]  # environment steps since reset (incl. no-ops)
    action_count: Int[Array, ""]  # non-RESET actions on the current level
    level_index: Int[Array, ""]  # index of the active level
    levels_completed: Int[Array, ""]  # cumulative levels cleared (== engine _score)
    game_state: Int[
        Array, ""
    ]  # one of constants.{NOT_PLAYED,NOT_FINISHED,WIN,GAME_OVER}
    done: Bool[Array, ""]  # True once WIN or GAME_OVER reached

    # --- Sprite / entity state (component arrays, length == max_sprites) ---
    sprite_x: Int[Array, " max_sprites"]  # top-left x (col), origin top-left
    sprite_y: Int[Array, " max_sprites"]  # top-left y (row)
    sprite_kind: Int[Array, " max_sprites"]  # index into Params.sprite_* tables
    sprite_active: Bool[Array, " max_sprites"]  # slot in use this level
    sprite_visible: Bool[Array, " max_sprites"]  # rendered if active & visible
    sprite_collidable: Bool[Array, " max_sprites"]  # participates in collisions

    # --- RNG ---------------------------------------------------------------
    key: UInt[Array, 2]

    def __check_init__(self) -> None:
        # Guard against placeholders during tracing; keep validation cheap.
        if not hasattr(self.sprite_x, "shape"):
            return
        try:
            chex.assert_rank(self.sprite_x, 1)
            chex.assert_equal_shape(
                [
                    self.sprite_x,
                    self.sprite_y,
                    self.sprite_kind,
                    self.sprite_active,
                    self.sprite_visible,
                    self.sprite_collidable,
                ]
            )
            chex.assert_shape(self.step_count, ())
            chex.assert_shape(self.game_state, ())
        except (AttributeError, TypeError):
            pass


class ArcAgi3Params(eqx.Module):
    """Static per-game configuration and level data.

    Level data is stored as fixed-size stacked arrays indexed by ``level_index``.
    Sprite appearance is stored in per-kind tables (padded to a common
    ``sprite_h x sprite_w``) so rendering/collision can gather by ``sprite_kind``.

    Static fields (``eqx.field(static=True)``) are hashable Python values used for
    shapes and control flow; they are baked into the JIT trace.
    """

    # --- Grid / camera -----------------------------------------------------
    height: int = eqx.field(static=True)  # camera output height (64)
    width: int = eqx.field(static=True)  # camera output width (64)

    # --- Episode limits / capacities (static shape drivers) ----------------
    max_steps: int = eqx.field(static=True)
    max_sprites: int = eqx.field(static=True)
    max_levels: int = eqx.field(static=True)
    sprite_h: int = eqx.field(static=True)  # padded sprite tile height
    sprite_w: int = eqx.field(static=True)  # padded sprite tile width
    player_kind: int = eqx.field(static=True)  # sprite_kind treated as the player
    goal_kind: int = eqx.field(static=True)  # sprite_kind treated as the goal
    background: int = eqx.field(static=True)  # camera background color
    letter_box: int = eqx.field(static=True)  # camera padding color when upscaled
    reset_mode: str = eqx.field(static=True)  # "level" or "full"

    # --- Per-level camera dimensions (ARCEngine resizes the camera to each ---
    # level's grid_size = (width, height); frames are then upscaled to 64x64).
    level_cam_w: Int[Array, " max_levels"]
    level_cam_h: Int[Array, " max_levels"]

    # --- Per-level initial sprite layout (shape [max_levels, max_sprites]) --
    init_sprite_x: Int[Array, "max_levels max_sprites"]
    init_sprite_y: Int[Array, "max_levels max_sprites"]
    init_sprite_kind: Int[Array, "max_levels max_sprites"]
    init_sprite_active: Bool[Array, "max_levels max_sprites"]
    init_sprite_visible: Bool[Array, "max_levels max_sprites"]
    init_sprite_collidable: Bool[Array, "max_levels max_sprites"]

    # --- Per-kind sprite appearance / physics ------------------------------
    # pixels padded to (sprite_h, sprite_w); TRANSPARENT (-1) marks padding/holes.
    sprite_pixels: Int[Array, "num_kinds sprite_h sprite_w"]
    sprite_layer: Int[Array, " num_kinds"]  # z-order (lower drawn first)
    sprite_blocking: Int[
        Array, " num_kinds"
    ]  # constants.{NOT_BLOCKED,BOUNDING_BOX,PIXEL_PERFECT}

    # --- Action availability (bool mask over the 8 action ids) -------------
    available_actions: Bool[Array, " 8"]

    # --- Identification ----------------------------------------------------
    game_id: str = eqx.field(static=True)
    transition_id: int = eqx.field(static=True)  # selects the per-game step fn

    # --- Pushable blocks / moving maze (optional; per-kind & per-level) ----
    # These drive complex_maze's push/annihilate and moving-maze mechanics.
    # Only complex_maze's transition reads them; movement-only games
    # (simple_maze) leave them at their empty defaults. All per-kind arrays are
    # length ``num_kinds``; per-level arrays are length ``max_levels``.
    #
    # - ``sprite_pushable`` : kinds that can be pushed.
    # - ``annihilates`` : ``annihilates[pushed_kind, other_kind]`` is True when
    #   pushing ``pushed_kind`` into ``other_kind`` destroys both (ARCEngine's
    #   asymmetric ``pushed.name.startswith(other.name)`` rule — e.g. the floating
    #   ``block_orange_flex`` annihilates the fixed ``block_orange``). Shape
    #   ``[num_kinds, num_kinds]``.
    # - ``sprite_is_maze``  : kinds that are maze walls (trigger move-maze).
    # - ``sprite_fixed``    : kinds tagged "fixed" — they translate together with
    #   the maze when it moves (player, exit, fixed block; NOT floating block).
    # - ``has_moving_maze`` : static gate; when False the move-maze branch is
    #   compiled out entirely.
    # - ``level_move_maze`` : per-level flag enabling the move-maze mechanic.
    # - ``level_max_delta_x/y`` : per-level bound on the maze's absolute
    #   displacement from the origin.
    has_moving_maze: bool = eqx.field(static=True, default=False)
    sprite_pushable: Bool[Array, " num_kinds"] = eqx.field(
        default_factory=_empty_bool_vec
    )
    annihilates: Bool[Array, "num_kinds num_kinds"] = eqx.field(
        default_factory=_empty_bool_mat
    )
    sprite_is_maze: Bool[Array, " num_kinds"] = eqx.field(
        default_factory=_empty_bool_vec
    )
    sprite_fixed: Bool[Array, " num_kinds"] = eqx.field(default_factory=_empty_bool_vec)
    level_move_maze: Bool[Array, " max_levels"] = eqx.field(
        default_factory=_empty_bool_vec
    )
    level_max_delta_x: Int[Array, " max_levels"] = eqx.field(
        default_factory=_empty_int_vec
    )
    level_max_delta_y: Int[Array, " max_levels"] = eqx.field(
        default_factory=_empty_int_vec
    )

    # --- Energy / UI overlay (optional; defaults => disabled) --------------
    # An "energy" budget of moves per level: when action_count exceeds
    # ``max_energy`` the game is lost (GAME_OVER). ``max_energy == 0`` disables
    # the mechanic entirely (e.g. simple_maze). The budget is drawn as a border
    # of ``num_ui_pills`` pills on the final 64x64 frame; pill i shows the
    # "off" color once ``i < action_count`` (consumed in ui_pill order).
    max_energy: int = eqx.field(static=True, default=0)
    num_ui_pills: int = eqx.field(static=True, default=0)
    ui_pill_size: int = eqx.field(static=True, default=0)
    ui_pill_on_color: int = eqx.field(static=True, default=0)
    ui_pill_off_color: int = eqx.field(static=True, default=0)
    ui_pill_x: Int[Array, " num_pills"] = eqx.field(default_factory=_empty_int_vec)
    ui_pill_y: Int[Array, " num_pills"] = eqx.field(default_factory=_empty_int_vec)

    def __check_init__(self) -> None:
        assert self.reset_mode in ("level", "full"), self.reset_mode
        assert self.max_levels >= 1
        assert self.max_sprites >= 1


__all__ = ["ArcAgi3Params", "ArcAgi3State"]
