"""Collision detection for the ARC-AGI-3 runtime subset.

Faithful JAX port of ``arcengine.Sprite.collides_with`` and
``ARCBaseGame.try_move`` (verified against ``arcprize/ARCEngine`` ``main``).

ARCEngine semantics reproduced here:
- A sprite never collides with itself.
- Both sprites must be collidable, else no collision.
- If either sprite is ``NOT_BLOCKED`` -> no collision.
- Bounding-box overlap is checked first (fast reject). Overlap uses half-open
  intervals: ``a.x < b.x + b.w`` and ``a.x + a.w > b.x`` (and same for y).
- If either sprite is ``PIXEL_PERFECT``, the overlapping region is compared
  pixel-by-pixel; a collision occurs where **both** sprites have a
  non-transparent pixel (value != TRANSPARENT). Otherwise bounding-box overlap
  alone is a collision.

Because sprite tiles are padded to a common ``(sprite_h, sprite_w)`` with
``TRANSPARENT`` (-1), we can vectorize pixel-perfect tests without ragged arrays:
we roll each tile to its world position on the 64x64 grid and AND the
non-transparent masks.

``try_move`` here operates on a single mover sprite (typically the player) vs.
all active/collidable sprites, returning a per-sprite boolean collision vector.
The caller decides whether to commit the move (ARCEngine reverts on any
collision).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from .constants import NOT_BLOCKED, PIXEL_PERFECT, TRANSPARENT


def _tile_to_grid_mask(
    pixels: Int[Array, "sh sw"],
    x: Int[Array, ""],
    y: Int[Array, ""],
    height: int,
    width: int,
) -> Bool[Array, "H W"]:
    """Place a sprite tile's non-transparent mask onto an (H, W) grid at (x, y).

    Out-of-bounds pixels are dropped. ``x`` is the column, ``y`` the row.
    """
    sh, sw = pixels.shape
    nonterp = pixels != TRANSPARENT  # (sh, sw) bool

    # Coordinates of each tile pixel in grid space.
    rows = jnp.arange(sh)[:, None] + y  # (sh, 1)
    cols = jnp.arange(sw)[None, :] + x  # (1, sw)
    rows = jnp.broadcast_to(rows, (sh, sw))
    cols = jnp.broadcast_to(cols, (sh, sw))

    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    place = nonterp & in_bounds

    flat = jnp.zeros((height * width,), dtype=jnp.bool_)
    clamped = jnp.clip(rows, 0, height - 1) * width + jnp.clip(cols, 0, width - 1)
    flat = flat.at[clamped.reshape(-1)].max((place.reshape(-1)).astype(jnp.bool_))
    return flat.reshape(height, width)


def _bbox_dims(pixels: Int[Array, "sh sw"]) -> tuple[Int[Array, ""], Int[Array, ""]]:
    """Effective (width, height) of a padded tile = extent of non-transparent pixels.

    Matches ``Sprite.width``/``height`` which are ``render().shape`` — but since
    tiles are padded with TRANSPARENT, we use the non-transparent bounding extent
    so trailing padding rows/cols don't inflate the bounding box.
    """
    nonterp = pixels != TRANSPARENT
    any_row = jnp.any(nonterp, axis=1)
    any_col = jnp.any(nonterp, axis=0)
    sh, sw = pixels.shape
    # width = last True col + 1 (0 if none); height similarly.
    col_idx = jnp.where(any_col, jnp.arange(sw), -1)
    row_idx = jnp.where(any_row, jnp.arange(sh), -1)
    width = jnp.maximum(jnp.max(col_idx) + 1, 0)
    height = jnp.maximum(jnp.max(row_idx) + 1, 0)
    return width.astype(jnp.int32), height.astype(jnp.int32)


def bbox_overlap(
    ax: Int[Array, ""],
    ay: Int[Array, ""],
    aw: Int[Array, ""],
    ah: Int[Array, ""],
    bx: Int[Array, ""],
    by: Int[Array, ""],
    bw: Int[Array, ""],
    bh: Int[Array, ""],
) -> Bool[Array, ""]:
    """Half-open AABB overlap test, matching ARCEngine's early-out condition."""
    separated = (ax >= bx + bw) | (ax + aw <= bx) | (ay >= by + bh) | (ay + ah <= by)
    return ~separated


def pair_collides(
    mover_pixels: Int[Array, "sh sw"],
    mover_x: Int[Array, ""],
    mover_y: Int[Array, ""],
    mover_blocking: Int[Array, ""],
    other_pixels: Int[Array, "sh sw"],
    other_x: Int[Array, ""],
    other_y: Int[Array, ""],
    other_blocking: Int[Array, ""],
) -> Bool[Array, ""]:
    """Whether two sprites collide, reproducing ``Sprite.collides_with`` logic.

    Assumes the caller has already excluded self-collision and non-collidable /
    inactive sprites. ``NOT_BLOCKED`` on either sprite disables collision.

    Collision is computed in sprite-relative space (bounding-box overlap plus a
    pixel-perfect check via a relative roll), independent of any world grid bounds
    — sprites may sit at negative coordinates and still collide, matching
    ARCEngine. (A moving maze pushes the player to negative coords; clamping to a
    grid there would miss real collisions.)
    """
    mw, mh = _bbox_dims(mover_pixels)
    ow, oh = _bbox_dims(other_pixels)

    overlap = bbox_overlap(mover_x, mover_y, mw, mh, other_x, other_y, ow, oh)
    any_not_blocked = (mover_blocking == NOT_BLOCKED) | (other_blocking == NOT_BLOCKED)

    def pixel_check() -> Bool[Array, ""]:
        # ARCEngine compares the two sprites' non-transparent pixels in their
        # overlapping world region, independent of any grid bounds (sprites may
        # sit at negative coordinates). Both tiles share the padded (sh, sw)
        # shape, so we roll ``other`` into ``mover``'s frame by the relative
        # offset and AND the non-transparent masks. ``jnp.roll`` wrapping is
        # harmless: a wrapped pixel lands >= sh/sw away, outside any real overlap,
        # and the bbox_overlap gate already bounds us to genuinely overlapping
        # placements.
        m_nt = mover_pixels != TRANSPARENT
        o_nt = other_pixels != TRANSPARENT
        off_r = other_y - mover_y
        off_c = other_x - mover_x
        o_shifted = jnp.roll(o_nt, shift=(off_r, off_c), axis=(0, 1))
        # o_shifted[r, c] reads o_nt[r - off_r, c - off_c]; that source index is a
        # real (non-wrapped) overlap only when it lies within [0, sh) x [0, sw).
        sh, sw = mover_pixels.shape
        r = jnp.arange(sh)
        c = jnp.arange(sw)
        valid_r = (r - off_r >= 0) & (r - off_r < sh)
        valid_c = (c - off_c >= 0) & (c - off_c < sw)
        valid = valid_r[:, None] & valid_c[None, :]
        return jnp.any(m_nt & o_shifted & valid)

    is_pixel_perfect = (mover_blocking == PIXEL_PERFECT) | (
        other_blocking == PIXEL_PERFECT
    )

    # If bbox doesn't overlap -> no collision. If it does: pixel-perfect pair uses
    # pixel check; otherwise bbox overlap is itself the collision.
    collide_when_overlap = jax.lax.cond(
        is_pixel_perfect, pixel_check, lambda: jnp.asarray(True)
    )
    result = overlap & collide_when_overlap
    # NOT_BLOCKED on either sprite disables collision entirely.
    return result & ~any_not_blocked


__all__ = ["bbox_overlap", "pair_collides"]
