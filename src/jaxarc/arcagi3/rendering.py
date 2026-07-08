"""Frame rendering for the ARC-AGI-3 runtime subset.

Faithful JAX port of ``arcengine.Camera`` (verified against ``arcprize/ARCEngine``
``main``). Two stages:

1. ``composite`` — the equivalent of ``Camera._raw_render``: fill a native-size
   canvas with the background color, then paint active & visible sprites in
   ascending layer order (lower layers first, higher layers overwrite),
   respecting transparent (-1) pixels. Insertion order breaks layer ties
   (stable sort), matching ARCEngine.

2. ``render`` — the equivalent of ``Camera.render``: uniformly upscale the native
   view by ``scale = min(64 // cam_w, 64 // cam_h)`` and center it on a 64x64
   canvas filled with the letter-box color. Implemented as a dynamic gather so it
   stays ``jit``/``vmap`` friendly even though ``cam_w``/``cam_h`` vary per level.

The output frame contains only values ``0..15`` (transparent regions fall through
to the background; nothing renders ``-1``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int

from .constants import GRID_SIZE
from .types import ArcAgi3Params, ArcAgi3State


def _tile_to_grid_values(
    pixels: Int[Array, "sh sw"],
    x: Int[Array, ""],
    y: Int[Array, ""],
    height: int,
    width: int,
) -> tuple[Int[Array, "H W"], Int[Array, "H W"]]:
    """Scatter a sprite tile onto an (H, W) canvas at (x, y).

    Returns ``(values, mask)`` where ``mask`` is 1 for in-bounds non-transparent
    tile pixels. Out-of-bounds pixels are dropped.
    """
    sh, sw = pixels.shape
    rows = jnp.broadcast_to(jnp.arange(sh)[:, None] + y, (sh, sw))
    cols = jnp.broadcast_to(jnp.arange(sw)[None, :] + x, (sh, sw))

    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    # ARCEngine paints only non-negative pixels (``>= 0``). Both -1 (fully
    # transparent) and -2 (invisible wall: solid for collision, not drawn) are
    # skipped here; collision uses ``!= -1`` (see collisions.py) so -2 still blocks.
    place = in_bounds & (pixels >= 0)

    lin = jnp.clip(rows, 0, height - 1) * width + jnp.clip(cols, 0, width - 1)
    lin = lin.reshape(-1)

    values = jnp.zeros((height * width,), dtype=jnp.int32)
    mask = jnp.zeros((height * width,), dtype=jnp.int32)
    values = values.at[lin].set(pixels.reshape(-1).astype(jnp.int32), mode="drop")
    mask = mask.at[lin].max(place.reshape(-1).astype(jnp.int32), mode="drop")
    # Zero out values where nothing was placed (in case set landed on a dropped/clamped cell).
    values = values * mask
    return values.reshape(height, width), mask.reshape(height, width)


def composite(params: ArcAgi3Params, state: ArcAgi3State) -> Int[Array, "H W"]:
    """Render active & visible sprites onto a native (height, width) canvas.

    Equivalent to ``Camera._raw_render``: background fill, then sprites painted in
    ascending layer order with transparency.
    """
    h, w = params.height, params.width
    canvas = jnp.full((h, w), params.background, dtype=jnp.int32)

    # Per-sprite layer (inactive/invisible pushed to the back so their position in
    # the draw order is irrelevant; they're masked out when painted anyway).
    layer = params.sprite_layer[state.sprite_kind]  # (max_sprites,)
    drawable = state.sprite_active & state.sprite_visible
    sort_key = jnp.where(drawable, layer, jnp.iinfo(jnp.int32).max)
    order = jnp.argsort(sort_key, stable=True)  # insertion order breaks ties

    def paint(
        canvas: Int[Array, "H W"], idx: Int[Array, ""]
    ) -> tuple[Int[Array, "H W"], None]:
        kind = state.sprite_kind[idx]
        pixels = params.sprite_pixels[kind]  # (sprite_h, sprite_w)
        values, mask = _tile_to_grid_values(
            pixels, state.sprite_x[idx], state.sprite_y[idx], h, w
        )
        paint_here = (mask > 0) & drawable[idx]
        return jnp.where(paint_here, values, canvas), None

    canvas, _ = jax.lax.scan(paint, canvas, order)
    return canvas


def scale_and_letterbox(
    view: Int[Array, "H W"],
    cam_h: Int[Array, ""],
    cam_w: Int[Array, ""],
    letter_box: int,
) -> Int[Array, "64 64"]:
    """Upscale the top-left ``cam_h x cam_w`` region of ``view`` onto a 64x64 canvas.

    Matches ``Camera.render``: ``scale = min(64 // cam_w, 64 // cam_h)``, centered
    with letter-box padding. Fully dynamic (no static repeat), so JIT-safe.
    """
    scale = jnp.minimum(GRID_SIZE // cam_w, GRID_SIZE // cam_h)
    scale = jnp.maximum(scale, 1)
    scaled_w = cam_w * scale
    scaled_h = cam_h * scale
    x_off = (GRID_SIZE - scaled_w) // 2
    y_off = (GRID_SIZE - scaled_h) // 2

    out_rows = jnp.arange(GRID_SIZE)[:, None]
    out_cols = jnp.arange(GRID_SIZE)[None, :]
    src_r = (out_rows - y_off) // scale
    src_c = (out_cols - x_off) // scale

    inside = (
        (out_rows >= y_off)
        & (out_rows < y_off + scaled_h)
        & (out_cols >= x_off)
        & (out_cols < x_off + scaled_w)
    )
    gathered = view[
        jnp.clip(src_r, 0, view.shape[0] - 1), jnp.clip(src_c, 0, view.shape[1] - 1)
    ]
    return jnp.where(inside, gathered, letter_box).astype(jnp.int32)


def draw_energy_overlay(
    frame: Int[Array, "64 64"], params: ArcAgi3Params, state: ArcAgi3State
) -> Int[Array, "64 64"]:
    """Draw the energy-pill border onto a 64x64 frame (``Camera`` interface stage).

    Reproduces ``ToggleableUserDisplay.render_interface``: ``num_ui_pills`` pills,
    each ``ui_pill_size x ui_pill_size``, at fixed display coordinates
    (``ui_pill_x``/``ui_pill_y``). Pills are consumed in list order — one per
    non-RESET action — so pill ``i`` shows the "off" color once
    ``i < action_count`` and the "on" color otherwise. A no-op when
    ``num_ui_pills == 0`` (the default), so games without energy are unaffected.
    """
    n = params.num_ui_pills
    if n == 0:
        return frame

    size = params.ui_pill_size
    consumed = jnp.arange(n) < state.action_count  # (n,) True => off color
    color = jnp.where(consumed, params.ui_pill_off_color, params.ui_pill_on_color)

    # Scatter each pill's size x size block onto the frame.
    def paint(
        frame: Int[Array, "64 64"], i: Int[Array, ""]
    ) -> tuple[Int[Array, "64 64"], None]:
        px, py = params.ui_pill_x[i], params.ui_pill_y[i]
        rows = jnp.clip(py + jnp.arange(size), 0, GRID_SIZE - 1)
        cols = jnp.clip(px + jnp.arange(size), 0, GRID_SIZE - 1)
        block = jnp.full((size, size), color[i], dtype=jnp.int32)
        return frame.at[jnp.ix_(rows, cols)].set(block), None

    frame, _ = jax.lax.scan(paint, frame, jnp.arange(n))
    return frame


def render(params: ArcAgi3Params, state: ArcAgi3State) -> Int[Array, "64 64"]:
    """Full render to a 64x64 frame, matching ``arcengine.Camera.render``.

    ARCEngine resizes its camera to the current level's ``grid_size`` and then
    uniformly upscales that view to 64x64 with letter-box padding, and finally
    lets registered UI interfaces draw over the frame. We reproduce that:
    composite onto the full 64x64 canvas, take the top-left ``cam_h x cam_w``
    region for the active level, scale + letterbox, then draw the energy overlay.

    Note: composite fills the whole 64x64 with ``background``; since sprites live
    in the top-left ``cam_h x cam_w`` block, cropping to that block reproduces the
    engine's raw camera view before upscaling.
    """
    canvas = composite(params, state)
    cam_w = params.level_cam_w[state.level_index]
    cam_h = params.level_cam_h[state.level_index]
    frame = scale_and_letterbox(canvas, cam_h, cam_w, params.letter_box)
    return draw_energy_overlay(frame, params, state)


__all__ = ["composite", "draw_energy_overlay", "render", "scale_and_letterbox"]
