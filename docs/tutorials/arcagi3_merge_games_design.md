# Design Note: Porting the ARCEngine "merge" games

Scoping for the two remaining ARCEngine demo games — `merge` and
`merge_detach` — verified against `arcprize/ARCEngine` `main`. **No code yet;**
this note exists so we can agree scope before building, the same way we planned
the maze ports.

## TL;DR

The merge games are **not** a same-tier addition like `complex_maze` was. Both
require a **dynamic per-sprite pixel model** — the player sprite's shape and size
*change at runtime* as it absorbs other sprites. Our current
`ArcAgi3State`/`ArcAgi3Params` design assumes a fixed set of sprite *kinds* with
static per-kind pixel tiles, which the maze games fit perfectly and the merge
games break. This is closer to the original plan's "PR v2: interaction mechanics"
milestone than to "another movement game."

Recommended path: implement `merge` first on a new dynamic representation, defer
`merge_detach` (adds `ACTION5` + detach history) as a follow-up.

## What the games do

### `merge` (3 levels, 16×16 grid)
- Player moves ACTION1–4. Colliding with a `"merge"`-tagged sprite **merges** it
  into the player: `self._player = self._player.merge(collide)` creates a NEW
  sprite whose pixels are the union of both (at a new, larger bounding box); the
  two originals are removed and the merged sprite is added, then moved by (dx,dy).
- **Win:** the merged player's rendered pixels exactly equal a `"target"`
  sprite's pixels — `np.array_equal(get_pixels_at_sprite(player), get_pixels_at_sprite(target))`.
- **Lose:** when only ≤1 `"merge"` sprite remains and win isn't met.

### `merge_detach` (adds `ACTION5`)
- Same core, plus **`ACTION5` = "Detach All"**: restores every absorbed sprite to
  the board and resets the player to its original single-cell form.
- `attach` tracks absorbed sprites in a `_detached` list + a UI overlay;
  `move_detached` moves the whole detached group with the player; `detatch_all`
  clears the list and re-adds the originals.
- Requires per-player **merge history** and a `Discrete(5)` action wrapper.

## Why our current model doesn't fit

| Assumption today (maze games) | What merge needs |
|---|---|
| Fixed `num_kinds`; `sprite_pixels[kind]` static tile | Player pixels **change at runtime** (union of merged shapes) |
| Sprites only translate (x, y change) | Sprites are **created/destroyed/reshaped** mid-episode |
| Win = player overlaps goal kind | Win = **pixel-equality** of a rendered region vs a target |
| Collision/kind lookups by `sprite_kind` | Merged sprite has no fixed kind; needs a per-slot pixel buffer |

`merge()` in ARCEngine allocates a fresh array sized to the combined bounding box
and composites non-transparent pixels (self takes precedence). In fixed-shape
JAX, we can't grow arrays — we'd pad to a max sprite size and carry a live pixel
buffer per sprite slot.

## Proposed representation change

Add a **per-sprite dynamic pixel buffer** (not per-kind):

- `ArcAgi3State.sprite_pixels: int32[max_sprites, max_sprite_h, max_sprite_w]` —
  each active sprite carries its own current pixels (TRANSPARENT-padded). Maze
  games would initialize these from the per-kind tiles once at reset; merge games
  mutate the player's buffer on each merge.
- `merge` op: given player slot `p` and collided slot `c`, compute the combined
  bounding box relative to a shared anchor, composite `c`'s non-transparent pixels
  under `p`'s, write the result into `p`'s buffer, update `p`'s (x, y) to the new
  top-left, and deactivate `c`. All at fixed `max_sprite_h × max_sprite_w`, so it
  stays `jit`-able (bounded, masked writes — no dynamic shapes).
- Win check: render the player's buffer region and the target's, compare equal
  over the padded window (mask to non-transparent union).

This is additive: the maze games keep working (their per-sprite buffers are just
constant copies of the kind tiles). It's ~a new `merge` engine op + a
pixel-equality win predicate + the state field. Rendering/collision already
operate on per-tile pixels, so they adapt with modest changes to read
per-sprite buffers instead of per-kind tiles.

`merge_detach` then adds: an `ACTION5` path, a per-player detached-history
structure (bounded stack of absorbed sprite slots), `move_detached`, and a
`Discrete(5)` wrapper — plus a UI overlay for the detached tray (like the energy
pills, drawable from state).

## Rough effort estimate

| Piece | Effort | Notes |
|---|---|---|
| Per-sprite dynamic pixel buffer in State + reset/rendering/collision reads | Medium–Large | Touches the core representation; must keep maze parity byte-exact |
| `merge` compositing op (bounded, masked) | Medium | The trickiest JAX bit: combined-bbox anchoring without dynamic shapes |
| Pixel-equality win vs target | Small–Medium | Render two regions, compare |
| `merge` game params (3 levels, 7 sprites) + golden + parity | Medium | Sprites have real shapes (16×16 level, up to ~16-wide sprites) |
| `merge_detach`: ACTION5, detach history, `move_detached`, Discrete(5) wrapper, UI tray | Large | Build only after `merge` is solid |

**Risk to watch:** `merge()`'s combined bounding box can exceed any single kind's
tile size, so `max_sprite_h/w` must be chosen to bound the largest possible merged
player per game (analyze the level sprites up front). Get this wrong and merges
silently clip — which the parity harness would catch, but it's the thing most
likely to need iteration.

## Recommendation

1. **`merge` first**, on the dynamic per-sprite buffer, with full golden + Tier-3
   parity (the harness will validate the merge compositing exactly as it did for
   the maze mechanics).
2. **`merge_detach` second**, once the dynamic representation and `ACTION5`
   plumbing are proven.

Both are genuinely portable — the parity harness makes the compositing/merge
mechanics verifiable — but this is a deliberate representation increment, not a
drop-in new game. Approve the representation change and we build `merge`.
