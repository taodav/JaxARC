# Build Plan: `merge` game (ARC-AGI-3 runtime)

Concrete, phased plan to port ARCEngine's `examples/merge.py` into
`src/jaxarc/arcagi3`. Builds on the scoping in
[`arcagi3_merge_games_design.md`](arcagi3_merge_games_design.md). Verified
against `arcprize/ARCEngine` `main`. Scope: `merge` only (`merge_detach`/ACTION5
is a later increment).

Guiding principle unchanged from the maze ports: **the Tier-3 parity harness is
the source of truth.** Every phase ends by comparing byte-for-byte against the
official engine; unit tests alone missed real bugs before.

## The one architectural change

Today both rendering (`rendering.composite`) and collision
(`engine.try_move_sprite`) read a **per-kind static tile**:
`params.sprite_pixels[state.sprite_kind[i]]`. The merge player reshapes at
runtime, so we move pixels from *per-kind* (Params) to *per-sprite* (State):

- **New:** `ArcAgi3State.sprite_pixels: int32[max_sprites, sprite_h, sprite_w]` —
  each active slot carries its own current pixels (TRANSPARENT-padded).
- Rendering/collision read `state.sprite_pixels[i]` instead of
  `params.sprite_pixels[kind]`.
- `params.sprite_pixels` stays as the per-kind **initial** tiles; `reset`/
  `load_level` gather them into the per-sprite buffer (`state.sprite_pixels =
  params.sprite_pixels[init_sprite_kind[level]]`).

This is **additive and backward-compatible**: maze games' per-sprite buffers are
just constant copies of their kind tiles, so their behaviour (and golden/parity
fixtures) must stay **byte-identical**. That invariant is the guardrail for
Phase 1.

## Phases

### Phase 0 — Prep & ground-truth (no behaviour change)
- Re-fetch `examples/merge.py` + `get_pixels_at_sprite`/`get_pixels`/`merge`
  from `sprites.py` and pin the exact semantics (see "Fidelity details" below).
- Vendor `test/arcagi3/reference/merge_reference.py` (verbatim MIT copy, header +
  ruff-excluded, like the maze references).
- Confirm `max_sprite_h/w` bound: the merged player can grow to the union bbox of
  everything it absorbs. Analyze all 3 levels' sprites to pick a safe static
  `sprite_h/w` for merge (likely up to the 16×16 level, since sprite-1 is 16-wide
  — but the *player* only merges `"merge"`-tagged pieces, so compute the real max).

### Phase 1 — Per-sprite pixel buffer (refactor, maze parity preserved)
- Add `sprite_pixels` to `ArcAgi3State`; initialize in `engine.load_level`/
  `env._empty_state` by gathering `params.sprite_pixels[kind]`.
- Repoint `rendering.composite` and `engine.try_move_sprite` (and
  `_bbox_dims`/`pair_collides` callers) to read `state.sprite_pixels[i]`.
- **Gate:** regenerate nothing; run existing golden + Tier-3 parity for
  simple_maze **and** complex_maze — must remain byte-identical. This phase ships
  only if maze parity is untouched.

### Phase 2 — `merge` engine op
- `merge_into(state, p, c) -> state`: composite slot `c`'s non-transparent pixels
  under slot `p`'s at the combined bbox, write into `p`'s buffer, update `p`'s
  (x, y) to the new top-left, deactivate `c`. Bounded, masked, `jit`-safe (no
  dynamic shapes). Port of `Sprite.merge` (self-pixels take precedence; combined
  bbox = min corner to max corner of both).
- Unit-test the op in isolation against hand-computed merges before wiring the
  game.

### Phase 3 — pixel-equality win
- `pixels_at(state, params, slot) -> (window, valid_mask)`: render the raw camera
  (reuse `composite` pre-scale) and extract the slot's bbox region — the port of
  `get_pixels_at_sprite`.
- Win = `np.array_equal(pixels_at(player), pixels_at(target))`, which in JAX means
  **same bbox dims AND equal content**. Differing dims ⇒ auto-False (handle via a
  dims-match guard over the padded window).

### Phase 4 — `transition_merge` + params
- Transition: ACTION1–4 move player via `try_move_player`; for each collided
  `"merge"`-tagged sprite, `merge_into` then move player by (dx,dy); check win →
  `next_level_or_win`; else if `≤1 merge sprite remains` → `lose_game`.
- Params for 3 levels (16×16), sprites player/sprite-1..7 with real pixel tiles,
  `merge`/`target` tag flags (reuse the per-kind flag pattern from complex_maze).
- Register `merge` in `games/__init__.py` (`TRANSITION_MERGE`, `GAME_BUILDERS`).

### Phase 5 — tests, fixtures, docs
- Golden traces + fixtures (`.npz` frames, newline-terminated JSON — the
  large-file/EOF lessons already apply).
- Engine unit tests: merge compositing, pixel-equality win, lose-on-≤1-merge.
- Tier-3 `test_parity_merge.py` vs the vendored reference; solver in
  `golden_common.solve_game` already generalizes (drives the real env), but merge
  may need the search to consider merges — verify it can reach WIN, else hand-author
  a winning trace.
- Update the games table in `arcagi3-environments.md`.

## Phase 0 findings (pinned from ARCEngine `main`)

**Sprite shapes (H×W)** and **tags** (from `examples/merge.py`):

| sprite | H×W | tag | notes |
|---|---|---|---|
| player | 1×1 | merge | pixels `[[9]]`; the absorbing sprite |
| sprite-1 | 16×16 | *(none)* | the maze walls; blocks, never merged |
| sprite-2 | 2×2 | merge | |
| sprite-3 | 2×2 | merge | |
| sprite-4 | 2×2 | target | win = player pixels == this |
| sprite-5 | 3×1 | merge | |
| sprite-6 | 3×3 | target | |
| sprite-7 | 3×4 | target | |

- **Only `"merge"`-tagged sprites are absorbed** (player, sprite-2/3/5). `"target"`
  sprites and the untagged maze (sprite-1) are never merged.
- **Per-level target** is the single `"target"`-tagged sprite (level 1 sprite-4,
  level 2 sprite-6, level 3 sprite-7).
- **Worst-case merged-player bbox** (union of merge sprites' initial positions):
  level 1 → 6×3, level 2 → 10×7, level 3 → 8×10. Merged pieces move with the
  player so exact extents shift, but all pieces live within the 16×16 grid.
  **Decision: `sprite_h = sprite_w = 16`** (the grid size) — a guaranteed upper
  bound that eliminates merge-clipping risk with negligible cost. (Contrast: the
  mazes used 12×12.)

**`Sprite.merge(other)` semantics (exact):**
- Rendered extents via `render().shape`. Combined bbox: `min_x = min(self.x,
  other.x)`, `min_y = min(self.y, other.y)`, `max_x = max(self.x+self.w,
  other.x+other.w)`, `max_y` likewise.
- Fill order: paint `other`'s non-transparent pixels first, then `self`'s over the
  top (**self wins ties**).
- New sprite: `name=self.name`, position `(min_x, min_y)`, `layer=max(...)`, tags
  = union. In our model: write composited pixels into the player's slot buffer,
  set player `(x,y) = (min_x, min_y)`, deactivate the absorbed slot.

**Win (`check_win_condition`):** `np.array_equal(get_pixels_at_sprite(player),
get_pixels_at_sprite(target))`. `get_pixels_at_sprite(s)` = `get_pixels(s.x -
cam.x, s.y - cam.y, s.width, s.height)` = a slice of **`camera._raw_render(...)`**
(the pre-scale/letterbox raw view — our `rendering.composite`) at the sprite's
bbox. So it compares two raw-camera windows sized to each sprite's bbox; differing
bbox sizes ⇒ `np.array_equal` is False.

**Colors:** `BACKGROUND_COLOR = 1`, `PADDING_COLOR = 3` (not 0 like the mazes).

## Fidelity details to get exactly right (parity will catch misses)

1. **`merge()` precedence & bbox:** the new sprite spans `min(x)`..`max(x+w)` of
   both; `self` (player) non-transparent pixels win over `other`'s. Verify the
   anchor math — the maze collision bug taught us relative-offset placement is
   error-prone.
2. **Win compares rendered regions, not sprites:** `get_pixels` renders the whole
   raw camera then slices the bbox — so the compared window includes background
   and anything overlapping. Reproduce that, don't shortcut to sprite-pixel
   equality.
3. **Dims-match requirement:** `np.array_equal` is False if player and target
   bboxes differ in size. The JAX version must return False on size mismatch, not
   silently compare a padded window.
4. **Lose timing:** checked *after* the win check each step (`≤1 merge sprite`),
   using the tag count — mirror the order.
5. **Background/padding:** merge uses `BACKGROUND_COLOR=1`, `PADDING_COLOR=3`
   (not 0 like the mazes) — thread through params, don't hardcode.

## Risk register

- **`max_sprite_h/w` too small ⇒ silent merge clipping.** Highest-risk item;
  bound it from the actual level sprites in Phase 0. Parity catches it, but it may
  need iteration.
- **Phase 1 regressing maze parity.** Mitigated by running the full existing
  parity/golden suite as the phase gate before any merge code lands.
- **Solver can't reach WIN for merge** (merges expand the state space). Fallback:
  hand-author a winning trace (drive the vendored reference engine, like we did
  when diagnosing complex_maze L5), commit it as the golden trace.

## Out of scope (this plan)
`merge_detach` / `ACTION5` "Detach All", the detached-tray UI overlay, and the
`Discrete(5)` wrapper — a follow-up once `merge` is parity-clean.
