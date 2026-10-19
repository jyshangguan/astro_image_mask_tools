#!/usr/bin/env python3
"""Build examples/02_draw_interactive.ipynb.

Kept as a script rather than hand-edited JSON so the notebook stays
regenerable and reviewable in diffs.  Run from anywhere:

    python examples/build_notebook.py            # (re)write the notebook
    python examples/build_notebook.py --verify   # ...and prove it executes

--verify executes a throwaway copy.  Do not `nbconvert --inplace` the real
file: ipympl dumps ~3.5 MB of widget state into metadata.widgets, which would
then be committed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "examples" / "02_draw_interactive.ipynb"

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C: list = []


# ============================================================ 0. title
C += [
    md(
        """# Drawing a mask by hand, in the notebook

Two ways to make a mask with `masktools`:

| | use it when |
|---|---|
| **`MaskDrawer`** (this notebook) | you want to trace an irregular region with the mouse |
| **`grow_mask` / `scale_mask`** (§4) | you already have a rough mask and need margin or magnification |

Everything here runs headless too: the interactive cells detect a non-GUI
backend and fall back to building the same polygon from coordinates, so this
notebook is a working test as much as a tutorial.
"""
    ),
    code(
        """%matplotlib widget
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import masktools as mt

print("matplotlib backend:", matplotlib.get_backend())

# Draw with the mouse, or build the same polygons from coordinates?
#
# Set this to True in a live notebook and the cells below open the widget and
# wait for you.  It stays False for a headless run (`jupyter nbconvert
# --execute`, CI), where there is nobody to click -- a notebook whose backend
# merely *supports* widgets would otherwise produce an empty mask and quietly
# validate nothing.  The programmatic branch is not a fake: it exercises the
# same drawer, the same rasteriser and the same save path.
DRAW_BY_HAND = False

def _demo_frame():
    # Locate data/example_cutout.fits.gz from wherever this notebook runs.
    #
    # Hard-coding "../data/..." breaks as soon as the notebook is opened from
    # another directory, run with nbconvert from a temp dir, or copied out.
    # Walk upward from the working directory the kernel starts in, and honour
    # MASKTOOLS_EXAMPLE for CI.
    import os
    from pathlib import Path

    named = os.environ.get("MASKTOOLS_EXAMPLE")
    candidates = [Path(named)] if named else []
    here = Path.cwd().resolve()
    for base in (here, *here.parents):
        candidates += [base / "data" / "example_cutout.fits.gz"]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "example_cutout.fits.gz not found; set MASKTOOLS_EXAMPLE to its path"
    )


PATH = str(_demo_frame())
print("using", PATH)
with fits.open(PATH) as hdul:
    image = np.asarray(hdul[0].data, dtype=float)
    shipped_mask = np.asarray(hdul[1].data).astype(bool)
    header = hdul[0].header

wcs = None
try:
    from astropy.wcs import WCS
    wcs = WCS(header)
except Exception:
    pass

print(f"image {image.shape}, {shipped_mask.sum()} pixels in the shipped mask")
print(f"drawing by hand: {DRAW_BY_HAND}")"""
    ),
]

# ============================================================ 1. look
C += [
    md(
        """## 1. Look at the field first

`plot_image` defaults to an `asinh` stretch clipped at the 16th and 99.9th
percentiles.  The asymmetric cut matters: this frame contains 582 exactly-zero
pixels (0.22% of it) inside the saturated star, and a *symmetric* clip — what
`astropy.visualization.simple_norm(percent=99.9)` does — drags `vmin` down to
them and turns the whole image grey.  Compare §8.
"""
    ),
    code(
        """ax = mt.plot_image(image, figsize=(8, 8))
mt.plot_mask(shipped_mask, ax=ax, color="red", alpha=0.45)
mt.plot_mask_contours(shipped_mask, ax=ax, edgecolor="cyan", lw=1.2)
ax.set_title("the shipped mask: a saturated star at (~107, ~422)")"""
    ),
]

# ============================================================ 2. draw
C += [
    md(
        """## 2. Draw

Construct the drawer over the image and show it.  Then, **in a real notebook**:

| gesture / key | effect |
|---|---|
| left click | place a vertex |
| move onto the first vertex, click | close the polygon |
| `n` | finish this polygon, start the next |
| `e` | re-open the last polygon and edit it |
| `d` / Backspace | delete the last polygon |
| right-click a vertex | remove that vertex |
| drag a vertex | move it |
| `Ctrl` + drag | move one vertex (the widget's own binding) |
| `Shift` + drag | move the whole polygon (the widget's own binding) |
| `Esc` | clear the polygon in progress |
| `s` | save to the `savepath` you passed |
| `q` / `c` | finish and close |
| `h` | print this table |

Outlines are drawn in **pixel coordinates** (`space="pixel"`), so what you see
lines up with `imshow(..., origin="lower")` and the numbers match
`WCS.pixel_to_world`.
"""
    ),
    code(
        """drawer = mt.MaskDrawer(data=image, mask_color="red", poly_color="cyan")

if DRAW_BY_HAND:
    drawer.show()          # live in the notebook; press q when done
else:
    # Headless: same result, built from coordinates instead of from clicks.
    # These are the vertices of the shipped star's neighbourhood, traced by hand.
    for ring_pts in [
        [(95, 410), (122, 410), (122, 436), (95, 436)],      # the core
    ]:
        drawer.add_polygon(ring_pts)
    drawer.add_polygon([(140, 470), (160, 470), (160, 492), (140, 492)])  # a galaxy
    print("built", len(drawer.vertices), "polygons programmatically")
    print(repr(drawer))"""
    ),
    md(
        """The drawer keeps two things apart: polygons you have **finished** (`n`) and the
one still being drawn.  `.vertices` and `.mask()` see both, so you can check
your progress mid-stroke without committing.
"""
    ),
    code(
        """print("finished polygons:", len(drawer._committed))
print("live vertices    :", len(drawer._selector.verts))
print("visible polygons :", len(drawer.vertices))
mask = drawer.mask()
print(f"drawn mask: {mask.sum()} pixels")"""
    ),
]

# ============================================================ 3. inspect
C += [
    md(
        """## 3. See what you drew

`plot_mask` fills, `plot_mask_contours` outlines.  Both take the same
half-pixel-aligned extent, so they sit on the same grid.
"""
    ),
    code(
        """ax = mt.plot_image(image)
mt.plot_mask(mask, ax=ax, color="red", alpha=0.35)
mt.plot_mask_contours(mask, ax=ax, edgecolor="lime", lw=1.2)
ax.set_title("hand-drawn mask over the image")"""
    ),
]

# ============================================================ 4. transform
C += [
    md(
        """## 4. Grow, shrink, scale

The usual reason to draw a tight mask is to then widen it.  `grow_mask` offsets
the outline by a **fixed pixel distance** — the margin is the same however big
the source is — while `scale_mask` **magnifies** about the centroid, so its
effect grows with the source.  Pick the one that matches your intent.
"""
    ),
    code(
        """print(f"{'operation':28s} {'pixels':>8s}")
print(f"{'drawn':28s} {mask.sum():8d}")
for d in (-2, -1, 0, 1, 2, 5):
    print(f"{f'grow_mask({d:+d})':28s} {mt.grow_mask(mask, d).sum():8d}")
for f in (0.5, 1.0, 1.5, 2.0):
    print(f"{f'scale_mask({f:.1f})':28s} {mt.scale_mask(mask, f).sum():8d}")"""
    ),
    code(
        """grown = mt.grow_mask(mask, 3)

fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharex=True, sharey=True)
for ax, (m, t, c) in zip(axes, [
    (mask, "drawn", "red"),
    (grown, "grow_mask(+3)", "orange"),
    (mt.scale_mask(mask, 1.5), "scale_mask(1.5)", "violet"),
]):
    mt.plot_image(image, ax=ax)
    mt.plot_mask(m, ax=ax, color=c, alpha=0.35)
    mt.plot_mask_contours(m, ax=ax, edgecolor="cyan", lw=1.0)
    ax.set_title(f"{t}  ->  {m.sum()} px")
fig.suptitle("draw tight, then widen")"""
    ),
    md(
        """### Three caveats worth knowing before you trust the numbers

**1. Rasterising a grown outline is quantised.** `rasterize` counts a pixel when
its *centre* falls inside, so the gap between the vector area and the pixel count
is bounded by about **half the outline length** — not by the area, which is why
it is proportionally worst on small or ragged regions.  The gap's *sign* is not
fixed: it flips as `distance` sweeps past a ring of pixel centres, and is worst
just below each half-integer.  `join_style="mitre"` is exact at multiples of 0.5
on a solid rectilinear outline, and nowhere else.

**2. Two things break even that mitre exactness.**  A surviving **hole** moves
against the pixel centres, so the raster comes out *larger* than the vector area
at half-integer distances.  And `grow_polygons` buffers each region on its own
while the mask is their **union**, so neighbouring regions that the buffer pushes
into overlap get double-counted by `sum(p.area ...)`.  Union them with
`shapely.unary_union` first if you need that comparison.

**3. Growing then shrinking is not an undo, and a sub-pixel grow is a no-op.**
`grow(+d)` then `grow(-d)` is a morphological opening: it erases anything thinner
than `2*d` and cannot restore corners.  Exact round-tripping needs *both* an
integer distance *and* a boxy outline — the shipped staircase loses pixels at
`d=3`.  At the other end, `distance < 0.5` adds **nothing at all** (centres are
1 pixel apart), then jumps by a whole ring at exactly 0.5.
"""
    ),
    code(
        """from shapely import unary_union

# Compare like with like: the ANALYTIC buffered outline (grow_polygons) against
# the rasterised count (grow_mask).  Re-vectorising the grown mask instead would
# be circular -- it returns the pixel count by construction.
print("-- analytic vector area vs rasterised pixel count --")
for src, name in ((mask, "drawn boxes"), (shipped_mask, "shipped staircase")):
    polys = mt.mask_to_polygons(src)
    for join in ("mitre", "round"):
        for d in (0.49, 0.5, 1.0, 1.5):
            gp = mt.grow_polygons(polys, d, join_style=join)
            g = mt.grow_mask(src, d, join_style=join)
            vec, pix = sum(p.area for p in gp), int(g.sum())
            bound = sum(p.length for p in gp) / 2
            print(f"{name:17s} {join:5s} d={d:<5} vector={vec:8.2f} pixels={pix:5d}"
                  f"  diff={vec - pix:+8.2f}  |bound|={bound:6.1f}")
print("The diff tracks outline LENGTH (<= ~perimeter/2), not area, and its sign")
print("flips around each half-integer.  mitre is exact only at multiples of 0.5.")"""
    ),
    code(
        """# The two exceptions to mitre exactness, both worth recognising.
print("-- a surviving hole inverts the sign --")
donut = np.zeros((60, 60), bool)
donut[10:40, 10:40] = True
donut[22:28, 22:28] = False
dp = mt.mask_to_polygons(donut)
print(f"hole present: {len(dp[0].interiors)} interior ring, {int(donut.sum())} px")
for d in (1.0, 0.5, 1.5):
    gp = mt.grow_polygons(dp, d, join_style="mitre")
    pix = int(mt.polygons_to_mask(gp, donut.shape).sum())
    print(f"  d={d:<4} vector={sum(p.area for p in gp):8.1f}  pixels={pix:5d}"
          f"  diff={sum(p.area for p in gp) - pix:+7.1f}")
print("Growing shrinks the hole; at a half-integer d its boundary lands on")
print("excluded pixel centres, so the raster keeps pixels the vector gave up.")

print("\\n-- overlapping neighbours double-count --")
pair = np.zeros((60, 60), bool)
pair[10:40, 15:50] = True
pair[20:25, 5:12] = True          # 3 columns clear of the big box
pp = mt.mask_to_polygons(pair)
for d in (1.5, 2.0):
    gp = mt.grow_polygons(pp, d, join_style="mitre")
    out = mt.polygons_to_mask(gp, pair.shape)
    naive, merged = sum(p.area for p in gp), unary_union(gp).area
    print(f"  d={d}: sum(areas)={naive:8.1f}  union={merged:8.1f}  pixels={int(out.sum()):5d}")
print("grow_polygons buffers each region alone but the mask is a union, so at")
print("d=2.0 the 1x9 overlap is counted twice by sum() and once by rasterise.")
print("Union the polygons first (or just trust .sum()) when comparing areas.")"""
    ),
    code(
        """# (2) Opening, not inversion.  Boxy input at integer d round-trips exactly --
# which would prove nothing -- so both shapes and both kinds of d are shown.
print("-- grow(+d) then grow(-d) --")
for src, name in ((mask, "drawn boxes"), (shipped_mask, "shipped staircase")):
    for d in (3, 2.5):
        back = mt.grow_mask(mt.grow_mask(src, d), -d)
        iou = (back & src).sum() / (back | src).sum()
        print(f"{name:18s} d={d:<5} {src.sum():5d} -> {back.sum():5d} px"
              f"  IoU={iou:.3f}  exact={np.array_equal(back, src)}")
print("Only boxes-at-integer-d is reversible.  A staircase loses pixels even at")
print("d=3, and a fractional d eats its notches.  Never grow 'temporarily'.")

print("\\n-- and below half a pixel, nothing happens at all --")
tight = np.zeros((200, 200), bool)
tight[50:150, 50:150] = True
for d in (0.01, 0.25, 0.49, 0.5, 1.0):
    print(f"  d={d:<5} -> {int(mt.grow_mask(tight, d).sum()):6d} px"
          f"  (was {int(tight.sum())})")
print("Pixel centres are 1 apart, so d < 0.5 cannot reach a new ring.  The")
print("count jumps by a whole ring at 0.5 -- there is no in-between.")"""
    ),
]

# ============================================================ 5. refine
C += [
    md(
        """## 5. Refine an existing mask

Pass a mask in and the drawer converts it to editable polygons
(`mask_to_polygons`) rather than treating it as a picture to trace over.  Delete
the pieces you do not want, add the ones you do, and the result re-rasterises.

This is the workflow for a threshold mask with a spurious detection in it.
"""
    ),
    code(
        """# A threshold mask with one bright star plus a few stray pixels
bkg = np.median(image)
sigma = 1.4826 * np.median(np.abs(image - bkg))
thresh = np.zeros_like(image, dtype=bool)
thresh[60:110, 440:490] = image[60:110, 440:490] > bkg + 5 * sigma
print(f"threshold mask: {thresh.sum()} px, "
      f"{len(mt.mask_to_polygons(thresh))} component(s)")

refine = mt.MaskDrawer(data=image, mask=thresh)
print(repr(refine))
if DRAW_BY_HAND:
    refine.show()          # delete the bits you do not want, then q
else:
    # headless: drop the smallest stray component -- polygons arrive largest
    # first, so the last one is the speck -- then add a region by hand.
    refine.delete_last()
    refine.add_polygon([(455, 75), (475, 75), (475, 95), (455, 95)])
    print("after delete_last + add_polygon:", len(refine.vertices))

refined = refine.mask()
print(f"refined mask: {refined.sum()} px")"""
    ),
]

# ============================================================ 6. save
C += [
    md(
        """## 6. Save it, with the WCS attached

`save()` writes `.npy` with numpy alone, or FITS through astropy.  Pass the
image's own header and the mask inherits its WCS, so it stays aligned when you
open it in ds9 / topcat / a later notebook.
"""
    ),
    code(
        """import os
os.makedirs("output", exist_ok=True)  # scratch, gitignored

npy_path = drawer.save("output/drawn_mask.npy")
fits_path = drawer.save("output/drawn_mask.fits", header=header)
print("wrote", npy_path, "and", fits_path)

with fits.open(fits_path) as hdul:
    round_trip = np.asarray(hdul[0].data).astype(bool)
    out_header = hdul[0].header
print("FITS round trip identical:", np.array_equal(round_trip, mask))
print("CTYPE preserved:", out_header.get("CTYPE1"), "/", out_header.get("CTYPE2"))
print("CRPIX preserved:", out_header.get("CRPIX1"), out_header.get("CRPIX2"))"""
    ),
    md(
        """### From a drawn mask to sky coordinates

Because the outline is in pixel space, `WCS.pixel_to_world` takes it directly —
no half-pixel correction anywhere:
"""
    ),
    code(
        """polys = mt.mask_to_polygons(mask)
if not polys:
    print("empty mask; nothing to project")
elif wcs is None:
    print("no WCS in this file; skipping")
else:
    cx, cy = polys[0].centroid.x, polys[0].centroid.y
    sky = wcs.pixel_to_world(float(cx), float(cy))
    print(f"drawn mask centroid (pixel {cx:.2f}, {cy:.2f}) -> {sky.to_string('hmsdms', precision=3)}")
    # the same call with no half-pixel correction anywhere: the polygon is
    # already in pixel-centre space
    ra, dec = sky.ra.deg, sky.dec.deg
    print(f"                                            = {ra:.6f}, {dec:.6f} deg")"""
    ),
]

# ============================================================ 7. one-liner
C += [
    md(
        """## 7. The one-liner

`draw_mask` opens the drawer, waits, and hands back the array — for when you do
not need to touch the polygons afterwards.
"""
    ),
    code(
        """if DRAW_BY_HAND:
    one_shot = mt.draw_mask(data=image)          # blocks, returns ndarray
    print(type(one_shot), one_shot.sum())
else:
    # block=False returns the drawer instead of the array, so this stays runnable
    one_shot = mt.draw_mask(shape=(100, 100), block=False, verbose=False)
    one_shot.add_polygon([(10, 10), (40, 10), (40, 40), (10, 40)])
    arr = one_shot.mask()
    print(type(arr), arr.shape, arr.sum())
    one_shot.teardown()"""
    ),
]

# ============================================================ 8. stretch
C += [
    md(
        """## 8. Why the stretch default is asymmetric

Same image, three clipping rules.  The left panel is what you get from
`astropy.visualization.simple_norm(percent=99.9)` on a frame that contains a
small population of zeroed pixels — the symmetric low cut lands on them, and
every faint source disappears into a flat grey.
"""
    ),
    code(
        """fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharex=True, sharey=True)
for ax, (label, kw) in zip(axes, [
    ("percent=99.9  (astropy)", dict(percent=99.9)),
    ("masktools default", {}),
    ("plain min/max", dict(min_percent=None, max_percent=None)),
]):
    norm = mt.make_norm(image, "asinh", **kw)
    mt.plot_image(image, ax=ax, norm=norm)
    ax.set_title(f"{label}\\nvmin={norm.vmin:.0f}  vmax={norm.vmax:.0f}", fontsize=10)
fig.suptitle("asinh stretch, three clipping rules")"""
    ),
    code(
        """# The same numbers, printed, so the claim is checkable
for label, kw in [
    ("percent=99.9 (astropy)", dict(percent=99.9)),
    ("default min16/max99.9", {}),
    ("min/max", dict(min_percent=None, max_percent=None)),
]:
    n = mt.make_norm(image, "asinh", **kw)
    print(f"{label:24s} vmin={n.vmin:8.1f}  vmax={n.vmax:8.1f}")
print(f"sky level = {np.median(image):.1f}, "
      f"pixels below 100 = {int((image < 100).sum())} "
      f"({(image < 100).mean() * 100:.2f}% of the frame)")"""
    ),
]

# ============================================================ 9. popup
C += [
    md(
        """## 9. Outside a notebook: the popup path

The same class works from a plain script; `show()` blocks on a real window
backend.  Nothing about the API changes.

```python
# examples/03_popup_draw.py
import matplotlib
matplotlib.use("TkAgg")          # or "macosx" / "QtAgg"
import masktools as mt

image = ...                      # your frame
drawer = mt.MaskDrawer(data=image, savepath="mask.fits")
drawer.show()                    # blocks until you press q or close
mask = drawer.mask()
```

Press `s` at any point to write the `savepath`, `q` when done.
"""
    ),
    code(
        """# Nothing to execute here -- see examples/03_popup_draw.py, which is run
# (headless, programmatically) by examples/01_vectorize_and_transform.py."""
    ),
]

# ============================================================ 10. recap
C += [
    md(
        """## Recap

```python
import masktools as mt

mask  = mt.draw_mask(data=image)                     # 1. draw by hand
polys = mt.mask_to_polygons(mask)                    # 2. vectorise
grown = mt.grow_mask(mask, 3)                        # 3. add margin
big   = mt.scale_mask(mask, 2.0)                     #    ...or magnify
back  = mt.polygons_to_mask(polys, image.shape)      # 4. rasterise again
drawer.save("mask.fits", header=header)              # 5. keep the WCS
```

Two conventions hold the whole thing together:

* **`space="pixel"`** (the default) puts integer coordinates at pixel *centres*,
  matching `imshow(origin="lower")` and `WCS.pixel_to_world`.  `space="corner"`
  gives the rasterio/GDAL corner grid instead.  Mixing them shifts every
  outline by half a pixel — 0.12″ on this frame.
* Distances in `grow_mask` are **pixels**; factors in `scale_mask` are
  **multiplications of area by factor²**.  They are not interchangeable.
"""
    ),
]

nb["cells"] = C
nb["metadata"].update(
    {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
)

nbf.validate(nb)
nbf.write(nb, str(OUT))
print(f"wrote {OUT.relative_to(ROOT)} with {len(C)} cells")

if "--verify" in sys.argv:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "check.ipynb"
        shutil.copy(OUT, tmp)
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"),
                   MASKTOOLS_EXAMPLE=str((ROOT / "data" / "example_cutout.fits.gz")))
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
            "--execute", "--inplace", "--timeout", "300", str(tmp),
        ]
        print("executing a throwaway copy ...")
        rc = subprocess.call(cmd, env=env, cwd=str(OUT.parent))
        if rc:
            sys.exit(f"nbconvert exited with {rc}")
        done = nbf.read(str(tmp), as_version=4)
        errs = [
            (i, o)
            for i, c in enumerate(done["cells"])
            for o in c.get("outputs", [])
            if o.get("output_type") == "error"
        ]
        code = [c for c in done["cells"] if c["cell_type"] == "code"]
        ran = [c for c in code if c.get("execution_count")]
        for i, o in errs:
            print(f"  ERROR cell {i}: {o['ename']}: {o['evalue']}")
        print(f"verified: {len(ran)}/{len(code)} code cells ran, {len(errs)} errors")
        if errs:
            sys.exit(1)
