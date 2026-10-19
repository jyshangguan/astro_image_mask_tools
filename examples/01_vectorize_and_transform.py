"""Vectorise a real mask, then scale and grow it.

Runs headless and writes PNGs into ``examples/output/``, so it doubles as a
regression check on the shipped data file.  For the interactive drawing example
open ``02_draw_interactive.ipynb`` instead.

    python examples/01_vectorize_and_transform.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: this script only writes files

import matplotlib.pyplot as plt
from astropy.io import fits

import masktools as mt

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"   # committed: these are the documentation images
FITS_PATH = HERE.parent / "data" / "example_cutout.fits.gz"


def load():
    """The shipped 512x512 EFOSC g-band cutout of J0000-0541, and its mask."""
    with fits.open(FITS_PATH) as hdul:
        image = np.asarray(hdul[0].data, dtype=float)
        mask = np.asarray(hdul[1].data).astype(bool)
        header = hdul[0].header
    return image, mask, header


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    image, mask, header = load()
    print(f"image {image.shape} {image.dtype}, masked pixels: {mask.sum()}")

    # --------------------------------------------------------------- 1. vectorise
    # The shipped MASK holds one 582-pixel island around a saturated star.  Its
    # outline is a pixel staircase: 33 vertices for a shape that is basically a
    # blob, which is exactly what vectorising is for.
    polys = mt.mask_to_polygons(mask)
    print(f"{len(polys)} polygon(s), {len(polys[0].exterior.coords)} vertices")
    print(f"vectorised area {polys[0].area:.1f} px^2 vs pixel count {mask.sum()}")
    print(f"centroid ({polys[0].centroid.x:.2f}, {polys[0].centroid.y:.2f})")

    fig, ax = base_axes(image)
    mt.plot_mask(mask, ax=ax, color="red", alpha=0.4)
    mt.plot_polygons(polys, ax=ax, edgecolor="cyan", lw=1.2)
    ax.set_title("mask (red) and its polygon outline (cyan)")
    save(fig, "01_mask_and_polygon.png")

    # --------------------------------------------------- 2. grow: the usual need
    # "Mask this star plus a few pixels of wings."  Distance is in pixels, so
    # the answer does not depend on how big the island happens to be.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharex=True, sharey=True)
    for ax, d in zip(axes, [0, 3, 8]):
        grown = mt.grow_mask(mask, d)
        mt.plot_image(image, ax=ax )
        mt.plot_mask(grown, ax=ax, color="orange", alpha=0.35)
        mt.plot_mask_contours(grown, ax=ax, edgecolor="lime", lw=0.9)
        ax.set_title(f"grow_mask(mask, {d:+d})  ->  {grown.sum()} px")
    fig.suptitle("growing a mask by a fixed pixel distance")
    fig.tight_layout()
    save(fig, "02_grow.png")

    # ------------------------------------------------ 3. scale: multiply the shape
    # Area goes as factor**2, so scaling is a magnification, not a widening.
    # Compare with the grow panel above to see which one you actually want.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharex=True, sharey=True)
    for ax, f in zip(axes, [0.5, 1.0, 2.0]):
        scaled = mt.scale_mask(mask, f)
        mt.plot_image(image, ax=ax )
        mt.plot_mask(scaled, ax=ax, color="violet", alpha=0.35)
        mt.plot_mask_contours(scaled, ax=ax, edgecolor="cyan", lw=0.9)
        ax.set_title(f"scale_mask(mask, {f})  ->  {scaled.sum()} px")
    fig.suptitle("scaling a mask by a factor (area ~ factor^2)")
    fig.tight_layout()
    save(fig, "03_scale.png")

    # ------------------------------------------- 4. grow beats scale for apertures
    # A scale factor of 2 on a small island is a huge relative change, while on
    # a large one it barely adds margin.  Distance behaves uniformly; here both
    # are applied to the same island and the pixel counts printed.
    base = int(mask.sum())
    print(f"\noriginal mask: {base} px")
    print("  grow  ->  px      (absolute margin; area ~ d * perimeter)")
    for d in (1, 2, 4, 8):
        print(f"   {d:+d}    ->  {int(mt.grow_mask(mask, d).sum()):6d}")
    print("  scale ->  px      (magnification; area ~ factor^2)")
    for f in (1.2, 1.5, 2.0, 3.0):
        print(f"   {f:.1f}   ->  {int(mt.scale_mask(mask, f).sum()):6d}")

    # ---------------------------------------------- 5. build a mask by thresholding
    # The other direction: pick a source from the image itself, threshold it into
    # a mask, then grow it.  numpy only -- no segmentation package needed.
    y0, y1, x0, x1 = 60, 110, 440, 490  # the brightest star in this cutout
    patch = image[y0:y1, x0:x1]
    bkg = np.median(image)
    sigma = 1.4826 * np.median(np.abs(image - bkg))
    thresh = bkg + 5 * sigma
    star = np.zeros_like(image, dtype=bool)
    star[y0:y1, x0:x1] = patch > thresh
    print(f"\nthreshold {thresh:.1f} e- (5 sigma, sigma={sigma:.2f}) -> {star.sum()} px")

    star_grown = mt.grow_mask(star, 4)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharex=True, sharey=True)
    for ax, (m, t) in zip(
        axes, [(star, "thresholded at 5 sigma"), (star_grown, "then grow_mask(+4 px)")]
    ):
        mt.plot_image(image, ax=ax )
        mt.plot_mask(m, ax=ax, color="red", alpha=0.4)
        ax.set_xlim(x0 - 25, x1 + 25)
        ax.set_ylim(y0 - 25, y1 + 25)
        ax.set_title(f"{t}  ->  {m.sum()} px")
    fig.tight_layout()
    save(fig, "04_threshold_then_grow.png")

    # ----------------------------------------------- 6. round trip is lossless
    back = mt.polygons_to_mask(mt.mask_to_polygons(star_grown), image.shape)
    print(f"mask -> polygons -> mask identical: {np.array_equal(back, star_grown)}")

    # ------------------------------------------- 7. holes survive the round trip
    donut = np.zeros((40, 40), bool)
    donut[10:30, 10:30] = True
    donut[17:23, 17:23] = False
    dp = mt.mask_to_polygons(donut)
    print(
        f"donut: {len(dp)} polygon, {len(dp[0].interiors)} hole(s), "
        f"area {dp[0].area:.0f} px^2 (pixel count {donut.sum()})"
    )

    # ------------------------------------ 8. the half-pixel convention, made visible
    # Pixel [5,5] alone.  In "pixel" space its outline spans 4.5..5.5 and is
    # centred on the pixel; in "corner" space it spans 5..6.  Off by half a
    # pixel, which at 0.24"/px is 0.12" -- enough to miss a small aperture.
    single = np.zeros((12, 12), bool)
    single[5, 5] = True
    for space in ("pixel", "corner"):
        p = mt.mask_to_polygons(single, space=space)[0]
        print(f"  space={space:6s} bounds {p.bounds}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.8), sharex=True, sharey=True)
    for ax, space in zip(axes, ("pixel", "corner")):
        p = mt.mask_to_polygons(single, space=space)[0]
        ax.imshow(single, origin="lower", cmap="Greys_r", vmin=0, vmax=1,
                  extent=[-0.5, 11.5, -0.5, 11.5])
        mt.plot_polygons(p, ax=ax, space=space, edgecolor="red", lw=1.5)
        ax.plot([5], [5], "g+", ms=14, mew=2)
        ax.set_xlim(3, 8)
        ax.set_ylim(3, 8)
        ax.set_title(f'space="{space}"')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "05_half_pixel.png")

    # ------------------------------------- 9. why the default stretch is asymmetric
    # astropy's simple_norm(percent=99.9) clips symmetrically, so the 582 zero
    # pixels in this frame -- 0.22% of it -- pull vmin down to 0 against a sky
    # of 497 and the whole image flattens to grey.  The default here cuts
    # asymmetrically (16th to 99.9th percentile) and stays readable.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharex=True, sharey=True)
    stretches = [
        ("percent=99.9  (astropy simple_norm)", dict(percent=99.9)),
        ("default: min16 / max99.9", {}),
        ("plain min/max", dict(min_percent=None, max_percent=None)),
    ]
    for ax, (label, kw) in zip(axes, stretches):
        norm = mt.make_norm(image, "asinh", **kw)
        mt.plot_image(image, ax=ax, norm=norm)
        mt.plot_mask(mask, ax=ax, color="red", alpha=0.5)
        ax.set_title(f"{label}\nvmin={norm.vmin:.0f}  vmax={norm.vmax:.0f}", fontsize=10)
    fig.suptitle("the stretch choice decides whether you can see anything")
    fig.tight_layout()
    save(fig, "06_stretch_choice.png")

    print(f"\nwrote PNGs to {OUT}")


def base_axes(image, figsize=(7.2, 7.2)):
    """A figure with the image drawn on it, ready for mask overlays."""
    fig, ax = plt.subplots(figsize=figsize)
    mt.plot_image(image, ax=ax )
    return fig, ax


def save(fig, name):
    # dpi 85 keeps the noise-dominated full-frame panels under a megabyte
    # total while staying legible at the size the README shows them.
    fig.savefig(OUT / name, dpi=85, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


if __name__ == "__main__":
    main()
