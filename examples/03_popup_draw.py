#!/usr/bin/env python3
"""Draw a mask in a popup window, from a plain script (no notebook).

The same :class:`masktools.MaskDrawer` as the notebook, but ``show()`` blocks on
a real GUI backend, so this is usable from a terminal, an editor, or a pipeline
step that needs a human decision.

    python examples/03_popup_draw.py                 # draw on the shipped cutout
    python examples/03_popup_draw.py my_image.fits   # draw on your own frame
    python examples/03_popup_draw.py --demo          # headless self-check

Press ``h`` in the window for the full key list, ``s`` to save, ``q`` when done.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_FITS = HERE.parent / "data" / "example_cutout.fits.gz"


def pick_backend() -> str:
    """First GUI backend this machine can actually open a window with."""
    import matplotlib

    preferred = ["macosx", "TkAgg", "QtAgg", "WXAgg", "GTK3Agg"]
    for name in preferred:
        try:
            matplotlib.use(name, force=True)
        except Exception:
            continue
        if matplotlib.get_backend().lower() == name.lower():
            return matplotlib.get_backend()
    return matplotlib.get_backend()


def load(path: str | None):
    """Return (image, header).  FITS through astropy; falls back to a .npy."""
    target = Path(path) if path else DEFAULT_FITS
    # Path.suffix on "foo.fits.gz" is ".gz", so match the whole tail instead.
    if target.name.lower().endswith((".fits", ".fit", ".fts", ".fits.gz", ".fit.gz")):
        from astropy.io import fits

        with fits.open(target) as hdul:
            for hdu in hdul:
                if hdu.data is not None and np.ndim(hdu.data) == 2:
                    return np.asarray(hdu.data, dtype=float), hdu.header
        raise ValueError(f"{target} has no 2-D image extension")
    return np.asarray(np.load(target), dtype=float), None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?", help="FITS or .npy frame (default: shipped cutout)")
    ap.add_argument("-o", "--out", default="drawn_mask.fits", help="output mask path")
    ap.add_argument("--shape", default=None, help="blank canvas, e.g. 512,512")
    ap.add_argument("--mask", default=None, help="existing mask to refine (.npy or FITS ext)")
    ap.add_argument("--demo", action="store_true", help="run headless and exit")
    args = ap.parse_args(argv)

    import masktools as mt

    header = None
    if args.shape:
        shape = tuple(int(v) for v in args.shape.split(","))
        image = None
        print(f"drawing on a blank {shape} canvas")
    else:
        image, header = load(args.image)
        shape = image.shape
        print(f"loaded {image.shape} frame" + (f" from {args.image}" if args.image else ""))

    ref = None
    if args.mask:
        ref, _ = load(args.mask)
        ref = ref.astype(bool)
        print(f"refining an existing mask: {ref.sum()} pixels")

    if args.demo:
        # Headless check: no window, but the whole code path still runs.
        import matplotlib

        matplotlib.use("Agg", force=True)
        drawer = mt.MaskDrawer(data=image, mask=ref, shape=None if image is not None else shape,
                               verbose=False)
        drawer.add_polygon([(shape[0] // 4, shape[1] // 4),
                            (shape[0] // 2, shape[1] // 4),
                            (shape[0] // 2, shape[1] // 2)])
        out = drawer.mask()
        print(f"[demo] backend={matplotlib.get_backend()}  polygons={len(drawer.vertices)}"
              f"  mask={out.sum()} px of {out.shape}")
        assert out.sum() > 0 and out.dtype == np.bool_
        drawer.teardown()
        print("[demo] ok -- run without --demo to open a window")
        return 0

    backend = pick_backend()
    print(f"matplotlib backend: {backend}")
    if backend.lower() in ("agg", "module://matplotlib.backends.backend_agg"):
        print("no GUI backend available; try --demo instead", file=sys.stderr)
        return 1

    drawer = mt.MaskDrawer(data=image, mask=ref, shape=None if image is not None else shape,
                           savepath=args.out)
    print("left click: vertex | n: finish polygon | e: re-edit | d: delete")
    print("s: save    | q or c: finish and close | h: this list again")
    drawer.show()  # blocks until q / window close

    mask = drawer.mask()
    if not mask.any():
        print("nothing was drawn; no file written")
        return 1
    path = drawer.save(header=header)  # the frame's WCS travels with the mask
    print(f"{mask.sum()} pixels -> {path}")
    drawer.teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
