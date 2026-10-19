"""Shape builders and paths shared by the tests and the examples.

Kept free of pytest so scripts (and the notebooks in ``examples/``) can import
the same helpers the tests use.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

# Tests never want a GUI.  Set before matplotlib is imported anywhere.
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXAMPLE_FITS = DATA / "example_cutout.fits.gz"


def load_example():
    """The shipped 512x512 NTT/EFOSC cutout: ``(data, mask, header)``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from astropy.io import fits

    with fits.open(EXAMPLE_FITS) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        mask = np.asarray(hdul[1].data).astype(bool)
        header = hdul[0].header.copy()
    return data, mask, header


def block(shape=(40, 40), origin=(18, 18), size=(3, 3), value=True):
    """A rectangular mask, the simplest case with a known analytic answer."""
    m = np.zeros(shape, bool)
    oy, ox = origin
    sy, sx = size
    m[oy : oy + sy, ox : ox + sx] = value
    return m


def ring(outer=(10, 10), inner=(4, 4), pad=3, shape=(20, 20)):
    """A donut mask: a filled box with a concentric hole."""
    m = np.zeros(shape, bool)
    oy, ox = pad, pad
    m[oy : oy + outer[0], ox : ox + outer[1]] = True
    iy, ix = (outer[0] - inner[0]) // 2 + oy, (outer[1] - inner[1]) // 2 + ox
    m[iy : iy + inner[0], ix : ix + inner[1]] = False
    return m


def staircase(n=12):
    """A jagged, non-convex mask -- the kind a threshold really produces.

    Row *i* spans columns ``0..i``, so the outline is a diagonal staircase with
    a notch.  This is the shape where buffering and scaling expose corner and
    self-intersection behaviour that a square never would.
    """
    m = np.zeros((n, n), bool)
    for i in range(n):
        m[i, : i + 1] = True
    m[n // 2, n // 4] = False  # a one-pixel notch
    return m
