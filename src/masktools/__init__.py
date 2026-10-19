"""masktools -- pixel masks as polygons.

Convert a boolean pixel mask to polygons, scale or grow it, and draw masks by
hand in a Jupyter notebook or a popup window.

The package is deliberately small.  Its whole job is the round trip::

    mask  --mask_to_polygons-->  polygons  --scale/grow-->  polygons
                                                        |
                                                 polygons_to_mask
                                                        v
                                                    mask

plus an interactive drawer that produces the same polygon objects a mouse does.

Coordinate convention
---------------------
Polygons are returned in ``"pixel"`` space by default: pixel ``[row, col]`` has
its centre at ``(col, row)``, which is what ``imshow(origin='lower')`` displays
and what :meth:`astropy.wcs.WCS.pixel_to_world` consumes.  Pass ``space=
"corner"`` to get the rasterio/GDAL convention instead, where integers are pixel
corners.  See :mod:`masktools._geometry` for why the half-pixel matters.
"""

from __future__ import annotations

from ._geometry import SPACES, corner_to_pixel, pixel_to_corner
from .draw import MaskDrawer, draw_mask
from .plotting import (
    make_norm,
    plot_image,
    plot_mask,
    plot_mask_contours,
    plot_polygons,
    show_mask,
)
from .transform import grow_mask, grow_polygons, scale_mask, scale_polygons
from .vectorize import mask_area, mask_to_polygons, polygons_to_mask

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # vectorise
    "mask_to_polygons",
    "polygons_to_mask",
    "mask_area",
    # transform
    "scale_mask",
    "scale_polygons",
    "grow_mask",
    "grow_polygons",
    # interactive
    "MaskDrawer",
    "draw_mask",
    # plotting
    "plot_image",
    "plot_mask",
    "plot_polygons",
    "plot_mask_contours",
    "show_mask",
    "make_norm",
    # coordinates
    "SPACES",
    "corner_to_pixel",
    "pixel_to_corner",
]
