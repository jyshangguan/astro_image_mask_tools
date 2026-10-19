"""Convert between boolean pixel masks and vector polygons."""

from __future__ import annotations

import numpy as np
from rasterio.features import rasterize, shapes
from shapely.geometry import Polygon
from shapely.geometry import shape as _to_shape

from ._geometry import as_polygon_list, check_space, iter_polygons, polygon_to_space, repair

__all__ = ["mask_to_polygons", "polygons_to_mask", "mask_area"]


def _as_binary(mask) -> np.ndarray:
    """Coerce *mask* to a C-contiguous boolean 2-D array."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {arr.shape}")
    # Any nonzero / True pixel is masked, which also makes a labeled or int
    # array behave sensibly instead of raising.
    return np.ascontiguousarray(arr != 0)


def mask_to_polygons(
    mask,
    connectivity: int = 8,
    space: str = "pixel",
    min_area: float = 0.0,
    fill_holes: bool = False,
) -> list[Polygon]:
    """Vectorise the masked region of *mask* into polygons.

    Each connected group of masked pixels becomes one :class:`shapely.Polygon`
    whose outline follows the pixel staircase.  Interior holes (unmasked pixels
    enclosed by masked ones) are kept as interior rings, so a donut-shaped mask
    round-trips correctly.

    Parameters
    ----------
    mask : array-like, 2-D
        The mask.  Any nonzero or ``True`` pixel counts as masked, so ``bool``,
        ``int`` and ``uint8`` masks all work.  A labeled segmentation map is
        flattened: every label becomes part of one boolean region.
    connectivity : {4, 8}
        How pixels are grouped into a region.  ``8`` (default) joins pixels
        touching at edges *or* corners; ``4`` joins edge-touching pixels only.
    space : {'pixel', 'corner'}
        Coordinate convention of the returned geometries; see
        :mod:`masktools._geometry`.  ``"pixel"`` (default) puts pixel centres on
        integers, so the polygons overlay ``imshow`` directly and feed straight
        into :meth:`astropy.wcs.WCS.pixel_to_world`.
    min_area : float
        Drop polygons whose area is below this many square pixels.  Useful for
        removing single-pixel specks that a threshold produces.  ``0`` keeps
        everything.
    fill_holes : bool
        If `True`, discard interior rings and return only the outer shell of
        each region.

    Returns
    -------
    list of shapely.geometry.Polygon
        One polygon per connected masked region, ordered largest area first.
        Empty if nothing is masked.  A region whose staircase outline
        self-intersects is repaired and, if the repair splits it, returned as
        several polygons -- never as a `MultiPolygon`, so callers can rely on
        ``.exterior`` and ``.interiors`` existing.

    Examples
    --------
    >>> import numpy as np
    >>> m = np.zeros((10, 10), bool)
    >>> m[3:6, 3:6] = True
    >>> polys = mask_to_polygons(m)
    >>> len(polys), round(polys[0].area, 3)
    (1, 9.0)
    """
    check_space(space)
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")

    binary = _as_binary(mask)
    if not binary.any():
        return []

    # rasterio.features.shapes wants an integer (or float) raster and reports
    # the value each ring group belongs to; 0 is the unmasked background.
    records = list(shapes(binary.astype(np.uint8), connectivity=connectivity))

    polys = []
    for geojson, value in records:
        if not value:
            continue
        # Repairing an invalid ring can change its type: rasterio emits
        # self-intersecting staircases on ragged outlines, and buffer(0) turns
        # one of those into a MultiPolygon (the shipped cutout does exactly
        # this, splitting a 1071-px island into a 1063-px body plus seven
        # 1-2-px slivers that become holes in it).  Flattening here is what
        # keeps the documented "one polygon per region" contract true.
        for poly in iter_polygons(repair(_to_shape(geojson))):
            if fill_holes and poly.interiors:
                poly = Polygon(poly.exterior)
            if min_area and poly.area < min_area:
                continue
            polys.append(poly)

    # Largest first, so polys[0] is the dominant region (the prototype's
    # get_mask_polygons()[0] relied on this being true without saying so).
    polys.sort(key=lambda p: p.area, reverse=True)
    return [polygon_to_space(p, space) for p in polys]


def polygons_to_mask(
    polygons,
    mask_shape,
    space: str = "pixel",
    all_touched: bool = False,
    dtype=None,
) -> np.ndarray:
    """Rasterise *polygons* into a boolean mask of *mask_shape*.

    This is the inverse of :func:`mask_to_polygons`, and also the way to turn
    hand-drawn polygons (see :class:`masktools._draw.MaskDrawer`) into a mask.

    Parameters
    ----------
    polygons : Polygon-like or list of it
        A shapely ``Polygon``/``MultiPolygon``, a GeoJSON-like dict, an
        ``(N, 2)`` vertex array, or any list/tuple mixing those.  Anything
        :func:`masktools._geometry.as_polygon_list` accepts.
    mask_shape : tuple of int
        ``(nrow, ncol)`` of the output mask.
    space : {'pixel', 'corner'}
        Coordinate convention the input polygons are expressed in.  It must
        match how they were produced; the default is ``"pixel"``, consistent
        with :func:`mask_to_polygons`.
    all_touched : bool
        If `False` (default) a pixel is set when its *centre* falls inside a
        polygon -- the convention that makes a mask round-trip exactly.  If
        `True` a pixel is set when the polygon touches it anywhere, which
        inflates thin features by up to a pixel on each side.
    dtype : numpy dtype, optional
        Output dtype.  Defaults to ``bool``; pass ``np.int32`` if you need a
        0/1 integer mask.

    Returns
    -------
    numpy.ndarray
        The rasterised mask, 2-D, with shape *mask_shape*.

    Notes
    -----
    Pixels outside the array are clipped silently, as rasterio must be.  A
    polygon entirely outside *mask_shape* therefore contributes nothing.
    """
    check_space(space)
    mask_shape = tuple(int(s) for s in mask_shape)
    if len(mask_shape) != 2:
        raise ValueError(f"mask_shape must be (nrow, ncol), got {mask_shape}")
    dtype = np.bool_ if dtype is None else np.dtype(dtype)

    # as_polygon_list already resolves *space* into canonical corner space,
    # which is exactly what rasterize wants -- no further translation here.
    geoms = as_polygon_list(polygons, space=space)
    if not geoms:
        return np.zeros(mask_shape, dtype=dtype)

    out = rasterize(geoms, out_shape=mask_shape, fill=0, all_touched=all_touched)
    return out.astype(dtype)


def mask_area(mask, connectivity: int = 8) -> float:
    """Return the vectorised area of *mask* in square pixels.

    Unlike ``mask.sum()`` this measures the polygon outline, so it is the area
    an aperture defined by the mask actually covers.  The two agree for a
    staircase region and differ by at most the perimeter/2 sliver introduced by
    buffering.
    """
    return float(sum(p.area for p in mask_to_polygons(mask, connectivity=connectivity)))
