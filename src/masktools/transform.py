"""Scale and grow masks, via their polygon outlines.

Both operations vectorise the mask, transform the outlines, and rasterise back.
Doing it in the vector domain (rather than by convolution or by resampling the
array) is what gives sub-pixel control: growing by 2.5 pixels is meaningful
here, whereas :func:`scipy.ndimage.binary_dilation` can only grow by whole
structuring elements.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from shapely import affinity, buffer
from shapely.geometry import Point, Polygon

from ._geometry import (
    HALF_PIXEL,
    as_polygon_list,
    check_space,
    iter_polygons,
    polys_to_space,
    repair,
)
from .vectorize import mask_to_polygons, polygons_to_mask

__all__ = ["scale_polygons", "grow_polygons", "scale_mask", "grow_mask"]

_JOIN_STYLES = ("round", "mitre", "bevel")


def _resolve_origin(origin, space: str = "pixel"):
    """Turn an ``origin`` keyword into the argument shapely's affinity wants.

    An explicit point is given in *space* (like every other coordinate the user
    touches) and is translated into canonical corner space here, so a pivot of
    ``(30, 30)`` means the same pixel whether the polygons are in ``"pixel"`` or
    ``"corner"`` space.
    """
    if isinstance(origin, str):
        if origin in ("centroid", "center"):
            return origin
        raise ValueError(
            "origin must be 'centroid', 'center', or an (x, y) point; "
            f"got {origin!r}"
        )
    if isinstance(origin, Point):
        origin = (origin.x, origin.y)
    arr = np.asarray(origin, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"origin point must have exactly 2 values, got {arr.shape}")
    if space == "pixel":
        arr = arr + HALF_PIXEL
    # Must be a plain tuple: shapely's interpret_origin starts with
    # `if origin == "center"`, and comparing an ndarray to a str raises
    # "truth value of an array is ambiguous" instead of returning False.
    return (float(arr[0]), float(arr[1]))


def scale_polygons(
    polygons,
    factor: float,
    origin: str | Sequence[float] | Point = "centroid",
    xfactor: float | None = None,
    yfactor: float | None = None,
    space: str = "pixel",
) -> list[Polygon]:
    """Scale polygon outlines by a multiplicative ``factor``.

    Area scales as ``factor ** 2``, so this is a *magnification* of the shape,
    not a uniform widening.  If you want "make the aperture 3 pixels bigger"
    you almost certainly want :func:`grow_polygons` instead.

    Parameters
    ----------
    polygons : Polygon-like or list of it
        See :func:`masktools.vectorize.polygons_to_mask`.
    factor : float
        Must be positive.  ``> 1`` enlarges, ``< 1`` shrinks.
    origin : {'centroid', 'center'} or (x, y) or `~shapely.geometry.Point`
        Pivot of the scaling.  ``"centroid"`` (default) scales each polygon
        about its own area centroid, which keeps a compact region in place even
        if it has a ragged outline or a stray attached pixel.  ``"center"``
        uses each polygon's bounding-box centre instead -- cheaper, but one
        outlier pixel drags the pivot and can visibly shear the result.
        Passing an explicit point applies that *same* pivot to every polygon,
        so relative positions between separate regions are preserved (use
        ``image_shape``-derived coordinates to zoom about the frame centre).
    xfactor, yfactor : float, optional
        If given, override *factor* along each axis for anisotropic scaling.
    space : {'pixel', 'corner'}
        Coordinate convention of the input and output polygons.

    Returns
    -------
    list of shapely.geometry.Polygon
        Scaled polygons, in *space*.
    """
    check_space(space)
    # Canonical corner space throughout: scaling about a centroid or a bbox
    # centre is translation-invariant, so doing the maths there and converting
    # once at the end is identical to working in *space* directly.
    polys = as_polygon_list(polygons, space=space)

    xf = factor if xfactor is None else xfactor
    yf = factor if yfactor is None else yfactor
    for name, val in (("factor", factor), ("xfactor", xf), ("yfactor", yf)):
        if not np.isfinite(val) or val <= 0:
            raise ValueError(f"{name} must be a positive finite number, got {val!r}")

    global_origin = None if isinstance(origin, str) else _resolve_origin(origin, space)

    out = []
    for poly in polys:
        if global_origin is not None:
            origin_arg = global_origin
        else:
            # A zero-area polygon has no centroid; fall back on its bbox centre.
            keyword = "center" if poly.area <= 0 else origin
            origin_arg = _resolve_origin(keyword, space)
        scaled = affinity.scale(poly, xfact=xf, yfact=yf, origin=origin_arg)
        out.extend(iter_polygons(repair(scaled)))
    return polys_to_space(out, space)


def grow_polygons(
    polygons,
    distance: float,
    join_style: str = "round",
    quad_segs: int = 8,
    min_area: float = 0.0,
    space: str = "pixel",
) -> list[Polygon]:
    """Widen (or narrow) polygon outlines by a fixed ``distance`` in pixels.

    This is a geometric buffer: every point of the outline moves outward along
    its normal by *distance*, so a compact region grows by roughly
    ``distance * perimeter`` in area regardless of its size.  Unlike
    :func:`scale_polygons` the growth is absolute, which is what you want when
    the goal is "clear the neighbouring star by 4 pixels".

    Parameters
    ----------
    polygons : Polygon-like or list of it
        See :func:`masktools.vectorize.polygons_to_mask`.
    distance : float
        Pixels to add.  Negative shrinks.  Shrinking a region thinner than
        ``2 * |distance|`` deletes it, and the polygon simply disappears from
        the output rather than becoming degenerate.
    join_style : {'round', 'mitre', 'bevel'}
        How corners are treated.  ``"round"`` (default) rounds the staircase
        corners into arcs, which is usually what you want for an aperture.
        ``"mitre"`` keeps sharp corners and so preserves a boxy outline.
    quad_segs : int
        Number of segments per quarter circle for ``join_style="round"``.
        Higher is smoother; 8 is plenty.
    min_area : float
        Drop results whose area is below this many square pixels, to remove
        slivers left behind by a shrink.
    space : {'pixel', 'corner'}
        Coordinate convention of the input and output polygons.

    Returns
    -------
    list of shapely.geometry.Polygon
        Buffered polygons, in *space*.  Interior rings are kept, so a hole in
        the mask closes up as the region grows and opens further as it shrinks.
    """
    check_space(space)
    if join_style not in _JOIN_STYLES:
        raise ValueError(
            f"join_style must be one of {_JOIN_STYLES}, got {join_style!r}"
        )
    if not np.isfinite(distance):
        raise ValueError(f"distance must be finite, got {distance!r}")
    quad_segs = int(quad_segs)
    if quad_segs < 1:
        raise ValueError(f"quad_segs must be >= 1, got {quad_segs}")

    polys = as_polygon_list(polygons, space=space)
    out = []
    for poly in polys:
        grown = buffer(
            poly, float(distance), join_style=join_style, quad_segs=quad_segs
        )
        for sub in iter_polygons(repair(grown)):
            if min_area and sub.area < min_area:
                continue
            out.append(sub)
    return polys_to_space(out, space)


def _warn_if_clipped(polys, shape_out):
    """Warn when the transformed outline falls partly outside the output grid.

    Rasterisation can only set pixels that exist, so any part of a polygon
    beyond the frame is dropped silently otherwise.  Note that *out_shape*
    extends the grid toward higher row/column only: growth into negative
    coordinates needs the input to be padded first.
    """
    if not polys:
        return
    x0 = min(p.bounds[0] for p in polys)
    y0 = min(p.bounds[1] for p in polys)
    x1 = max(p.bounds[2] for p in polys)
    y1 = max(p.bounds[3] for p in polys)
    nrow, ncol = shape_out
    if x0 < -0.5 or y0 < -0.5 or x1 > ncol - 0.5 or y1 > nrow - 0.5:
        warnings.warn(
            "the transformed mask extends outside the "
            f"{nrow}x{ncol} output grid (x: {x0:.1f}..{x1:.1f}, "
            f"y: {y0:.1f}..{y1:.1f}), so those pixels are dropped. "
            "Enlarge out_shape to grow toward higher row/column, or pad the "
            "input mask (numpy.pad) when the region touches the low/left edge.",
            stacklevel=3,
        )


def _transform_mask(
    mask,
    func,
    connectivity,
    out_shape,
    all_touched,
    dtype,
    **kwargs,
):
    """Shared body of :func:`scale_mask` and :func:`grow_mask`."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")

    shape_out = tuple(out_shape) if out_shape is not None else tuple(mask.shape)
    if not np.any(mask):
        # Nothing to do, but still honour an explicit out_shape and dtype so
        # the return type never depends on the input being non-empty.
        return np.zeros(shape_out, dtype=np.dtype(bool if dtype is None else dtype))

    polys = mask_to_polygons(mask, connectivity=connectivity, space="pixel")
    new_polys = func(polys, space="pixel", **kwargs)
    _warn_if_clipped(new_polys, shape_out)
    return polygons_to_mask(
        new_polys, shape_out, space="pixel", all_touched=all_touched, dtype=dtype
    )


def scale_mask(
    mask,
    factor: float,
    connectivity: int = 8,
    origin: str | Sequence[float] | Point = "centroid",
    xfactor: float | None = None,
    yfactor: float | None = None,
    out_shape=None,
    all_touched: bool = False,
    dtype=None,
) -> np.ndarray:
    """Scale a mask by a multiplicative ``factor``.

    Convenience wrapper: :func:`~masktools.vectorize.mask_to_polygons` ->
    :func:`scale_polygons` -> :func:`~masktools.vectorize.polygons_to_mask`.
    See :func:`scale_polygons` for the meaning of *factor* and *origin*, and
    :func:`grow_mask` for the usually-preferred absolute alternative.

    Parameters
    ----------
    mask : array-like, 2-D
        Input mask; nonzero is masked.
    factor, origin, xfactor, yfactor
        Passed to :func:`scale_polygons`.
    connectivity : {4, 8}
        Pixel grouping of the input mask.
    out_shape : (int, int), optional
        Shape of the returned mask.  Defaults to *mask*'s shape.  Coordinates
        are untouched, so the result stays aligned with the input's lower-left
        corner and the extra room appears only at the *upper-right*.  A region
        growing past the low/left edge needs the input padded with
        :func:`numpy.pad` instead; a warning is raised when anything is
        dropped.
    all_touched : bool
        Rasterisation rule; see
        :func:`~masktools.vectorize.polygons_to_mask`.
    dtype : numpy dtype, optional
        Output dtype; defaults to ``bool``.

    Returns
    -------
    numpy.ndarray
        The transformed mask.

    Notes
    -----
    Scaling about each polygon's own centroid is the default and is *not* the
    same as scaling about the bounding-box centre, which is what
    ``shapely.affinity.scale`` does if you leave its ``origin`` alone.  For a
    ragged region the two pivots can differ by several pixels.

    The vector stage is exact: the polygon centroid is preserved to machine
    precision and the area scales as ``factor ** 2``.  The final rasterisation
    is quantised to the pixel grid, so when the scaled width happens to be an
    even number of pixels centred on an integer pixel *centre*, the mean of the
    masked pixel indices can sit up to half a pixel off the vector centroid
    (GDAL treats a cell as half-open, taking the lower/left edge centre and
    dropping the upper/right one).  The pixel *count* is unaffected.  Pass an
    explicit ``origin`` if you need the rasterised centre on a specific pixel.
    """
    return _transform_mask(
        mask,
        scale_polygons,
        connectivity,
        out_shape,
        all_touched,
        dtype,
        factor=factor,
        origin=origin,
        xfactor=xfactor,
        yfactor=yfactor,
    )


def grow_mask(
    mask,
    distance: float,
    connectivity: int = 8,
    join_style: str = "round",
    quad_segs: int = 8,
    min_area: float = 0.0,
    out_shape=None,
    all_touched: bool = False,
    dtype=None,
) -> np.ndarray:
    """Grow a mask by ``distance`` pixels (negative to shrink).

    Convenience wrapper: :func:`~masktools.vectorize.mask_to_polygons` ->
    :func:`grow_polygons` -> :func:`~masktools.vectorize.polygons_to_mask`.

    Parameters
    ----------
    mask : array-like, 2-D
        Input mask; nonzero is masked.
    distance : float
        Pixels to add; negative shrinks the region.
    connectivity : {4, 8}
        Pixel grouping of the input mask.
    join_style, quad_segs, min_area
        Passed to :func:`grow_polygons`.
    out_shape : (int, int), optional
        Shape of the returned mask; defaults to *mask*'s shape.  Extends the
        grid toward higher row/column only -- see :func:`scale_mask`.
    all_touched : bool
        Rasterisation rule; see
        :func:`~masktools.vectorize.polygons_to_mask`.
    dtype : numpy dtype, optional
        Output dtype; defaults to ``bool``.

    Returns
    -------
    numpy.ndarray
        The grown mask.

    Notes
    -----
    Three consequences of working in the vector domain, all measured rather than
    assumed:

    * **Rasterisation is quantised, by up to half the perimeter.**  ``rasterize``
      counts a pixel when its *centre* falls inside, so the pixel count differs
      from the polygon area by at most about ``perimeter / 2`` -- a bound on the
      *outline length*, not on the area, which is why the discrepancy is
      proportionally worst on small or ragged regions.  The sign is not fixed:
      it flips as *distance* sweeps past a ring of pixel centres, and the worst
      case sits just below a half-integer (at ``distance=0.49`` on a 100x100 box
      the vector area exceeds the pixel count by 197 of its ~200-pixel bound).
      ``join_style`` does not rescue this.  ``"mitre"`` is *exactly* the vector
      area at multiples of 0.5 on a **solid, isolated** rectilinear outline --
      and only there: a surviving interior ring inverts the sign (its boundary
      lands on excluded pixel centres, so the raster comes out *larger*), and
      two regions whose buffers overlap double-count the overlap because
      :func:`grow_polygons` treats each region alone while the mask is a union.
      Call ``shapely.unary_union`` on the result before summing areas if you
      need that comparison to mean anything.
    * **A distance below half a pixel changes nothing.**  Pixel centres are 1
      apart, so ``distance=0.49`` leaves a 100x100 box at exactly 10000 pixels
      while ``distance=0.5`` jumps it to 10201 -- a whole extra ring.  Use
      ``distance >= 1`` for a guaranteed ring, or ``all_touched=True`` to be
      inclusive.
    * **Grow then shrink is a smoothing, not an inverse.**  ``buffer(+d)`` then
      ``buffer(-d)`` is a morphological opening: it rounds off convex corners
      and cannot restore them.  On a 16-px staircase with ``d=2.5`` it keeps
      only ~40% of the original pixels.  Whole-pixel *d* on an axis-aligned box
      *does* round-trip exactly, but a jagged outline does not even then -- a
      582-px staircase island comes back as 593 pixels after ``+3, -3``.  Never
      grow a mask "temporarily" and expect it back; keep the original array.

    This is not equivalent to :func:`scipy.ndimage.binary_dilation`, which
    dilates by a structuring element on the pixel grid and therefore preserves
    square corners and adds exactly the elements' footprint.

    Examples
    --------
    >>> import numpy as np
    >>> m = np.zeros((20, 20), bool)
    >>> m[8:12, 8:12] = True
    >>> grown = grow_mask(m, 2)
    >>> int(grown.sum()) > int(m.sum())
    True
    >>> np.array_equal(grow_mask(grown, -2) & m, m)  # shrink undoes grow
    True
    """
    return _transform_mask(
        mask,
        grow_polygons,
        connectivity,
        out_shape,
        all_touched,
        dtype,
        distance=distance,
        join_style=join_style,
        quad_segs=quad_segs,
        min_area=min_area,
    )
