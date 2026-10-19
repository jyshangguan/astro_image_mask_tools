"""Coordinate conventions and geometry plumbing.

This module owns the *only* ambiguous thing about mask vectorisation: where a
pixel's coordinates point.  Everything else in the package is derived from the
two conventions defined here, and the conversion happens in exactly one place.

Two conventions
---------------

``"corner"`` -- raster / GDAL convention
    Pixel ``[row, col]`` occupies the unit square
    ``[col, col+1) x [row, row+1)``.  Integer coordinates land on pixel
    *corners*, so a pixel's centre is at ``(col+0.5, row+0.5)``.
    :func:`rasterio.features.shapes` returns geometries in this space, and so
    does GDAL/OGR, GeoJSON written by ``rio``, etc.

``"pixel"`` -- array / matplotlib / astropy convention
    Pixel ``[row, col]`` has its *centre* at the integer coordinate
    ``(col, row)``.  This is what ``imshow(..., origin='lower')`` puts on the
    screen and what :meth:`astropy.wcs.WCS.pixel_to_world` expects (0-based).

Why it matters
--------------
A mask of a single pixel at ``[5, 5]`` is the square ``[5,6) x [5,6)`` in
``"corner"`` space and the square ``[4.5,5.5) x [4.5,5.5)`` in ``"pixel"``
space.  Mixing the two shifts every contour by half a pixel -- about 0.03" on
a 0.24"/px image, which is enough to silently mis-register a small aperture.
The public functions in this package therefore default to ``space="pixel"``
(plot and WCS ready) and let you ask for ``"corner"`` when you need to hand
geometries to rasterio.

Use :func:`to_space` / :func:`polygon_to_space` rather than sprinkling ``0.5``
through user code.
"""

from __future__ import annotations

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, base, shape
from shapely.geometry.collection import GeometryCollection

__all__ = [
    "HALF_PIXEL",
    "SPACES",
    "check_space",
    "corner_to_pixel",
    "pixel_to_corner",
    "to_space",
    "polygon_to_space",
    "pixel_to_corner_geom",
    "corner_to_pixel_geom",
    "polys_to_space",
    "iter_polygons",
    "as_polygon",
    "as_polygon_list",
    "coords_of",
    "repair",
]

#: Offset between the two conventions of this module.
HALF_PIXEL = 0.5

#: Accepted values of the ``space`` keyword.
SPACES = ("pixel", "corner")


def check_space(space: str) -> str:
    """Validate a ``space`` keyword, returning it unchanged.

    Raises
    ------
    ValueError
        If *space* is not one of :data:`SPACES`.
    """
    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES}, got {space!r}")
    return space


def corner_to_pixel(coords, /):
    """Shift ``"corner"`` coordinates into ``"pixel"`` space (``-0.5``)."""
    return np.asarray(coords, dtype=float) - HALF_PIXEL


def pixel_to_corner(coords, /):
    """Shift ``"pixel"`` coordinates into ``"corner"`` space (``+0.5``)."""
    return np.asarray(coords, dtype=float) + HALF_PIXEL


def to_space(coords, space: str):
    """Bring *coords* (already in ``"corner"`` space) into *space*."""
    check_space(space)
    coords = np.asarray(coords, dtype=float)
    return coords if space == "corner" else coords - HALF_PIXEL


def polygon_to_space(geom, space: str):
    """Move a geometry that is in ``"corner"`` space into *space*.

    Read the name as "this corner-space geometry, into which space?".  It is a
    one-way dispatcher: *corner* is the input convention, always.  Use
    :func:`pixel_to_corner_geom` to go back.

    Interior rings (holes) are preserved, because the translation is an affine
    shift applied to the whole geometry rather than to its shell alone.
    """
    check_space(space)
    if space == "corner" or geom is None or geom.is_empty:
        return geom
    return affinity.translate(geom, xoff=-HALF_PIXEL, yoff=-HALF_PIXEL)


def pixel_to_corner_geom(geom):
    """Move a geometry from ``"pixel"`` space into ``"corner"`` (``+0.5``).

    Inverse of :func:`corner_to_pixel_geom`.
    """
    if geom is None or geom.is_empty:
        return geom
    return affinity.translate(geom, xoff=HALF_PIXEL, yoff=HALF_PIXEL)


def corner_to_pixel_geom(geom):
    """Move a geometry from ``"corner"`` space into ``"pixel"`` (``-0.5``).

    Same as ``polygon_to_space(geom, "pixel")``; provided so the two
    directions are named as a symmetric pair.
    """
    if geom is None or geom.is_empty:
        return geom
    return affinity.translate(geom, xoff=-HALF_PIXEL, yoff=-HALF_PIXEL)


def iter_polygons(geom):
    """Yield every :class:`~shapely.geometry.Polygon` inside *geom*.

    Buffers and other operations happily return ``MultiPolygon`` or even
    ``GeometryCollection`` (a shrink can leave a mixture of both).  Flattening
    through this generator keeps the rest of the package polygon-only.
    Empty members and non-polygon members are dropped.
    """
    if geom is None:
        return
    if isinstance(geom, Polygon):
        if not geom.is_empty:
            yield geom
        return
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        for sub in geom.geoms:
            yield from iter_polygons(sub)
        return
    # Zero-area results (a shrink that collapsed a region to a line or a point)
    # cannot be rasterised, so they are dropped here rather than in every caller.
    if getattr(geom, "geom_type", None) in (
        "LinearRing",
        "LineString",
        "MultiLineString",
        "Point",
        "MultiPoint",
    ):
        return
    # Last resort: anything iterable that quacks like a collection.
    geoms = getattr(geom, "geoms", None)
    if geoms is not None:
        for sub in geoms:
            yield from iter_polygons(sub)


def repair(geom):
    """Return *geom* with self-intersections resolved, when it has any.

    A negative buffer on a jagged staircase outline routinely produces
    self-intersecting results.  ``buffer(0)`` is the standard, topology
    preserving fix; it is skipped when the geometry is already valid so
    well-behaved input costs nothing.
    """
    if geom is None or geom.is_empty:
        return geom
    if geom.is_valid:
        return geom
    fixed = geom.buffer(0)
    return fixed if fixed is not None else geom


def as_polygon(obj):
    """Coerce *obj* -- a shapely geometry or a GeoJSON-like mapping -- to a Polygon.

    Accepting the GeoJSON dict form keeps interoperability with
    :func:`rasterio.features.shapes` output and with hand-written polygons
    (``{"type": "Polygon", "coordinates": [...]}``), which is what the
    prototype in ``pixsed`` passed around.
    """
    if isinstance(obj, Polygon):
        return obj
    if isinstance(obj, (MultiPolygon, GeometryCollection)):
        polys = list(iter_polygons(obj))
        if not polys:
            return Polygon()
        return polys[0] if len(polys) == 1 else MultiPolygon(polys)
    if isinstance(obj, dict) or hasattr(obj, "keys"):
        return shape(obj)
    if hasattr(obj, "__geo_interface__"):
        return shape(obj.__geo_interface__)
    # Fall back on treating it as a sequence of (x, y) vertices.
    try:
        arr = np.asarray(obj, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "cannot interpret object of type "
            f"{type(obj).__name__} as a polygon; pass a shapely Polygon, a "
            "GeoJSON-like dict, or an (N, 2) array of vertices"
        ) from exc
    if arr.ndim != 2 or arr.shape[1] != 2:
        hint = ""
        if arr.ndim == 2 and arr.shape[1] != 2:
            hint = (
                " (a 2-D array that is not (N, 2) looks like a pixel mask -- "
                "run it through mask_to_polygons() first to get polygons)"
            )
        raise TypeError(
            "cannot interpret object of type "
            f"{type(obj).__name__} as a polygon; pass a shapely Polygon, a "
            "GeoJSON-like dict, or an (N, 2) array of vertices" + hint
        )
    return Polygon(arr)


def as_polygon_list(obj, space: str = "pixel") -> list:
    """Normalise *obj* to a ``list[Polygon]`` in **canonical (``"corner"``) space**.

    This is the single place where user-supplied polygons enter the package, so
    it is also the single place where the *space* they were authored in is
    converted away.  Internally everything downstream -- rasterisation in
    particular -- works in ``"corner"`` space, which is what
    :func:`rasterio.features.shapes` emits, so the round trip costs exactly one
    translation in and one out.

    Accepts a lone Polygon, a MultiPolygon, a GeoJSON-like dict, an ``(N, 2)``
    vertex array, or any list/tuple mixing those.

    Parameters
    ----------
    obj : Polygon-like or list of it
    space : {'pixel', 'corner'}
        The convention *obj* is expressed in.  ``"pixel"`` (the default, and
        what :func:`masktools.vectorize.mask_to_polygons` returns) shifts by
        ``+0.5`` to reach canonical space; ``"corner"`` is a no-op.
    """
    check_space(space)
    if obj is None:
        return []
    if isinstance(obj, (list, tuple, set)) or (
        hasattr(obj, "geoms") and not isinstance(obj, Polygon)
    ):
        items = list(obj) if not hasattr(obj, "geoms") else list(obj.geoms)
    else:
        items = [obj]

    out = []
    for item in items:
        poly = as_polygon(item)
        for sub in iter_polygons(repair(poly)):
            out.append(pixel_to_corner_geom(sub) if space == "pixel" else sub)
    return out


def polys_to_space(polys, space: str = "pixel") -> list:
    """Convert canonical (``"corner"``) polygons out to *space*."""
    check_space(space)
    if space == "corner":
        return list(polys)
    return [polygon_to_space(p, space) for p in polys]


def coords_of(poly, space: str = "pixel", exterior_only: bool = True):
    """Return the ``(N, 2)`` vertex array of *poly*, given in *space*.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        Geometry currently expressed in ``"corner"`` space.
    space : {'pixel', 'corner'}
        Which convention to label the output with.
    exterior_only : bool
        If `False`, interior rings are appended, separated by rows of ``nan``
        so the result can be plotted in one ``plot`` call.
    """
    if exterior_only:
        return to_space(np.asarray(poly.exterior.coords, dtype=float), space)

    xs, ys = [], []
    for ring in [poly.exterior, *poly.interiors]:
        c = to_space(np.asarray(ring.coords, dtype=float), space)
        if xs:
            xs.append([np.nan])
            ys.append([np.nan])
        xs.append(c[:, 0])
        ys.append(c[:, 1])
    return np.column_stack([np.concatenate(xs), np.concatenate(ys)])
