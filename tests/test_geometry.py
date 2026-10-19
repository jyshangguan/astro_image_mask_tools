"""Tests for the coordinate contract, the part that is easy to get wrong."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

from masktools._geometry import (
    HALF_PIXEL,
    as_polygon,
    as_polygon_list,
    check_space,
    corner_to_pixel_geom,
    corner_to_pixel,
    iter_polygons,
    pixel_to_corner,
    pixel_to_corner_geom,
    polygon_to_space,
    polys_to_space,
    repair,
)


def test_space_validation():
    assert check_space("pixel") == "pixel"
    assert check_space("corner") == "corner"
    for bad in ("pixels", "world", "sky", None, 0):
        with pytest.raises(ValueError, match="space must be"):
            check_space(bad)


def test_corner_pixel_roundtrip():
    xy = np.array([[0.0, 10.0], [5.5, 3.25]])
    assert np.allclose(corner_to_pixel(pixel_to_corner(xy)), xy)
    assert np.allclose(pixel_to_corner(corner_to_pixel(xy)), xy)
    # and the offset is exactly a half pixel
    assert np.allclose(corner_to_pixel(np.array([[0.0, 0.0]])), [[-HALF_PIXEL, -HALF_PIXEL]])


def test_single_pixel_bounds_in_each_space():
    """The canary: pixel [5,5] is [4.5,5.5] in pixel space and [5,6] in corner.

    Note the direction: polygon_to_space() takes a *corner*-space geometry as
    its input, so feeding it a pixel-space polygon is a bug, not a conversion.
    The named pair pixel_to_corner_geom / corner_to_pixel_geom is the safe way
    to move in both directions.
    """
    pixel = Polygon([(4.5, 4.5), (5.5, 4.5), (5.5, 5.5), (4.5, 5.5)])
    corner = pixel_to_corner_geom(pixel)
    assert corner.bounds == pytest.approx((5.0, 5.0, 6.0, 6.0))
    assert corner_to_pixel_geom(corner).equals(pixel)
    # polygon_to_space dispatches identically, from corner space
    assert polygon_to_space(corner, "pixel").equals(pixel)
    assert polygon_to_space(corner, "corner") is corner


def test_polygon_to_space_preserves_holes():
    donut = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(4, 4), (6, 4), (6, 6), (4, 6)]])
    out = polygon_to_space(donut, "pixel")
    assert len(out.interiors) == 1
    # both rings shift by the same amount, so the hole stays concentric
    assert out.area == pytest.approx(donut.area)
    assert out.bounds == pytest.approx(tuple(b - HALF_PIXEL for b in donut.bounds))


def test_polygon_to_space_corner_is_noop():
    p = Polygon([(0, 0), (1, 0), (1, 1)])
    assert polygon_to_space(p, "corner") is p


def test_polygon_to_space_passes_through_none_and_empty():
    assert polygon_to_space(None, "pixel") is None
    empty = Polygon()
    assert polygon_to_space(empty, "pixel") is empty


@pytest.mark.parametrize(
    "obj",
    [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        {"type": "Polygon", "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]},
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
    ],
)
def test_as_polygon_accepts_three_forms(obj):
    p = as_polygon(obj)
    assert p.area == pytest.approx(1.0)


def test_as_polygon_rejects_junk():
    with pytest.raises(TypeError, match="cannot interpret"):
        as_polygon("not a polygon")
    with pytest.raises(TypeError, match="cannot interpret"):
        as_polygon(np.zeros((3, 3)))


def test_as_polygon_list_normalises_to_corner_space():
    """Input authored in pixel space must come back shifted by +0.5, once."""
    px = Polygon([(4.5, 4.5), (5.5, 4.5), (5.5, 5.5), (4.5, 5.5)])
    (out,) = as_polygon_list(px, space="pixel")
    assert out.bounds == pytest.approx((5.0, 5.0, 6.0, 6.0))
    # already-corner input is left alone
    (same,) = as_polygon_list(out, space="corner")
    assert same.bounds == pytest.approx((5.0, 5.0, 6.0, 6.0))


def test_as_polygon_list_roundtrip_via_polys_to_space():
    px = Polygon([(4.5, 4.5), (5.5, 4.5), (5.5, 5.5), (4.5, 5.5)])
    canon = as_polygon_list(px, space="pixel")
    assert polys_to_space(canon, "pixel")[0].equals(px)
    assert polys_to_space(canon, "corner") == canon


def test_as_polygon_list_flattens_multipolygon():
    mp = MultiPolygon(
        [Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(5, 5), (6, 5), (6, 6)])]
    )
    assert len(as_polygon_list(mp, space="corner")) == 2


def test_as_polygon_list_none_and_empty():
    assert as_polygon_list(None) == []
    assert as_polygon_list([]) == []
    assert as_polygon_list(Polygon()) == []


def test_iter_polygons_drops_zero_area_results():
    """A shrink can leave a LineString/Point; those cannot be rasterised."""
    from shapely import buffer

    tiny = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    collapsed = buffer(tiny, -2)  # empty polygon
    assert list(iter_polygons(collapsed)) == []
    line = tiny.intersection(Polygon([(0, 0), (1, 0), (1, 0.0001), (0, 0.0001)]))
    assert list(iter_polygons(Point(0, 0))) == []
    assert list(iter_polygons(None)) == []


def test_iter_polygons_keeps_real_polygons():
    mp = MultiPolygon(
        [Polygon([(0, 0), (2, 0), (2, 2)]), Polygon([(5, 5), (6, 5), (6, 6)])]
    )
    polys = list(iter_polygons(mp))
    assert len(polys) == 2
    assert all(isinstance(p, Polygon) for p in polys)


def test_repair_fixes_self_intersection():
    """buffer(0) on a figure-eight keeps the positively-wound lobe only.

    The ring (0,0)->(2,2)->(2,0)->(0,2) self-crosses at (1,1) into two lobes of
    area 1 with *opposite* orientation, which is why the raw signed area is 0.
    So the repaired area is 1.0, not the naive 2.0.
    """
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    assert not bowtie.is_valid
    assert bowtie.area == pytest.approx(0.0)
    fixed = repair(bowtie)
    assert fixed.is_valid
    assert fixed.area == pytest.approx(1.0)
    assert repair(fixed) is fixed  # idempotent: no copy when already valid


def test_repair_leaves_valid_geometry_untouched():
    p = Polygon([(0, 0), (1, 0), (1, 1)])
    assert repair(p) is p  # no copy when nothing to fix
    assert repair(None) is None
    empty = Polygon()
    assert repair(empty) is empty
