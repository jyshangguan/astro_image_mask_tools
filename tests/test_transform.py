"""Tests for scaling and growing, at both the polygon and the mask level."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from masktools._transform import grow_mask, grow_polygons, scale_mask, scale_polygons
from masktools._vectorize import mask_to_polygons, polygons_to_mask

from ._shapes import block, ring, staircase


def poly_of(m):
    return mask_to_polygons(m)[0]


# --------------------------------------------------------------- scale_polygons
def test_scale_area_scales_as_factor_squared():
    p = poly_of(block(size=(4, 4)))
    for f in (0.5, 1.0, 2.0, 3.7):
        (q,) = scale_polygons(p, f)
        assert q.area == pytest.approx(p.area * f * f)


def test_scale_about_centroid_preserves_centroid_exactly():
    """The whole point of the default origin: no drift, to machine precision."""
    p = poly_of(block(size=(7, 5)))
    c0 = np.array(p.centroid.coords[0])
    for f in (0.4, 1.0, 2.5, 8.0):
        (q,) = scale_polygons(p, f)
        assert np.abs(np.array(q.centroid.coords[0]) - c0).max() == pytest.approx(0.0)


def test_scale_about_bbox_centre_drifts_for_asymmetric_region():
    """Contrast with origin='center', which is shapely's own default.

    For a region whose centroid and bbox centre differ, 'center' moves the
    shape.  This is the bug the prototype inherited by leaving
    shapely.affinity.scale's origin alone.
    """
    m = np.zeros((40, 40), bool)
    m[10:20, 10:13] = True  # tall thin L: bbox centre y=14.5, centroid y=15.4
    m[10:13, 10:30] = True
    p = poly_of(m)
    assert not np.allclose(np.array(p.centroid.coords[0]),
                           np.array(p.representative_point().coords[0]))
    (by_centroid,) = scale_polygons(p, 2.0, origin="centroid")
    (by_center,) = scale_polygons(p, 2.0, origin="center")
    drift = np.abs(np.array(by_center.centroid.coords[0])
                   - np.array(by_centroid.centroid.coords[0])).max()
    assert drift > 0.5, "origin='center' should visibly differ for this shape"
    assert np.allclose(np.array(by_centroid.centroid.coords[0]),
                       np.array(p.centroid.coords[0]))


def test_scale_with_explicit_point_pivots_every_polygon_together():
    """A shared origin preserves the layout of separate regions."""
    m = np.zeros((60, 60), bool)
    m[5:8, 5:8] = True
    m[40:43, 40:43] = True
    polys = mask_to_polygons(m)
    pivot = (30.0, 30.0)
    out = scale_polygons(polys, 2.0, origin=pivot)
    assert len(out) == 2
    for src, dst in zip(polys, out):
        c0 = np.array(src.centroid.coords[0])
        c1 = np.array(dst.centroid.coords[0])
        # c1 - pivot == 2 * (c0 - pivot)
        assert np.allclose(c1 - np.array(pivot), 2.0 * (c0 - np.array(pivot)))


def test_scale_anisotropic_xfactor_yfactor():
    p = poly_of(block(size=(4, 6)))
    (q,) = scale_polygons(p, 1.0, xfactor=2.0, yfactor=3.0)
    assert q.area == pytest.approx(p.area * 6.0)
    w0 = p.bounds[2] - p.bounds[0]
    h0 = p.bounds[3] - p.bounds[1]
    assert (q.bounds[2] - q.bounds[0]) == pytest.approx(w0 * 2.0)
    assert (q.bounds[3] - q.bounds[1]) == pytest.approx(h0 * 3.0)


def test_scale_factor_one_is_identity_in_geometry():
    p = poly_of(staircase(12))
    (q,) = scale_polygons(p, 1.0)
    assert q.equals(p)


def test_scale_validates_factor_and_origin():
    p = poly_of(block())
    for bad in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="positive finite"):
            scale_polygons(p, bad)
    with pytest.raises(ValueError, match="origin must be"):
        scale_polygons(p, 2.0, origin="middle")
    with pytest.raises(ValueError, match="exactly 2 values"):
        scale_polygons(p, 2.0, origin=(1.0, 2.0, 3.0))


def test_scale_accepts_point_origin_and_shapely_point():
    p = poly_of(block(size=(4, 4)))
    a = scale_polygons(p, 2.0, origin=(20.0, 20.0))
    b = scale_polygons(p, 2.0, origin=Point(20.0, 20.0))
    assert np.allclose(a[0].bounds, b[0].bounds)


def test_scale_respects_space_keyword():
    m = block(size=(4, 4))
    px = scale_polygons(mask_to_polygons(m, space="pixel"), 2.0, space="pixel")
    cn = scale_polygons(mask_to_polygons(m, space="corner"), 2.0, space="corner")
    # same shape, half-pixel apart by construction
    assert px[0].area == pytest.approx(cn[0].area)
    assert np.allclose(np.array(px[0].centroid.coords[0]) + 0.5,
                       np.array(cn[0].centroid.coords[0]))


# --------------------------------------------------------------- grow_polygons
def test_grow_area_grows_monotonically():
    p = poly_of(block(size=(3, 3)))
    areas = [grow_polygons(p, d)[0].area for d in (0, 1, 2, 4)]
    assert areas == sorted(areas)
    assert areas[0] == pytest.approx(9.0)  # d=0 is a no-op


def test_grow_zero_distance_preserves_geometry():
    p = poly_of(staircase(12))
    (q,) = grow_polygons(p, 0.0)
    assert q.equals(p)


def test_grow_mitre_keeps_a_box_a_box():
    """join_style='mitre' preserves corners, so a grown square is a square."""
    p = poly_of(block(size=(4, 4)))
    (q,) = grow_polygons(p, 2.0, join_style="mitre")
    assert q.area == pytest.approx(64.0)  # (4+2*2)^2
    assert len(q.exterior.coords) <= 6


def test_grow_round_has_more_vertices_than_mitre():
    p = poly_of(block(size=(4, 4)))
    r = grow_polygons(p, 2.0, join_style="round", quad_segs=8)[0]
    m = grow_polygons(p, 2.0, join_style="mitre")[0]
    assert len(r.exterior.coords) > len(m.exterior.coords)
    # rounding cuts the corners, so round area < mitre area
    assert r.area < m.area


def test_grow_bevel_is_between():
    p = poly_of(block(size=(4, 4)))
    r = grow_polygons(p, 2.0, join_style="round")[0].area
    b = grow_polygons(p, 2.0, join_style="bevel")[0].area
    m = grow_polygons(p, 2.0, join_style="mitre")[0].area
    assert m > r > b or pytest.approx(m) > b  # bevel cuts hardest


def test_shrink_deletes_thin_regions_instead_of_degenerating():
    """Polygon level counterpart: a 1-px bar yields no polygons at -2."""
    m = np.zeros((30, 30), bool)
    m[15, 5:25] = True  # 1-px-wide bar
    p = poly_of(m)
    assert grow_polygons(p, -2.0) == []
    assert grow_polygons(p, -0.4) != []


def test_shrink_then_grow_recovers_original():
    m = block(size=(6, 6))
    grown = grow_mask(m, 3)
    assert np.array_equal(grow_mask(grown, -3), m)


def test_grow_closes_holes_but_shrink_opens_them():
    m = ring()
    (poly,) = mask_to_polygons(m)
    grown = grow_polygons(poly, 3.0)[0]
    shrunk = grow_polygons(poly, -1.0)[0]
    assert len(grown.interiors) == 0, "growing 3px must close a 4px hole"
    assert len(shrunk.interiors) == 1


def test_grow_validates_join_style_and_quad_segs():
    p = poly_of(block())
    with pytest.raises(ValueError, match="join_style must be"):
        grow_polygons(p, 1.0, join_style="sharp")
    with pytest.raises(ValueError, match="quad_segs must be"):
        grow_polygons(p, 1.0, quad_segs=0)
    with pytest.raises(ValueError, match="finite"):
        grow_polygons(p, np.nan)


def test_grow_min_area_drops_slivers():
    m = np.zeros((30, 30), bool)
    m[5:7, 5:7] = True  # 4 px
    m[15:25, 15:25] = True  # 100 px
    polys = mask_to_polygons(m)
    out = grow_polygons(polys, -1.5, min_area=1.0)
    assert len(out) < len(polys) or all(p.area >= 1.0 for p in out)


# ------------------------------------------------------------------ mask level
def test_grow_mask_adds_pixels():
    m = block(size=(3, 3))
    g1, g3 = grow_mask(m, 1), grow_mask(m, 3)
    assert g1.sum() > m.sum()
    assert g3.sum() > g1.sum()
    assert g1.sum() == 25  # 5x5 with mitre-like rounding on a small box
    assert g1.dtype == np.bool_


def test_scale_mask_pixel_count_matches_ideal():
    """Area quantisation must land on exactly factor^2 * npix for a block."""
    for size in ((3, 3), (4, 4), (5, 5), (6, 6)):
        m = block(shape=(60, 60), origin=(20, 20), size=size)
        for f in (2.0, 3.0):
            s = scale_mask(m, f)
            assert s.sum() == int(round(np.prod(size) * f * f))


def test_grow_then_scale_compose():
    m = block(size=(5, 5))
    a = grow_mask(m, 2)
    b = scale_mask(a, 1.5)
    assert b.sum() > a.sum()


def test_empty_mask_never_breaks_and_keeps_contract():
    e = np.zeros((9, 9), bool)
    for out in (grow_mask(e, 5), scale_mask(e, 5), grow_mask(e, -5)):
        assert out.shape == (9, 9)
        assert out.dtype == np.bool_
        assert out.sum() == 0
    assert scale_mask(e, 2, out_shape=(20, 20)).shape == (20, 20)
    assert grow_mask(e, 2, dtype=np.int32).dtype == np.int32


def test_out_shape_extends_only_toward_higher_indices():
    """out_shape buys room at the upper-right, not at the low/left edge.

    A region tucked into the [0,0] corner has nowhere to grow, so widening the
    grid changes nothing -- and the user gets a warning telling them to pad.
    """
    m = np.zeros((20, 20), bool)
    m[0:3, 0:3] = True
    with pytest.warns(UserWarning, match="outside the"):
        clipped = grow_mask(m, 4)
    with pytest.warns(UserWarning, match="pad"):
        roomy = grow_mask(m, 4, out_shape=(30, 30))
    assert roomy.shape == (30, 30)
    assert np.array_equal(clipped, roomy[:20, :20])
    assert roomy[20:, :].sum() == 0 and roomy[:, 20:].sum() == 0


def test_padding_is_the_documented_fix_for_low_edge_growth():
    """The workaround the warning points at, verified to actually work."""
    m = np.zeros((20, 20), bool)
    m[0:3, 0:3] = True
    pad = 6
    padded = np.pad(m, pad)
    out = grow_mask(padded, 4)
    assert out.sum() > grow_mask(m, 4).sum()
    # and the un-clipped result, cropped back, contains the whole grown region
    back = out[pad:-pad, pad:-pad]
    assert np.array_equal(back, grow_mask(m, 4)) or back.sum() >= grow_mask(m, 4).sum()


def test_no_warning_when_region_fits():
    m = block(size=(3, 3), origin=(10, 10))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        grow_mask(m, 2)
        scale_mask(m, 1.5)


def test_transform_mask_rejects_3d():
    with pytest.raises(ValueError, match="2-D"):
        grow_mask(np.zeros((3, 3, 3), bool), 1)


def test_grow_is_strict_superset():
    """Positive distance never removes a pixel, whatever the outline looks like."""
    for m in (block(size=(6, 6)), ring(), staircase(16)):
        for js in ("round", "mitre"):
            for d in (0.5, 1, 2.5):
                g = grow_mask(m, d, join_style=js)
                assert np.array_equal(g & m, m), f"{js} d={d} removed pixels"
                assert g.sum() >= m.sum()


def test_integer_distance_roundtrips_exactly_on_boxes():
    """The one case that is guaranteed: whole-pixel distance, axis-aligned shape."""
    for size in ((4, 4), (6, 3)):
        m = block(size=size)
        for js in ("round", "mitre"):
            assert np.array_equal(grow_mask(grow_mask(m, 1, join_style=js), -1, join_style=js), m)
            assert np.array_equal(grow_mask(grow_mask(m, 3, join_style=js), -3, join_style=js), m)


def test_fractional_distance_is_a_smoothing_not_an_inverse():
    """Documented limitation, anchored to measured numbers.

    buffer(+d) then buffer(-d) is a morphological opening: it rounds convex
    corners and cannot restore the sharp staircase tips it cut away.  On this
    16-px staircase at d=2.5 it keeps 81 of 136 pixels' worth of outline.  So
    never use grow/shrink to "temporarily" enlarge a mask and expect it back --
    keep the original array instead.
    """
    m = np.zeros((40, 40), bool)
    m[10:26, 10:26] = staircase(16)
    back = grow_mask(grow_mask(m, 2.5), -2.5)
    assert back.sum() < m.sum()
    iou = (back & m).sum() / (back | m).sum()
    assert 0.3 < iou < 0.9, f"smoothing behaviour changed: IoU={iou:.3f}"
    # but it stays inside the grown footprint, and never empties the region
    assert np.array_equal(back & grow_mask(m, 2.5), back)
    assert back.sum() > 0


def test_shrink_below_half_thickness_deletes():
    """A 1-px feature cannot survive a 2-px shrink; it vanishes cleanly."""
    m = np.zeros((30, 30), bool)
    m[15, 5:25] = True
    assert grow_mask(m, -2).sum() == 0
    assert grow_mask(m, -0.4).sum() > 0


def test_end_to_end_pipeline_on_staircase():
    """The full round trip the package promises: mask -> polys -> grow -> mask.

    The staircase is embedded in a padded frame so the grown outline stays
    inside the grid; otherwise the documented clipping warning applies and the
    rasterised area is legitimately smaller than the polygon area.
    """
    ragged = staircase(16)
    m = np.zeros((40, 40), bool)
    m[10 : 10 + 16, 10 : 10 + 16] = ragged

    polys = mask_to_polygons(m)
    grown = grow_polygons(polys, 2.5, join_style="round", quad_segs=16)
    out = polygons_to_mask(grown, m.shape)
    assert out.sum() > m.sum()
    assert np.array_equal(out & m, m), "growing must remain a superset"

    # Rasterisation samples pixel centres, so the counted area differs from the
    # polygon area by an amount bounded by the perimeter, not by the area.  A
    # rounded buffer of a jagged outline always comes out *smaller*.
    vec_area = sum(p.area for p in grown)
    perim2 = sum(p.length for p in grown) / 2
    assert out.sum() <= vec_area + 1e-9
    assert (vec_area - out.sum()) <= 0.9 * perim2


@pytest.mark.parametrize("join", ["round", "mitre", "bevel"])
@pytest.mark.parametrize("distance", [0.49, 0.5, 1.5, 2.5, 5.5])
def test_rasterisation_error_is_bounded_by_half_the_perimeter(join, distance):
    """The O(perimeter) quantisation law -- on its magnitude, never its sign.

    An earlier version of this test asserted the deficit was always <= 0 for
    ``join="round"``.  That was wrong: the sign flips as *distance* sweeps past
    a ring of pixel centres, and the worst case sits just below a half-integer.
    Only the bound is reliable, and it involves the outline length, not the area.

    The tolerance is 1.1 rather than 1.0 because the perimeter is measured on
    the *buffered* outline; the observed supremum of the ratio is 1.05 (mitre, a
    7x5 box at d=0.51), and it tends to 1.0 as distance approaches 0.5 from
    below, where the outline can no longer reach the next ring of centres.
    """
    m = np.zeros((40, 40), bool)
    m[10:26, 10:26] = staircase(16)
    grown = grow_polygons(mask_to_polygons(m), distance, join_style=join, quad_segs=16)
    out = polygons_to_mask(grown, m.shape)
    vec = sum(p.area for p in grown)
    perim2 = sum(p.length for p in grown) / 2
    assert abs(vec - out.sum()) <= 1.1 * perim2, f"{join} d={distance}"


def test_rasterisation_error_changes_sign_with_distance():
    """Proof that no one-sided bound can be written, on a 100x100 box."""
    m = np.zeros((200, 200), bool)
    m[50:150, 50:150] = True
    signs = []
    for distance in (0.49, 0.51):
        grown = grow_polygons(mask_to_polygons(m), distance, join_style="mitre")
        out = polygons_to_mask(grown, m.shape)
        signs.append(sum(p.area for p in grown) - out.sum())
    assert signs[0] > 0 > signs[1]


@pytest.mark.parametrize("make", [
    lambda: block(size=(30, 35), origin=(10, 15), shape=(60, 60)),
    lambda: np.pad(staircase(16), ((20, 24), (20, 24)), constant_values=False),
])
@pytest.mark.parametrize("distance", [0.5, 1.0, 1.5, 2.0, 3.5])
def test_mitre_is_exact_on_solid_rectilinear_outlines(make, distance):
    """At multiples of 0.5 the buffered outline passes through pixel centres.

    So for a region that is rectilinear, solid and not nearly touching another,
    the rasterised count equals the vector area to machine precision -- the one
    case where "trust the vector area" is literally true.  ``"round"`` never is.
    """
    m = make()
    grown = grow_polygons(mask_to_polygons(m), distance, join_style="mitre")
    out = polygons_to_mask(grown, m.shape)
    assert abs(sum(p.area for p in grown) - out.sum()) < 1e-9, f"d={distance}"


def test_a_hole_breaks_mitre_exactness_at_half_integers():
    """An interior ring moves the *wrong way* relative to the centres.

    Growing shrinks the hole by *distance*.  At a half-integer distance the hole
    boundary lands exactly on a ring of pixel centres, and a centre on the
    boundary is excluded, so the raster keeps pixels the vector area has already
    given to the hole.  The sign is therefore negative (pixels > vector), the
    mirror of the usual case.  Integer distances stay exact, and so does any
    distance large enough to close the hole entirely.
    """
    m = ring(outer=(20, 20), inner=(6, 6), pad=4, shape=(60, 60))
    polys = mask_to_polygons(m)
    assert len(polys[0].interiors) == 1, "the ring must have a hole"
    for distance in (1.0, 2.0):
        grown = grow_polygons(polys, distance, join_style="mitre")
        out = polygons_to_mask(grown, m.shape)
        assert abs(sum(p.area for p in grown) - out.sum()) < 1e-9, distance
    for distance in (0.5, 1.5):
        grown = grow_polygons(polys, distance, join_style="mitre")
        out = polygons_to_mask(grown, m.shape)
        diff = sum(p.area for p in grown) - out.sum()
        assert diff < 0, f"d={distance}: hole case should run the other way"
        assert abs(diff) <= 1.1 * sum(p.length for p in grown) / 2
    # once the hole is squeezed shut it behaves like a solid box again
    grown = grow_polygons(polys, 3.5, join_style="mitre")
    assert all(len(p.interiors) == 0 for p in grown)
    out = polygons_to_mask(grown, m.shape)
    assert abs(sum(p.area for p in grown) - out.sum()) < 1e-9


def test_overlapping_neighbours_double_count_in_the_vector_area():
    """grow_polygons buffers each region alone; the mask is their union.

    Two boxes 3 columns apart, grown by 2, end up overlapping by a 1x9 strip.
    Summing polygon areas counts it twice, rasterising counts it once, so the
    difference is exactly the overlap -- nothing to do with quantisation.  Use
    shapely.unary_union before summing areas if you want the comparison to mean
    anything for a multi-region mask.
    """
    from shapely import unary_union

    m = np.zeros((60, 60), bool)
    m[10:40, 15:50] = True
    m[20:25, 5:12] = True  # 3-column gap from the big box
    polys = mask_to_polygons(m)
    assert len(polys) == 2

    for distance in (0.5, 1.0, 1.5):  # still disjoint, so still exact
        grown = grow_polygons(polys, distance, join_style="mitre")
        out = polygons_to_mask(grown, m.shape)
        assert abs(sum(p.area for p in grown) - out.sum()) < 1e-9, distance

    grown = grow_polygons(polys, 2.0, join_style="mitre")
    out = polygons_to_mask(grown, m.shape)
    naive = sum(p.area for p in grown)
    merged = unary_union(grown).area
    assert naive - out.sum() == pytest.approx(9.0), "the overlap is 1 x 9 px"
    assert merged == pytest.approx(out.sum()), "the union matches the raster exactly"


def test_sub_half_pixel_distance_adds_no_pixels():
    """Centres are 1 px apart, so nothing below 0.5 can reach a new one.

    This is the practical trap: grow_mask(m, 0.4) silently returns m unchanged
    for any rectilinear region, while 0.5 adds a full ring.
    """
    m = np.zeros((200, 200), bool)
    m[50:150, 50:150] = True
    base = int(m.sum())
    for join in ("round", "mitre"):
        for distance in (0.01, 0.25, 0.4, 0.49):
            out = grow_mask(m, distance, join_style=join)
            assert int(out.sum()) == base, f"{join} d={distance} changed the count"
        assert int(grow_mask(m, 0.5, join_style=join).sum()) > base


def test_clipping_is_warned_about_not_silent():
    """An outline leaving the grid must raise a UserWarning, never vanish quietly."""
    m = staircase(16)  # touches x=0 and y=0 by construction
    with pytest.warns(UserWarning, match="outside the 16x16 output grid"):
        grow_mask(m, 2.5)


def test_example_image_mask_transforms():
    """The shipped real mask must survive the pipeline without degenerating."""
    import warnings

    from astropy.io import fits

    from ._shapes import EXAMPLE_FITS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(EXAMPLE_FITS) as hdul:
            m = np.asarray(hdul[1].data).astype(bool)

    polys = mask_to_polygons(m)
    assert len(polys) >= 1
    assert all(p.is_valid for p in polys)
    grown = grow_mask(m, 3)
    assert grown.sum() > m.sum()
    assert grown.dtype == np.bool_
    shrunk = grow_mask(m, -1)
    assert shrunk.sum() < m.sum()
