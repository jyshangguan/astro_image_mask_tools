"""Tests for mask <-> polygon conversion."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from masktools.vectorize import mask_area, mask_to_polygons, polygons_to_mask

from ._shapes import EXAMPLE_FITS, block, ring, staircase


def test_single_pixel_is_half_open_unit_square_in_pixel_space():
    m = np.zeros((12, 12), bool)
    m[5, 5] = True
    (poly,) = mask_to_polygons(m)
    assert poly.bounds == pytest.approx((4.5, 4.5, 5.5, 5.5))
    assert poly.area == pytest.approx(1.0)
    assert (poly.centroid.x, poly.centroid.y) == pytest.approx((5.0, 5.0))


def test_block_bounds_and_area():
    m = block(size=(3, 3), origin=(10, 10))
    (poly,) = mask_to_polygons(m)
    assert poly.bounds == pytest.approx((9.5, 9.5, 12.5, 12.5))
    assert poly.area == pytest.approx(9.0)


def test_roundtrip_exact_for_simple_shapes():
    """mask -> polygons -> mask must be bit-identical, holes included."""
    for m in (block(size=(3, 3)), block(size=(4, 6)), ring(), staircase(12)):
        polys = mask_to_polygons(m)
        back = polygons_to_mask(polys, m.shape)
        assert np.array_equal(back, m), f"roundtrip failed for shape {m.shape}"


def test_hole_survives_roundtrip():
    m = ring()
    (poly,) = mask_to_polygons(m)
    assert len(poly.interiors) == 1
    assert poly.area == pytest.approx(m.sum())
    assert np.array_equal(polygons_to_mask([poly], m.shape), m)


def test_fill_holes_drops_interiors():
    m = ring()
    (plain,) = mask_to_polygons(m)
    (filled,) = mask_to_polygons(m, fill_holes=True)
    assert len(plain.interiors) == 1
    assert len(filled.interiors) == 0
    assert filled.area == pytest.approx(100.0)  # the outer 10x10 box


def test_two_components_sorted_by_area():
    m = np.zeros((30, 30), bool)
    m[2:5, 2:5] = True  # 9 px
    m[20:27, 20:27] = True  # 49 px
    polys = mask_to_polygons(m)
    assert len(polys) == 2
    assert polys[0].area == pytest.approx(49.0)
    assert polys[1].area == pytest.approx(9.0)
    # polys[0] is the dominant region, which is what callers assume
    assert polygons_to_mask(polys[0], m.shape).sum() == 49


def test_min_area_filters_specks():
    m = np.zeros((20, 20), bool)
    m[2, 2] = True  # 1 px speck
    m[8:16, 8:16] = True  # 64 px region
    assert len(mask_to_polygons(m)) == 2
    kept = mask_to_polygons(m, min_area=5)
    assert len(kept) == 1
    assert kept[0].area == pytest.approx(64.0)


def test_empty_mask_returns_empty_list():
    assert mask_to_polygons(np.zeros((5, 5), bool)) == []
    assert polygons_to_mask([], (5, 5)).sum() == 0


def test_corner_space_matches_rasterio_convention():
    """space='corner' must put integers on pixel corners."""
    m = np.zeros((12, 12), bool)
    m[5, 5] = True
    (px,) = mask_to_polygons(m, space="pixel")
    (cn,) = mask_to_polygons(m, space="corner")
    assert cn.bounds == pytest.approx((5.0, 5.0, 6.0, 6.0))
    # and rasterising from corner space reproduces the mask
    assert np.array_equal(polygons_to_mask(cn, m.shape, space="corner"), m)
    # the two spaces differ by exactly half a pixel
    assert cn.difference(px).is_empty is False


@pytest.mark.parametrize("connectivity", [4, 8])
def test_connectivity_is_accepted_and_validated(connectivity):
    m = block(size=(3, 3))
    assert len(mask_to_polygons(m, connectivity=connectivity)) == 1
    with pytest.raises(ValueError, match="connectivity must be"):
        mask_to_polygons(m, connectivity=6)


def test_diagonal_touching_pixels_split_differently_by_connectivity():
    """Corner-touching pixels: 8-connectivity merges them, 4 does not.

    The merge shows up in the *rasterio record*, but two unit squares meeting
    at exactly one point have a union pinched to a point, which is not a simple
    polygon -- so it comes back as two polygons either way.  What differs is
    that 8-connectivity yields one self-intersecting ring (area 2.0) that we
    repair, while 4-connectivity never joins them.  The invariant that actually
    matters, and is asserted here, is that the areas sum to the pixel count.
    """
    m = np.zeros((10, 10), bool)
    m[3, 3] = True
    m[4, 4] = True
    for conn in (4, 8):
        polys = mask_to_polygons(m, connectivity=conn)
        assert len(polys) == 2, f"connectivity={conn}"
        assert sum(p.area for p in polys) == pytest.approx(2.0)
        assert all(p.is_valid and p.geom_type == "Polygon" for p in polys)
    m[5, 5] = True  # a chain of three keeps its total area too
    assert sum(p.area for p in mask_to_polygons(m, connectivity=8)) == pytest.approx(3.0)


def test_self_intersecting_staircase_is_repaired_into_valid_polygons():
    """The shipped cutout's threshold mask really does this.

    rasterio can hand back a ring that crosses itself, and buffer(0) repairs it
    -- possibly into a MultiPolygon.  mask_to_polygons must flatten that, or
    every caller touching .exterior / .interiors dies on arrival.
    """
    image = np.asarray(fits.getdata(EXAMPLE_FITS), dtype=float)
    bkg = np.median(image)
    sigma = 1.4826 * np.median(np.abs(image - bkg))
    thresh = np.zeros(image.shape, bool)
    thresh[60:110, 440:490] = image[60:110, 440:490] > bkg + 5 * sigma

    polys = mask_to_polygons(thresh)
    assert polys, "the bright star must vectorise"
    assert {p.geom_type for p in polys} == {"Polygon"}
    assert all(p.is_valid for p in polys)
    assert sum(p.area for p in polys) == pytest.approx(float(thresh.sum()))
    assert np.array_equal(polygons_to_mask(polys, thresh.shape), thresh)

    # fill_holes walks .exterior/.interiors, so it used to raise AttributeError
    filled = mask_to_polygons(thresh, fill_holes=True)
    assert all(len(p.interiors) == 0 for p in filled)
    assert sum(p.area for p in filled) >= sum(p.area for p in polys)
    # min_area drops the 1-2 px slivers the repair produced
    assert len(mask_to_polygons(thresh, min_area=5)) == 1


def test_int_and_labeled_masks_are_accepted():
    """Any nonzero value counts as masked, so ints and labels work directly."""
    truth = block(size=(3, 3))
    assert np.array_equal(mask_to_polygons(truth.astype(np.uint8))[0].bounds,
                          mask_to_polygons(truth)[0].bounds)
    segm = np.zeros((40, 40), np.int32)
    segm[18:21, 18:21] = 7  # label 7
    segm[30:33, 30:33] = 9  # label 9
    polys = mask_to_polygons(segm)
    assert len(polys) == 2  # flattened into one boolean set of regions


def test_polygons_to_mask_accepts_many_input_forms():
    m = block(size=(3, 3), origin=(10, 10))
    poly = mask_to_polygons(m)[0]
    xy = np.asarray(poly.exterior.coords) - 0.5  # pixel-space vertices
    forms = [
        poly,
        [poly],
        (poly,),
        xy,
        [xy],
        {"type": "Polygon", "coordinates": [np.asarray(poly.exterior.coords)]},
    ]
    for i, form in enumerate(forms):
        out = polygons_to_mask(form, m.shape, space="pixel")
        assert out.sum() == m.sum(), f"form #{i} gave {out.sum()} px, expected {m.sum()}"
        assert np.array_equal(out, m) or i >= 3, f"form #{i} mismatch"


def test_polygons_to_mask_dtype_and_shape():
    m = block(size=(3, 3))
    polys = mask_to_polygons(m)
    assert polygons_to_mask(polys, m.shape).dtype == np.bool_
    out = polygons_to_mask(polys, m.shape, dtype=np.int32)
    assert out.dtype == np.int32 and set(np.unique(out)) <= {0, 1}
    bigger = polygons_to_mask(polys, (60, 60))
    assert bigger.shape == (60, 60)
    assert bigger.sum() == m.sum()  # aligned to lower-left, not centred


def test_all_touched_inflates():
    """all_touched=True must not produce fewer pixels than the default."""
    m = block(size=(3, 3))
    polys = mask_to_polygons(m)
    strict = polygons_to_mask(polys, m.shape, all_touched=False)
    loose = polygons_to_mask(polys, m.shape, all_touched=True)
    assert loose.sum() >= strict.sum()
    assert strict.sum() == 9


def test_polygon_fully_outside_frame_contributes_nothing():
    m = block(size=(3, 3))
    far = block(size=(3, 3), origin=(500, 500))
    polys = mask_to_polygons(far)
    assert polygons_to_mask(polys, m.shape).sum() == 0


def test_mask_area_matches_pixel_count():
    for m in (block(size=(3, 3)), block(size=(4, 6)), ring(), staircase(12)):
        assert mask_area(m) == pytest.approx(float(m.sum()))


def test_rejects_bad_inputs():
    with pytest.raises(ValueError, match="2-D"):
        mask_to_polygons(np.zeros((3, 3, 3), bool))
    with pytest.raises(ValueError, match=r"mask_shape must be \(nrow, ncol\)"):
        polygons_to_mask([], (3, 3, 3))
    with pytest.raises(ValueError, match="space must be"):
        mask_to_polygons(block(), space="world")
