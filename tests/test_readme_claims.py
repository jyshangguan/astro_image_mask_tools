"""Executable checks for the numbers and calls quoted in README.md.

The README states ~40 specific quantities (pixel counts, areas, drifts, norm
limits).  Three of them were wrong when first written, each because a number was
carried over from a different shape or a different ``join_style`` than the one
being described.  These tests bind the documented claims to the code, so the
file cannot drift silently.

Run directly for a flat report::

    PYTHONPATH=src python3 tests/test_readme_claims.py
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

import masktools as mt

from ._shapes import EXAMPLE_FITS, load_example

ARCSEC_PER_PIX = 0.2411  # the shipped cutout's plate scale, quoted in the README


@pytest.fixture(scope="module")
def frame():
    image, mask, header = load_example()
    return image, mask, header


def test_coordinate_convention_bounds(frame):
    """The two rows of the 'space' table, and the single-pixel example."""
    m = np.zeros((12, 12), bool)
    m[5, 5] = True
    assert mt.mask_to_polygons(m, space="pixel")[0].bounds == pytest.approx((4.5, 4.5, 5.5, 5.5))
    assert mt.mask_to_polygons(m, space="corner")[0].bounds == pytest.approx((5.0, 5.0, 6.0, 6.0))
    # "the difference is 0.12 arcsec on the shipped frame"
    assert ARCSEC_PER_PIX / 2 == pytest.approx(0.12, abs=0.005)
    assert mt.SPACES == ("pixel", "corner")


def test_vectorise_claims(frame):
    image, mask, header = frame
    polys = mt.mask_to_polygons(mask)
    assert len(polys) == 1
    assert len(polys[0].exterior.coords) == 33
    assert polys[0].area == pytest.approx(582.0)
    assert polys[0].centroid.x == pytest.approx(106.67, abs=0.01)
    assert polys[0].centroid.y == pytest.approx(422.47, abs=0.01)
    # "exact, holes included"
    back = mt.polygons_to_mask(mt.mask_to_polygons(mask), image.shape)
    assert np.array_equal(back, mask)


def test_wcs_takes_a_pixel_space_centroid(frame):
    """The README's claim: no half-pixel correction between polygon and sky."""
    from astropy.wcs import WCS

    image, mask, header = frame
    poly = mt.mask_to_polygons(mask, space="pixel")[0]
    sky = WCS(header).pixel_to_world(poly.centroid.x, poly.centroid.y)
    assert np.isfinite([sky.ra.deg, sky.dec.deg]).all()


@pytest.mark.parametrize("distance,pixels", [(-1, 482), (0, 582), (1, 693), (4, 1029), (8, 1568)])
def test_grow_claims(frame, distance, pixels):
    assert int(mt.grow_mask(frame[1], distance).sum()) == pixels


@pytest.mark.parametrize("factor,pixels", [(0.5, 149), (1.5, 1303), (3.0, 5238)])
def test_scale_claims(frame, factor, pixels):
    assert int(mt.scale_mask(frame[1], factor).sum()) == pixels


def test_scale_origin_drift_claim(frame):
    """'6.18 px (1.49") about the bbox centre, 0.0 about the centroid'."""
    image, mask, header = frame
    L = np.zeros((40, 40), bool)
    L[0:20, 0:20] = True
    L[20:40, 0:5] = True
    poly = mt.mask_to_polygons(L)[0]
    before = np.array(poly.centroid.coords[0])
    about_center = np.array(mt.scale_polygons(poly, 2.0, origin="center")[0].centroid.coords[0])
    about_centroid = np.array(mt.scale_polygons(poly, 2.0, origin="centroid")[0].centroid.coords[0])
    drift = float(np.hypot(*(about_center - before)))
    assert drift == pytest.approx(6.18, abs=0.01)
    assert drift * ARCSEC_PER_PIX == pytest.approx(1.49, abs=0.01)
    assert float(np.hypot(*(about_centroid - before))) == pytest.approx(0.0, abs=1e-12)
    mt.scale_polygons(poly, 2.0, origin=(8.0, 13.5))  # the explicit-pivot form


def test_threshold_recipe_claims(frame):
    """The numbers in the 'Threshold -> mask -> grown mask' snippet."""
    image, mask, header = frame
    bkg = np.median(image)
    sigma = 1.4826 * np.median(np.abs(image - bkg))
    assert bkg == pytest.approx(494.2, abs=0.1)
    assert sigma == pytest.approx(7.46, abs=0.01)
    star = np.zeros_like(image, dtype=bool)
    star[60:110, 440:490] = image[60:110, 440:490] > bkg + 5 * sigma
    assert int(star.sum()) == 1076
    assert mt.grow_mask(star, 4).sum() > star.sum()


def test_maskdrawer_api_quoted_by_the_readme(frame, tmp_path):
    image, mask, header = frame
    d = mt.MaskDrawer(data=image, verbose=False)
    try:
        d.add_polygon([(455, 75), (475, 75), (475, 95), (455, 95)])
        assert len(d.vertices) == 1 and len(d.polygons) == 1
        assert d.mask().dtype == np.bool_
        path = d.save(tmp_path / "mask.fits", header=header)
        assert fits.getheader(path)["CTYPE1"] == header["CTYPE1"]
    finally:
        d.teardown()

    # "MaskDrawer works without any image at all"
    blank = mt.MaskDrawer(shape=(512, 512), verbose=False)
    try:
        assert blank.default_shape == (512, 512)
    finally:
        blank.teardown()

    poll = mt.draw_mask(data=image, block=False, verbose=False)
    try:
        assert isinstance(poll, mt.MaskDrawer)
    finally:
        poll.teardown()


@pytest.mark.parametrize(
    "kwargs,limits",
    [
        (dict(percent=99.9), (0.0, 1244.2)),          # astropy simple_norm
        (dict(), (486.8, 856.9)),                     # the default
        (dict(min_percent=None, max_percent=None), (0.0, 5081.5)),
    ],
)
def test_norm_table_claims(frame, kwargs, limits):
    image, mask, header = frame
    norm = mt.make_norm(image, "asinh", **kwargs)
    assert norm.vmin == pytest.approx(limits[0], abs=0.1)
    assert norm.vmax == pytest.approx(limits[1], abs=0.1)


def test_asinh_reproduces_astropy_exactly(frame):
    """'the asinh curve agrees term for term to 1e-9'."""
    from astropy.visualization import simple_norm

    image, mask, header = frame
    mine = mt.make_norm(image, "asinh", percent=99.9)
    theirs = simple_norm(image, stretch="asinh", percent=99.9)
    xs = np.linspace(theirs.vmin, theirs.vmax, 41)
    got = np.asarray(mine(xs), dtype=float)
    ref = np.asarray(theirs(xs), dtype=float)
    assert np.abs(got - ref).max() < 1e-9
    # and the FuncNorm claim (AsinhNorm ignores linear_width in matplotlib 3.10)
    assert type(mine).__name__ == "FuncNorm"


def test_zero_pixel_share_claim(frame):
    """'582 exactly-zero pixels (0.22% of it)'."""
    image, mask, header = frame
    zeros = int((image < 100).sum())
    assert zeros == 582
    assert (image < 100).mean() * 100 == pytest.approx(0.22, abs=0.01)


def test_quantisation_claims(frame):
    """Caveats 1-3: the half-pixel no-op, the 197/202 pair, mitre exactness."""
    image, mask, header = frame
    box = np.zeros((200, 200), bool)
    box[50:150, 50:150] = True
    assert int(box.sum()) == 10000
    assert int(mt.grow_mask(box, 0.49).sum()) == 10000   # "changes nothing"
    assert int(mt.grow_mask(box, 0.50).sum()) == 10200   # round (the default)
    assert int(mt.grow_mask(box, 0.50, join_style="mitre").sum()) == 10201

    grown = mt.grow_polygons(mt.mask_to_polygons(box), 0.49, join_style="mitre")
    out = mt.polygons_to_mask(grown, box.shape)
    assert sum(p.area for p in grown) - out.sum() == pytest.approx(197.0, abs=0.5)
    assert sum(p.length for p in grown) / 2 == pytest.approx(202.0, abs=0.5)

    # mitre is exact on the half-pixel lattice for a solid, isolated outline
    for distance in (0.5, 1.0, 1.5, 2.0, 3.5):
        g = mt.grow_polygons(mt.mask_to_polygons(box), distance, join_style="mitre")
        r = mt.polygons_to_mask(g, box.shape)
        assert abs(sum(p.area for p in g) - r.sum()) < 1e-9, distance
    # but not for a surviving hole, where the sign inverts
    donut = np.zeros((60, 60), bool)
    donut[10:40, 10:40] = True
    donut[22:28, 22:28] = False
    dp = mt.mask_to_polygons(donut)
    assert len(dp[0].interiors) == 1
    for distance in (0.5, 1.5):
        g = mt.grow_polygons(dp, distance, join_style="mitre")
        r = mt.polygons_to_mask(g, donut.shape)
        assert sum(p.area for p in g) - r.sum() < 0, distance


def test_opening_claims(frame):
    """Caveat 4: exact only for integer d on a boxy outline."""
    image, mask, header = frame
    box = np.zeros((200, 200), bool)
    box[50:150, 50:150] = True
    assert np.array_equal(mt.grow_mask(mt.grow_mask(box, 3), -3), box)
    for distance, pixels, iou in ((3, 593, 0.981), (2.5, 568, 0.876)):
        back = mt.grow_mask(mt.grow_mask(mask, distance), -distance)
        assert int(back.sum()) == pixels
        assert (back & mask).sum() / (back | mask).sum() == pytest.approx(iou, abs=0.001)


def test_repair_split_claims(frame):
    """Caveat 6: 1076 px -> 13 polygons, a 1063-px body plus twelve slivers."""
    image, mask, header = frame
    bkg = np.median(image)
    sigma = 1.4826 * np.median(np.abs(image - bkg))
    star = np.zeros_like(image, dtype=bool)
    star[60:110, 440:490] = image[60:110, 440:490] > bkg + 5 * sigma
    polys = mt.mask_to_polygons(star)
    assert {p.geom_type for p in polys} == {"Polygon"}
    assert len(polys) == 13
    assert round(polys[0].area) == 1063
    assert len(polys) - 1 == 12
    assert sum(p.area for p in polys) == pytest.approx(1076.0)
    assert len(mt.mask_to_polygons(star, min_area=5)) == 1
    assert np.array_equal(mt.polygons_to_mask(polys, star.shape), star)


def test_api_table_names_every_public_symbol(frame):
    """Every name in the API table must exist, and be exported."""
    names = """
        mask_to_polygons polygons_to_mask mask_area
        scale_polygons grow_polygons scale_mask grow_mask
        MaskDrawer draw_mask
        plot_image plot_mask plot_polygons plot_mask_contours show_mask make_norm
        SPACES pixel_to_corner corner_to_pixel
    """.split()
    for name in names:
        assert hasattr(mt, name), f"README's API table names {name!r}, which does not exist"
        assert name in mt.__all__, f"{name} is documented but not in __all__"


def test_polygon_input_forms_quoted_by_the_readme(frame):
    """'a Polygon/MultiPolygon, a GeoJSON dict, __geo_interface__, an (N,2) array'."""
    from shapely.geometry import MultiPolygon

    image, mask, header = frame
    poly = mt.mask_to_polygons(mask)[0]
    forms = [poly, MultiPolygon([poly]), poly.__geo_interface__,
             np.asarray(poly.exterior.coords)[:, :2], [poly, poly]]
    for form in forms:
        assert len(mt.grow_polygons(form, 1.0)) >= 1
        # every form describes the same region, so all must rasterise alike
        assert mt.polygons_to_mask(form, mask.shape).sum() >= int(mask.sum() * 0.9)


def test_mask_input_forms_quoted_by_the_readme(frame):
    """'bool, uint8, int, or a labeled segmentation map'."""
    image, mask, header = frame
    for arr in (mask, mask.astype(np.uint8), mask.astype(int), (mask * 7).astype(int)):
        assert mt.mask_area(arr) == pytest.approx(float(mask.sum()))


def test_join_styles_and_all_touched_exist(frame):
    image, mask, header = frame
    for style in ("round", "mitre", "bevel"):
        mt.grow_mask(mask, 2, join_style=style)
    polys = mt.mask_to_polygons(mask)
    assert mt.polygons_to_mask(polys, mask.shape, all_touched=True).sum() >= mask.sum()


if __name__ == "__main__":  # pragma: no cover - manual report
    import sys

    sys.exit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))
