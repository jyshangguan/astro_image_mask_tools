"""Tests for the display helpers.

These check geometry as well as plumbing: the point of the package is that an
outline lands on the pixels it describes, so the overlay alignment is asserted
numerically rather than by eye.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize, PowerNorm

import masktools._plotting as mp
from masktools._geometry import HALF_PIXEL
from masktools._vectorize import mask_to_polygons

from ._shapes import block, load_example, ring, staircase


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ------------------------------------------------------------------- make_norm
def test_make_norm_returns_the_right_class_per_stretch():
    data = np.linspace(0, 1000, 100).reshape(10, 10)
    assert isinstance(mp.make_norm(data, "linear"), Normalize)
    assert isinstance(mp.make_norm(data, "sqrt"), PowerNorm)
    assert isinstance(mp.make_norm(data, "log"), LogNorm)
    asinh = mp.make_norm(data, "asinh")
    assert type(asinh).__name__ == "FuncNorm"


def test_make_norm_validates_stretch():
    data = np.zeros((4, 4))
    with pytest.raises(ValueError, match="stretch must be"):
        mp.make_norm(data, "arcsinh")  # the common misspelling
    with pytest.raises(ValueError, match="stretch must be"):
        mp.make_norm(data, "squared")


def test_make_norm_rejects_all_non_finite():
    with pytest.raises(ValueError, match="no finite pixels"):
        mp.make_norm(np.full((3, 3), np.nan))


def test_make_norm_percent_is_symmetric_like_astropy():
    """percent=P means p[(100-P)/2] .. p[100-(100-P)/2], not median .. pP."""
    data = np.arange(10000, dtype=float)
    norm = mp.make_norm(data, "linear", percent=99.0)
    assert norm.vmin == pytest.approx(np.percentile(data, 0.5))
    assert norm.vmax == pytest.approx(np.percentile(data, 99.5))
    assert norm.clip is True


def test_make_norm_plain_min_max_needs_all_three_off():
    """min/max_percent are the default, so opt out of them to get min/max."""
    data = np.array([-5.0, 0.0, 0.0, 0.0, 3.0])
    norm = mp.make_norm(data, "linear", min_percent=None, max_percent=None)
    assert norm.vmin == pytest.approx(-5.0)
    assert norm.vmax == pytest.approx(3.0)


def test_make_norm_precedence_vmin_vmax_then_percent_then_min_max():
    data = np.arange(1000, dtype=float)
    # explicit limits beat everything
    n = mp.make_norm(data, "linear", percent=99.0, vmin=-1, vmax=42)
    assert (n.vmin, n.vmax) == (-1.0, 42.0)
    # percent beats min/max_percent
    n = mp.make_norm(data, "linear", percent=99.0, min_percent=16, max_percent=99.9)
    assert n.vmin == pytest.approx(np.percentile(data, 0.5))
    assert n.vmax == pytest.approx(np.percentile(data, 99.5))
    # and with no percent, the asymmetric defaults apply
    n = mp.make_norm(data, "linear")
    assert n.vmin == pytest.approx(np.percentile(data, 16.0))
    assert n.vmax == pytest.approx(np.percentile(data, 99.9))


def test_make_norm_default_beats_symmetric_clipping_on_a_frame_with_zeros():
    """The regression this default exists for.

    A frame with a small population of zeroed pixels (a saturated star masked
    out, say) has those pixels far below sky.  Symmetric percent=99.9 drags
    vmin down to them and flattens the image; the asymmetric default does not.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(500.0, 10.0, (200, 200))
    data[:25, :25] = 0.0  # 0.25% of the frame zeroed, like the shipped cutout

    sym = mp.make_norm(data, "asinh", percent=99.9)
    default = mp.make_norm(data, "asinh")
    assert sym.vmin < 1.0, "symmetric clipping should have been dragged to zero"
    assert default.vmin > 400.0, "asymmetric default must stay near the sky level"
    # and the default window actually separates sky from a 5-sigma source
    sigma = 1.4826 * np.median(np.abs(data[data > 0] - 500.0))
    bright = float(default(np.array([500.0 + 5 * sigma]))[0])
    sky = float(default(np.array([500.0]))[0])
    assert bright > sky + 0.25


def test_make_norm_explicit_limits_win():
    data = np.arange(100, dtype=float).reshape(10, 10)
    norm = mp.make_norm(data, "linear", percent=50, vmin=-1, vmax=42)
    assert (norm.vmin, norm.vmax) == (-1.0, 42.0)


def test_make_norm_flat_image_does_not_divide_by_zero():
    """A constant image would make vmax == vmin; the norm must still be usable."""
    norm = mp.make_norm(np.full((5, 5), 3.0), "linear")
    assert norm.vmax > norm.vmin
    assert np.all(np.isfinite(norm(np.array([3.0]))))


def test_make_norm_stretch_is_monotonic():
    data = np.geomspace(1, 1e6, 500)
    for stretch in ("linear", "sqrt", "asinh"):
        norm = mp.make_norm(data, stretch, percent=None, vmin=1, vmax=1e6)
        out = np.asarray(norm(data), dtype=float)
        assert np.all(np.diff(out) >= -1e-12), f"{stretch} is not monotonic"
        assert np.isfinite(out).all(), f"{stretch} produced nan/inf"


def test_make_norm_asinh_lifts_faint_structure():
    """The reason for asinh: everything below the bright tail gets pushed up."""
    data = np.array([0.0, 1.0, 10.0, 1e3, 1e5])
    lin = np.asarray(mp.make_norm(data, "linear", vmin=0, vmax=1e5)(data))
    for a in (1e-3, 1e-2, 0.1):
        asi = np.asarray(mp.make_norm(data, "asinh", vmin=0, vmax=1e5, asinh_a=a)(data))
        assert np.all(asi >= lin - 1e-12), f"asinh_a={a} darkened some pixel"
        assert asi[2] > lin[2], f"asinh_a={a} did not lift the faint end"
        assert asi[-1] == pytest.approx(1.0, abs=1e-6)
        assert np.all(np.diff(asi) >= -1e-12)


def test_make_norm_log_clips_non_positive():
    data = np.array([-100.0, 0.0, 1.0, 1e4])
    norm = mp.make_norm(data, "log", percent=None)
    assert norm.vmin > 0, "LogNorm floor must be positive"
    out = np.asarray(norm(data), dtype=float)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("a", [1e-3, 1e-2, 0.1, 0.5, 1.0])
def test_make_norm_asinh_matches_astropy_exactly(a):
    """Term-for-term the same curve as simple_norm(asinh_a=...), so porting is safe.

    Compared over astropy's *own* limits: its percent= rule picks vmin/vmax
    from the data, and comparing two different windows would measure that
    choice rather than the stretch.
    """
    astropy = pytest.importorskip("astropy.visualization")
    data = np.random.default_rng(0).normal(500, 10, (40, 40))
    theirs = astropy.simple_norm(data, stretch="asinh", percent=99.9, asinh_a=a)
    mine = mp.make_norm(data, "asinh", percent=99.9, asinh_a=a)
    assert (mine.vmin, mine.vmax) == pytest.approx((theirs.vmin, theirs.vmax))
    xs = np.linspace(theirs.vmin, theirs.vmax, 41)
    got = np.asarray(mine(xs), dtype=float)
    ref = np.asarray(theirs(xs), dtype=float)
    assert np.abs(got - ref).max() < 1e-9, f"asinh_a={a} diverges from astropy"


# ----------------------------------------------------------------- plot_image
def test_plot_image_returns_axes_and_draws():
    data = np.arange(100, dtype=float).reshape(10, 10)
    ax = mp.plot_image(data)
    assert isinstance(ax, Axes)
    assert len(ax.images) == 1
    assert ax.get_aspect() in ("equal", 1.0)


def test_plot_image_accepts_existing_axes():
    fig, axs = plt.subplots(1, 2)
    out = mp.plot_image(np.zeros((5, 5)), ax=axs[1])
    assert out is axs[1]
    assert len(axs[1].images) == 1
    assert len(axs[0].images) == 0


def test_plot_image_keeps_non_finite_transparent():
    data = np.ones((6, 6))
    data[0, 0] = np.nan
    ax = mp.plot_image(data)
    shown = ax.images[0].get_array()
    assert isinstance(shown, np.ma.MaskedArray)
    assert shown.mask[0, 0]
    assert not shown.mask[1, 1]


def test_plot_image_rejects_3d():
    with pytest.raises(ValueError, match="2-D"):
        mp.plot_image(np.zeros((3, 3, 3)))


def test_plot_image_stretch_kwargs_reach_the_norm():
    data = np.linspace(0, 100, 25).reshape(5, 5)
    ax = mp.plot_image(data, stretch="sqrt", percent=None)
    assert isinstance(ax.images[0].norm, PowerNorm)


# ------------------------------------------------------------------ plot_mask
def test_plot_mask_overlay_on_data():
    data = np.zeros((20, 20))
    m = block(size=(4, 4), origin=(5, 5), shape=(20, 20))
    ax = mp.plot_mask(m, data=data)
    assert len(ax.images) == 2  # background + overlay
    overlay = ax.images[1]
    assert overlay.get_extent() == pytest.approx([-0.5, 19.5, -0.5, 19.5])


def test_plot_mask_extent_puts_pixel_centres_on_integers():
    """The overlay must share the imshow grid, or contours look misaligned."""
    m = np.zeros((10, 14), bool)
    ax = mp.plot_mask(m)
    left, right, bottom, top = ax.images[0].get_extent()
    assert (left, right) == pytest.approx((-0.5, 13.5))
    assert (bottom, top) == pytest.approx((-0.5, 9.5))


def test_plot_mask_without_data():
    ax = mp.plot_mask(np.zeros((8, 8), bool))
    assert len(ax.images) == 1


def test_plot_mask_upper_origin_flips_extent():
    m = np.zeros((10, 10), bool)
    ax = mp.plot_mask(m, origin="upper")
    _, _, bottom, top = ax.images[0].get_extent()
    assert (bottom, top) == pytest.approx((9.5, -0.5))


# -------------------------------------------------------------- plot_polygons
def test_plot_polygons_outline_passes_through_pixel_centres():
    """The half-pixel canary at the display level.

    A single masked pixel at [5,5] must have its outline drawn around
    (4.5..5.5), i.e. centred on the pixel, not offset onto the corner grid.
    """
    m = np.zeros((12, 12), bool)
    m[5, 5] = True
    ax = mp.plot_mask_contours(m)
    line = ax.lines[0]
    x, y = line.get_xdata(), line.get_ydata()
    assert x.min() == pytest.approx(4.5) and x.max() == pytest.approx(5.5)
    assert y.min() == pytest.approx(4.5) and y.max() == pytest.approx(5.5)
    # and the pixel centre is strictly inside the drawn outline
    assert x.min() < 5.0 < x.max()


def test_plot_polygons_aligns_with_the_mask_overlay():
    """Contours and the raster overlay must share one coordinate frame.

    Both are drawn over the same imshow extent, so the outline's centre must
    land on the masked pixels' centre -- not half a pixel away.
    """
    m = block(size=(5, 5), origin=(10, 10), shape=(30, 30))
    ax = mp.plot_mask(m, data=np.zeros((30, 30)))
    mp.plot_mask_contours(m, ax=ax)
    left, right, bottom, top = ax.images[1].get_extent()
    xs, ys = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    assert xs.min() >= left and xs.max() <= right
    assert ys.min() >= bottom and ys.max() <= top
    # a 5x5 block at [10:15, 10:15] is centred on pixel (12, 12)
    assert 0.5 * (xs.min() + xs.max()) == pytest.approx(12.0)
    assert 0.5 * (ys.min() + ys.max()) == pytest.approx(12.0)
    # and the outline spans half a pixel beyond the outermost centres
    assert xs.min() == pytest.approx(9.5) and xs.max() == pytest.approx(14.5)


def test_plot_polygons_accepts_corner_space_input():
    m = np.zeros((12, 12), bool)
    m[5, 5] = True
    px = mp.plot_polygons(mask_to_polygons(m, space="pixel"))
    cn = mp.plot_polygons(mask_to_polygons(m, space="corner"), space="corner")
    a = px.lines[0].get_xdata()
    b = cn.lines[0].get_xdata()
    assert np.allclose(np.sort(a), np.sort(b)), "both spaces must draw the same outline"


def test_plot_polygons_draws_holes_dashed():
    m = ring()
    ax = mp.plot_polygons(mask_to_polygons(m))
    styles = [ln.get_linestyle() for ln in ax.lines]
    assert "-" in styles and "--" in styles


def test_plot_polygons_labels_and_fill():
    m = np.zeros((40, 40), bool)
    m[5:10, 5:10] = True
    m[25:32, 25:32] = True
    ax = mp.plot_polygons(mask_to_polygons(m), labels=True, fill=True,
                          facecolor="blue")
    assert len(ax.texts) == 2
    assert len(ax.patches) == 2


def test_plot_polygons_empty_input_is_harmless():
    ax = mp.plot_polygons([])
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 0


def test_plot_mask_contours_on_real_image():
    """The shipped cutout's jagged island must vectorise and draw."""
    _, m, _ = load_example()
    ax = mp.plot_mask_contours(m)
    assert len(ax.lines) >= 1
    xs = np.concatenate([ln.get_xdata() for ln in ax.lines])
    ys = np.concatenate([ln.get_ydata() for ln in ax.lines])
    assert xs.min() >= -0.5 and xs.max() <= m.shape[1] - 0.5
    assert ys.min() >= -0.5 and ys.max() <= m.shape[0] - 0.5
    # a staircase outline has many more vertices than a rectangle's 5
    assert len(xs) > 20


# ------------------------------------------------------------------ show_mask
def test_show_mask_two_panels_with_data():
    m = block(size=(4, 4), shape=(20, 20))
    fig = mp.show_mask(m, data=np.zeros((20, 20)), title="hi")
    assert len(fig.axes) == 2
    assert {a.get_title() for a in fig.axes} == {"data", "data + mask"}
    assert fig._suptitle.get_text() == "hi"


def test_show_mask_one_panel_without_data():
    fig = mp.show_mask(block(size=(4, 4), shape=(20, 20)))
    assert len(fig.axes) == 1


def test_show_mask_overlay_false_gives_one_panel():
    m = block(size=(4, 4), shape=(20, 20))
    fig = mp.show_mask(m, data=np.zeros((20, 20)), overlay=False)
    assert len(fig.axes) == 1


def test_grown_mask_still_draws_inside_the_frame():
    """End-to-end: vectorise, grow, display, and stay on the grid."""
    from masktools._transform import grow_mask

    _, m, _ = load_example()
    grown = grow_mask(m, 4)
    ax = mp.plot_mask_contours(grown)
    xs = np.concatenate([ln.get_xdata() for ln in ax.lines])
    assert xs.max() <= m.shape[1] - 0.5
    assert grown.sum() > m.sum()
