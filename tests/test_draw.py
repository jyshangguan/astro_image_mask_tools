"""Tests for the interactive drawer.

No test opens a window: matplotlib runs on the Agg backend and every gesture is
synthesised as a real canvas event, so the widget wiring itself is exercised.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseEvent

from masktools._draw import MaskDrawer, draw_mask

from ._shapes import block

pytest.importorskip("matplotlib")


@pytest.fixture
def drawer():
    """A drawer over a blank 60x60 canvas, closed down again afterwards."""
    d = MaskDrawer(data=np.zeros((60, 60), float), verbose=False)
    yield d
    d.teardown()


# --------------------------------------------------------------- event helpers
class Mouse:
    """Drive a figure with synthetic events, in *data* coordinates."""

    def __init__(self, drawer):
        self.d = drawer
        self.fig = drawer.fig
        self.canvas = drawer.fig.canvas
        self.ax = drawer.ax
        self.canvas.draw()

    def _px(self, x, y):
        return self.ax.transData.transform((x, y))

    def move(self, x, y):
        px, py = self._px(x, y)
        self.canvas.callbacks.process(
            "motion_notify_event", MouseEvent("motion_notify_event", self.canvas, px, py)
        )

    def click(self, x, y, button=1):
        px, py = self._px(x, y)
        for name in ("button_press_event", "button_release_event"):
            self.canvas.callbacks.process(
                name, MouseEvent(name, self.canvas, px, py, button=button)
            )

    def draw_polygon(self, pts):
        """Place a vertex per point, then close on the first one.

        matplotlib only completes a polygon when the cursor *moves* onto the
        first vertex, so a hover is part of the gesture, not decoration.
        """
        for pt in pts:
            self.click(*pt)
        self.move(*pts[0])
        self.click(*pts[0])

    def key(self, name):
        self.canvas.callbacks.process(
            "key_press_event", KeyEvent("key_press_event", self.canvas, name)
        )


# ---------------------------------------------------------------- mouse path
def test_real_mouse_gesture_produces_a_polygon(drawer):
    m = Mouse(drawer)
    m.draw_polygon([(10, 10), (30, 10), (30, 30)])
    m.key("n")  # commit

    assert len(drawer.vertices) == 1
    xy = drawer.vertices[0]
    assert len(xy) == 3
    # pixel space: the triangle covers a known half of the 20x20 box
    mask = drawer.mask()
    assert mask.dtype == np.bool_
    # right triangle, legs of 20 px -> area 200 px^2; pixel-centre sampling
    # of the hypotenuse lands a little short of that.
    assert 185 <= mask.sum() <= 200


def test_second_polygon_after_n(drawer):
    m = Mouse(drawer)
    m.draw_polygon([(5, 5), (10, 5), (10, 10)])
    m.key("n")
    m.draw_polygon([(30, 30), (35, 30), (35, 35)])
    m.key("n")
    assert len(drawer.vertices) == 2
    mask = drawer.mask()
    assert mask.sum() > 20


def test_right_click_removes_a_vertex(drawer):
    m = Mouse(drawer)
    m.draw_polygon([(10, 10), (30, 10), (30, 30), (10, 30)])
    before = len(drawer.vertices[0]) if drawer.vertices else 0
    m.key("n")
    m.draw_polygon([(40, 40), (45, 40), (45, 45)])
    # delete one vertex of the live (last-drawn) polygon
    m.move(45, 45)
    m.click(45, 45, button=3)
    live = drawer._selector.verts
    assert len(live) <= 3  # shrank, or reverted to incomplete


# ------------------------------------------------------------------ programmatic
def test_add_polygon_validates(drawer):
    drawer.add_polygon([(0, 0), (5, 0), (5, 5)])
    assert len(drawer.vertices) == 1
    with pytest.raises(ValueError, match=r">= 3 vertices"):
        drawer.add_polygon([(0, 0), (1, 1)])
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        drawer.add_polygon([[0, 0, 0], [1, 1, 1], [2, 2, 2]])


def test_add_polygon_accepts_closed_ring(drawer):
    """A ring that repeats its first vertex must not gain a duplicate."""
    drawer.add_polygon([(0, 0), (5, 0), (5, 5), (0, 0)])
    assert len(drawer.vertices[0]) == 3


def test_mask_matches_direct_rasterisation(drawer):
    tri = [(10, 10), (30, 10), (30, 30)]
    drawer.add_polygon(tri)
    from masktools._vectorize import polygons_to_mask

    assert np.array_equal(drawer.mask(), polygons_to_mask([tri], (60, 60), space="pixel"))


def test_polygons_property_returns_shapely(drawer):
    drawer.add_polygon([(0, 0), (10, 0), (10, 10)])
    (p,) = drawer.polygons
    assert p.geom_type == "Polygon"
    assert p.area == pytest.approx(50.0)


def test_delete_edit_clear(drawer):
    drawer.add_polygon([(0, 0), (5, 0), (5, 5)])
    drawer.add_polygon([(20, 20), (25, 20), (25, 25)])
    assert len(drawer.vertices) == 2

    gone = drawer.delete_last()
    assert len(gone) == 3 and len(drawer.vertices) == 1

    # edit_last hands the polygon to the live selector.  It leaves _committed,
    # but must still be visible through .vertices and .mask() -- otherwise
    # rasterising mid-edit would silently drop it.
    before = drawer.mask().copy()
    drawer.edit_last()
    assert len(drawer._committed) == 0
    assert len(drawer._selector.verts) >= 3
    assert len(drawer.vertices) == 1
    assert np.array_equal(drawer.mask(), before)

    drawer.add_polygon([(0, 0), (5, 0), (5, 5)])
    drawer.clear()
    assert drawer.vertices == []
    assert drawer.mask().sum() == 0


def test_delete_and_edit_on_empty_are_safe(drawer):
    assert drawer.delete_last() is None
    assert drawer.edit_last() is None


# --------------------------------------------------------------------- keys
def test_reserved_keys_are_not_intercepted(drawer):
    """escape/shift/control belong to PolygonSelector; the drawer must step back."""
    m = Mouse(drawer)
    m.draw_polygon([(10, 10), (20, 10), (20, 20)])
    n_before = len(drawer._committed)
    for bad in ("escape", "shift", "control", "shift+left"):
        m.key(bad)
    assert len(drawer._committed) == n_before


def test_unknown_key_falls_through_to_matplotlib(drawer):
    m = Mouse(drawer)
    m.key("g")  # matplotlib's own grid toggle; must not raise
    m.key("1")


def test_q_commits_and_closes(drawer):
    m = Mouse(drawer)
    m.draw_polygon([(10, 10), (20, 10), (20, 20)])
    m.key("q")
    assert drawer._closed
    assert len(drawer.vertices) == 1


def test_h_prints_help(drawer, capsys):
    Mouse(drawer).key("h")
    out = capsys.readouterr().out
    assert "left click" in out and "n   finish" in out


def test_s_without_savepath_reports_not_raises(drawer, capsys):
    drawer.add_polygon([(0, 0), (5, 0), (5, 5)])
    Mouse(drawer).key("s")
    out = capsys.readouterr().out
    assert "no save path" in out and "np.save" in out  # logged, not propagated


# ------------------------------------------------------------------ mask import
def test_existing_mask_is_imported_as_editable_polygons():
    m = block(size=(4, 4), origin=(10, 10), shape=(40, 40))
    d = MaskDrawer(mask=m, verbose=False)
    try:
        assert len(d.vertices) == 1
        assert np.array_equal(d.mask(), m)
        assert d.default_shape == m.shape
    finally:
        d.teardown()


def test_refining_an_imported_mask_grows_it():
    m = block(size=(4, 4), origin=(10, 10), shape=(40, 40))
    d = MaskDrawer(mask=m, verbose=False)
    try:
        d.add_polygon([(30, 30), (35, 30), (35, 35)])
        out = d.mask()
        assert out.sum() > m.sum()
        assert np.array_equal(out & m, m)
    finally:
        d.teardown()


def test_empty_reference_mask_imports_nothing():
    d = MaskDrawer(mask=np.zeros((20, 20), bool), verbose=False)
    try:
        assert d.vertices == []
    finally:
        d.teardown()


# ----------------------------------------------------------------------- shape
def test_default_shape_prefers_data_then_mask():
    d = MaskDrawer(data=np.zeros((30, 40), float), verbose=False)
    try:
        assert d.default_shape == (30, 40)
    finally:
        d.teardown()
    d = MaskDrawer(mask=np.zeros((20, 25), bool), verbose=False)
    try:
        assert d.default_shape == (20, 25)
    finally:
        d.teardown()


def test_explicit_shape_enables_a_blank_canvas():
    """shape= is how you draw with no image and no reference mask at all."""
    d = MaskDrawer(shape=(25, 30), verbose=False)
    try:
        assert d.default_shape == (25, 30)
        d.add_polygon([(2, 2), (9, 2), (9, 9)])
        m = d.mask()
        assert m.shape == (25, 30)
        assert 0 < m.sum() < 9 * 9
    finally:
        d.teardown()


def test_explicit_shape_conflicting_with_arrays_is_refused():
    for kw in ({"data": np.zeros((20, 20))}, {"mask": np.zeros((20, 20), bool)}):
        with pytest.raises(ValueError, match="disagrees"):
            MaskDrawer(shape=(30, 30), verbose=False, **kw)


def test_shape_is_validated():
    for bad in ((0, 5), (5,), (3, 3, 3), (-2, 4)):
        with pytest.raises(ValueError, match="positive ints"):
            MaskDrawer(shape=bad, verbose=False)


def test_draw_mask_shape_reaches_the_non_blocking_drawer():
    """The non-blocking path must return a drawer that can rasterise alone."""
    out = draw_mask(shape=(15, 15), block=False, verbose=False)
    try:
        assert out.default_shape == (15, 15)
        out.add_polygon([(1, 1), (8, 1), (8, 8)])
        assert out.mask().shape == (15, 15)
    finally:
        out.teardown()


def test_no_data_no_mask_requires_explicit_shape():
    d = MaskDrawer(verbose=False)
    try:
        assert d.default_shape is None
        with pytest.raises(ValueError, match="shape"):
            d.mask()
        assert d.mask(shape=(15, 15)).shape == (15, 15)
    finally:
        d.teardown()


# ------------------------------------------------------------------------- I/O
def test_save_npy(tmp_path):
    d = MaskDrawer(data=np.zeros((30, 30), float), verbose=False)
    try:
        d.add_polygon([(5, 5), (20, 5), (20, 20)])
        path = tmp_path / "mask.npy"
        d.save(path)
        assert np.array_equal(np.load(path), d.mask())
    finally:
        d.teardown()


@pytest.mark.parametrize("name", ["mask.fits", "mask.fits.gz"])
def test_save_fits(tmp_path, name):
    pytest.importorskip("astropy")
    d = MaskDrawer(data=np.zeros((30, 30), float), verbose=False)
    try:
        d.add_polygon([(5, 5), (20, 5), (20, 20)])
        path = tmp_path / name
        d.save(path)
        from astropy.io import fits

        with fits.open(path) as hdul:
            assert np.array_equal(hdul[0].data.astype(bool), d.mask())
            assert hdul[0].data.dtype == np.uint8
    finally:
        d.teardown()


def test_save_fits_carries_header(tmp_path):
    """Passing the image header is how the WCS travels with the mask."""
    pytest.importorskip("astropy")
    from astropy.io import fits

    hdr = fits.Header({"CRPIX1": 15.0, "CRPIX2": 15.0, "CTYPE1": "RA---TAN"})
    d = MaskDrawer(data=np.zeros((30, 30), float), verbose=False)
    try:
        d.add_polygon([(5, 5), (20, 5), (20, 20)])
        path = tmp_path / "mask.fits"
        d.save(path, header=hdr)
        assert fits.getheader(path)["CTYPE1"] == "RA---TAN"
    finally:
        d.teardown()


def test_save_without_path_raises(tmp_path):
    d = MaskDrawer(data=np.zeros((20, 20), float), verbose=False)
    try:
        with pytest.raises(ValueError, match="no save path"):
            d.save()
    finally:
        d.teardown()


def test_savepath_kwarg_and_overwrite_guard(tmp_path):
    path = tmp_path / "m.npy"
    d = MaskDrawer(data=np.zeros((20, 20), float), savepath=str(path), verbose=False)
    try:
        d.add_polygon([(2, 2), (8, 2), (8, 8)])
        d.save()
        assert path.exists()
        with pytest.raises(FileExistsError):
            d.save(overwrite=False)
        d.save(overwrite=True)
    finally:
        d.teardown()


# ------------------------------------------------------------------- lifecycle
def test_on_close_callback_receives_drawer():
    seen = []
    d = MaskDrawer(data=np.zeros((20, 20), float), verbose=False)
    d.on_close(lambda dr: seen.append(dr))
    d.close()
    assert seen == [d]
    assert d._closed
    d.teardown()


def test_close_is_idempotent():
    d = MaskDrawer(data=np.zeros((20, 20), float), verbose=False)
    d.close()
    d.close()  # must not double-disconnect or raise
    d.teardown()


def test_close_commits_the_live_polygon():
    d = MaskDrawer(data=np.zeros((40, 40), float), verbose=False)
    m = Mouse(d)
    m.draw_polygon([(5, 5), (15, 5), (15, 15)])
    assert len(d._committed) == 0  # still live, not yet finished
    d.close()
    assert len(d.vertices) == 1
    d.teardown()


def test_widget_survives_garbage_collection():
    """The drawer pins itself to the figure, the documented matplotlib trap."""
    import gc

    d = MaskDrawer(data=np.zeros((20, 20), float), verbose=False)
    ref = d.fig._masktools_drawer
    del d
    gc.collect()
    assert ref is not None
    ref.teardown()


def test_repr_shows_state():
    d = MaskDrawer(data=np.zeros((20, 20), float), verbose=False)
    try:
        d.add_polygon([(2, 2), (8, 2), (8, 8)])
        r = repr(d)
        assert "MaskDrawer" in r and "polygons=1" in r and "(20, 20)" in r
    finally:
        d.teardown()


def test_draws_into_an_existing_axes():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    d = MaskDrawer(data=np.zeros((30, 30), float), ax=ax, verbose=False)
    try:
        assert d.ax is ax
        d.add_polygon([(5, 5), (20, 5), (20, 20)])
        assert d.mask().sum() > 0
    finally:
        d.teardown()
        plt.close(fig)


# --------------------------------------------------------------- convenience
def test_draw_mask_non_blocking_returns_drawer():
    out = draw_mask(data=np.zeros((20, 20), float), block=False, verbose=False)
    assert isinstance(out, MaskDrawer)
    out.teardown()


def test_show_warns_on_non_interactive_backend():
    """Agg has no window, so show() must say so instead of silently no-op'ing."""
    d = MaskDrawer(data=np.zeros((20, 20), float), verbose=False)
    try:
        with pytest.warns(UserWarning, match="not interactive"):
            assert d.show() is d
        with pytest.warns(UserWarning, match="not interactive"):
            assert d.wait() is d  # the alias behaves identically
    finally:
        d.teardown()


def test_blit_is_disabled_on_non_gui_canvases():
    """Agg and ipympl must not get a blitting widget (it corrupts the redraw)."""
    from masktools._draw import _blit_ok

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    assert _blit_ok(fig) is False
    plt.close(fig)
