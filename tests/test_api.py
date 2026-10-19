"""Tests for the two public functions: mask_poly_expand and mask_manual."""

from __future__ import annotations

import builtins
import contextlib

import matplotlib
import numpy as np
import pytest

import masktools as mt
from masktools._draw import MaskDrawer, _run_in_popup
from masktools._vectorize import mask_to_polygons

from ._shapes import block, load_example, ring, staircase

# --------------------------------------------------------------------------
# mask_poly_expand
# --------------------------------------------------------------------------

#: The exact numbers the README and the docstrings quote.
REAL_MASK_COUNTS = {0.5: 149, 1.0: 582, 1.5: 1303, 2.0: 2328, 3.0: 5238}


def test_factor_one_is_the_identity():
    """factor=1 must give the input back bit-for-bit, ragged edges included."""
    for m in (block(size=(10, 10)), ring(), staircase(12), load_example()[1]):
        assert np.array_equal(mt.mask_poly_expand(m, 1.0), m)


def test_docstring_box_numbers():
    m = np.zeros((40, 40), bool)
    m[10:20, 10:20] = True
    assert mt.mask_poly_expand(m, 2.0).sum() == 400
    assert mt.mask_poly_expand(m, 0.5).sum() == 25
    assert mt.mask_poly_expand(m, 1.5).sum() == 225


@pytest.mark.parametrize(("factor", "count"), sorted(REAL_MASK_COUNTS.items()))
def test_real_mask_counts(factor, count):
    """The shipped cutout's island, at the factors the README tabulates."""
    data, mask, header = load_example()
    assert mt.mask_poly_expand(mask, factor).sum() == count


def test_area_scales_as_factor_squared_within_the_pixel_bound():
    """Pixels ~ factor**2 * N, off by no more than half the scaled perimeter.

    The bound is the measured rasterisation law of the pipeline.  The test also
    checks it is not vacuous: the tolerance stays well under the area itself.
    """
    data, mask, header = load_example()
    perimeter = sum(p.length for p in mask_to_polygons(mask))
    n = mask.sum()
    for factor in (0.5, 0.8, 1.0, 1.3, 1.5, 2.0, 2.7, 3.0):
        got = mt.mask_poly_expand(mask, factor).sum()
        expected = factor**2 * n
        tol = 1.1 * factor * perimeter / 2
        assert abs(got - expected) <= tol, f"factor={factor}: {got} vs {expected}"
        assert tol < 0.25 * expected, "the bound went vacuous at this scale"


def test_monotone_in_factor():
    data, mask, header = load_example()
    counts = [mt.mask_poly_expand(mask, f).sum() for f in (0.5, 1.0, 1.5, 2.0, 3.0)]
    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts), "counts must strictly increase"


def test_returns_bool_of_the_same_shape():
    data, mask, header = load_example()
    out = mt.mask_poly_expand(mask, 1.7)
    assert out.dtype == bool
    assert out.shape == mask.shape


def test_shrink_floor_depends_on_where_the_centroid_lands():
    """A region cannot shrink below one pixel if its centroid is on a centre.

    Rasterisation samples pixel centres.  An odd-sized box has its centroid on
    one, so a shrunken copy still covers it and one pixel always survives; an
    even-sized box has its centroid between four centres, so it vanishes.
    """
    for size, floor in ((3, 1), (4, 0), (5, 1), (6, 0)):
        m = np.zeros((40, 40), bool)
        m[18 : 18 + size, 18 : 18 + size] = True
        for factor in (0.1, 0.03, 0.01, 0.001):
            assert mt.mask_poly_expand(m, factor).sum() == floor, f"size={size}"


def test_empty_mask_stays_empty():
    m = np.zeros((25, 30), bool)
    out = mt.mask_poly_expand(m, 3.0)
    assert out.sum() == 0 and out.shape == m.shape and out.dtype == bool


def test_non_boolean_input_is_accepted_by_nonzero():
    """uint8 and labelled integer masks behave like their boolean view."""
    labeled = np.zeros((40, 40), np.int32)
    labeled[10:20, 10:20] = 7
    labeled[25:28, 25:28] = 3  # a second, separate region
    assert np.array_equal(
        mt.mask_poly_expand(labeled, 1.5),
        mt.mask_poly_expand(labeled != 0, 1.5),
    )


def test_regions_scale_about_their_own_centroids():
    """Two separated regions must each stay put; a shared pivot would shear them."""
    m = np.zeros((80, 80), bool)
    m[10:20, 10:20] = True
    m[50:56, 12:18] = True
    out = mt.mask_poly_expand(m, 2.0)
    for name, sl in (("upper", np.s_[0:35, :]), ("lower", np.s_[35:, :])):
        rows_in, cols_in = np.nonzero(m[sl])
        rows_out, cols_out = np.nonzero(out[sl])
        ci = (rows_in.mean(), cols_in.mean())
        co = (rows_out.mean(), cols_out.mean())
        assert np.allclose(ci, co, atol=0.5), f"{name} region drifted: {ci} -> {co}"


def test_holes_scale_with_the_outline():
    """A donut stays a donut: the hole doubles too, it is not filled in."""
    m = ring(outer=(10, 10), inner=(4, 4), pad=15, shape=(40, 40))
    out = mt.mask_poly_expand(m, 2.0)
    assert out.sum() == 4 * m.sum() == 336
    polys = mask_to_polygons(out)
    assert len(polys) == 1 and len(polys[0].interiors) == 1
    assert not out[16:24, 16:24].any(), "the scaled hole must stay open"
    assert out[10:15, 10:15].all(), "the scaled body must stay masked"


def test_growth_past_the_frame_edge_is_dropped_with_a_warning():
    m = np.zeros((20, 20), bool)
    m[15:20, 15:20] = True
    with pytest.warns(UserWarning, match="outside"):
        out = mt.mask_poly_expand(m, 4.0)
    assert out.shape == (20, 20)
    assert out.sum() < 16 * m.sum(), "clipped growth cannot keep its full area"


def test_padding_the_input_keeps_everything():
    """The documented workaround for the warning above."""
    m = np.zeros((20, 20), bool)
    m[8:12, 8:12] = True
    padded = np.pad(m, 20)
    out = mt.mask_poly_expand(padded, 4.0)
    assert out.sum() == 16 * m.sum()
    assert np.array_equal(out[20:-20, 20:-20], mt.mask_poly_expand(m, 4.0))


@pytest.mark.parametrize("bad", [0, -1, -0.5, np.nan, np.inf, -np.inf])
def test_bad_factor_raises(bad):
    with pytest.raises(ValueError, match="positive finite"):
        mt.mask_poly_expand(block(size=(4, 4)), bad)


def test_output_is_always_one_valid_mask_with_no_shape_leak():
    """Whatever the vector stage does, the raster comes back as one array."""
    data, mask, header = load_example()
    for factor in (0.5, 1.1, 1.5, 2.0, 2.5, 3.0):
        out = mt.mask_poly_expand(mask, factor)
        assert out.shape == mask.shape and out.dtype == bool


# --------------------------------------------------------------------------
# mask_manual
# --------------------------------------------------------------------------


def _fake_popup(action):
    """Stand-in for the window runner that never opens a window.

    *action* is ``"q"`` (close immediately, nothing drawn) or a callable handed
    the drawer, which is how a test "draws".
    """

    def runner(factory, figsize=(8, 8), title=""):
        ax = matplotlib.figure.Figure().add_subplot(111)
        drawer = factory(ax)
        if callable(action):
            action(drawer)
        else:
            drawer.close()
        return drawer

    return runner


def _draw_one(drawer):
    drawer.add_polygon([(2, 2), (9, 2), (9, 9)])
    drawer.close()


def test_requires_something_to_draw_on():
    with pytest.raises(ValueError, match="nothing to draw on"):
        mt.mask_manual()


def test_block_false_returns_the_drawer_for_programmatic_use(monkeypatch):
    monkeypatch.setattr(mt.api, "_run_in_popup", _fake_popup("q"))
    drawer = mt.mask_manual(shape=(30, 30), block=False, verbose=False)
    try:
        drawer.add_polygon([(2, 2), (9, 2), (9, 9)])
        m = drawer.mask()
        assert m.dtype == bool and m.shape == (30, 30)
        assert m.sum() == 28
    finally:
        drawer.teardown()


def test_popup_path_returns_the_drawn_mask(monkeypatch):
    monkeypatch.setattr(mt.api, "_run_in_popup", _fake_popup(_draw_one))
    out = mt.mask_manual(shape=(30, 30), verbose=False)
    assert out.dtype == bool and out.shape == (30, 30)
    assert out.sum() == 28


def test_existing_mask_is_kept_and_new_regions_are_added(monkeypatch):
    data, mask, header = load_example()
    monkeypatch.setattr(mt.api, "_run_in_popup", _fake_popup(_draw_one))
    out = mt.mask_manual(data, mask=mask, verbose=False)
    assert out.shape == data.shape
    assert np.all(out[mask]), "every originally-masked pixel stays masked"
    # The drawn triangle is far from the island, so the areas simply add up.
    assert not mask[2:10, 2:10].any()
    assert out.sum() == mask.sum() + 28


def test_closing_without_drawing_returns_the_input_unchanged(monkeypatch):
    """Aborting is lossless: 'q' with no new polygon gives the mask back."""
    data, mask, header = load_example()
    monkeypatch.setattr(mt.api, "_run_in_popup", _fake_popup("q"))
    assert np.array_equal(mt.mask_manual(data, mask=mask, verbose=False), mask)


def test_closing_without_drawing_and_no_mask_gives_an_empty_mask(monkeypatch):
    monkeypatch.setattr(mt.api, "_run_in_popup", _fake_popup("q"))
    out = mt.mask_manual(shape=(16, 24), verbose=False)
    assert out.shape == (16, 24) and out.dtype == bool and out.sum() == 0


def test_shape_comes_from_the_image(monkeypatch):
    data, mask, header = load_example()
    monkeypatch.setattr(mt.api, "_run_in_popup", _fake_popup(_draw_one))
    assert mt.mask_manual(data, verbose=False).shape == data.shape


def test_blank_canvas_gets_a_pixel_extent(monkeypatch):
    """With no array to look at, the clicks still mean pixel coordinates."""
    seen = {}

    def runner(factory, figsize=(8, 8), title=""):
        ax = matplotlib.figure.Figure().add_subplot(111)
        drawer = factory(ax)
        seen["xlim"], seen["ylim"] = ax.get_xlim(), ax.get_ylim()
        drawer.close()
        return drawer

    monkeypatch.setattr(mt.api, "_run_in_popup", runner)
    mt.mask_manual(shape=(40, 60), verbose=False)
    assert seen["xlim"] == pytest.approx((-0.5, 59.5))
    assert seen["ylim"] == pytest.approx((-0.5, 39.5))


def test_help_is_printed_unless_quiet(monkeypatch, capsys):
    monkeypatch.setattr(mt.api, "_run_in_popup", _fake_popup("q"))
    mt.mask_manual(shape=(10, 10))
    printed = capsys.readouterr().out
    assert "left click" in printed and "finish and close" in printed
    mt.mask_manual(shape=(10, 10), verbose=False)
    assert capsys.readouterr().out == ""


def test_unknown_options_are_not_swallowed(monkeypatch):
    monkeypatch.setattr(mt.api, "_run_in_popup", _fake_popup("q"))
    with pytest.raises(TypeError, match="not_a_drawer_option"):
        mt.mask_manual(shape=(10, 10), verbose=False, not_a_drawer_option=1)


@contextlib.contextmanager
def _no_tkinter():
    real = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in ("tkinter", "matplotlib.backends.backend_tkagg"):
            raise ImportError(f"blocked: {name}")
        return real(name, *args, **kwargs)

    builtins.__import__ = blocked
    try:
        yield
    finally:
        builtins.__import__ = real


def test_window_raises_a_clear_error_without_a_window_system():
    ax = matplotlib.figure.Figure().add_subplot(111)
    with _no_tkinter():
        with pytest.raises(RuntimeError, match="tkinter"):
            _run_in_popup(lambda a: a)


def test_window_raises_a_clear_error_when_tk_has_no_display(monkeypatch):
    """The headless-kernel case: tkinter imports, but Tk() itself refuses."""
    import tkinter as tk

    def refuse(*args, **kwargs):
        raise tk.TclError("no display name and no $DISPLAY environment variable")

    monkeypatch.setattr(tk, "Tk", refuse)
    with pytest.raises(RuntimeError, match="window system"):
        _run_in_popup(lambda a: a)


# --------------------------------------------------------------------------
# the real window, when there is a display to open one on
# --------------------------------------------------------------------------


def _display_available() -> bool:
    try:
        import tkinter as tk
    except Exception:
        return False
    try:
        root = tk.Tk()
    except Exception:
        return False
    root.withdraw()
    root.destroy()
    return True


@pytest.mark.skipif(not _display_available(), reason="needs a window system")
def test_real_window_opens_runs_and_terminates_by_itself():
    """End-to-end on the actual Tk path -- the one thing no fake can prove.

    The drawer schedules its own ``close`` through Tk's ``after``, so the window
    opens, runs its event loop, takes the click-free programmatic polygon, and
    shuts down without a human.  A watchdog destroys the root after four
    seconds so a regression here fails instead of hanging the suite.
    """

    def factory(ax):
        widget = ax.figure.canvas.get_tk_widget()
        root = widget.winfo_toplevel().master
        drawer = MaskDrawer(shape=(30, 30), ax=ax, verbose=False)
        drawer.add_polygon([(4, 4), (20, 4), (20, 20), (8, 22)])
        widget.after(20, drawer.close)
        widget.after(4000, root.destroy)
        return drawer

    drawer = _run_in_popup(factory)
    try:
        m = drawer.mask()
        assert m.shape == (30, 30) and m.dtype == bool
        assert m.sum() > 100, f"the polygon should have rasterised, got {m.sum()} px"
    finally:
        drawer.teardown()


# ------------------------------------------------------- the hang regression
# mask_manual could hang forever, silently, when the user closed the window
# after drawing.  close() committed the polygon in progress *before* arming the
# teardown, so any failure there skipped root.destroy(): mainloop() kept
# spinning, the window refused to close, and because the exception had happened
# inside a Tk event callback it went to Tk's background-error hook, whose output
# a notebook does not show.  Symptom: a dead panel and no traceback anywhere.
#
# Every test below presses the close button from INSIDE the event loop.  That is
# not stylistic: _run_in_popup registers WM_DELETE_WINDOW after factory() has
# returned, so a handler read during construction is matplotlib's, not ours, and
# the test would silently exercise nothing.  A watchdog destroys the root after
# four seconds so a regression fails instead of hanging the suite.


@pytest.mark.skipif(not _display_available(), reason="needs a window system")
def test_close_button_still_closes_when_the_commit_fails(monkeypatch):
    """The reported hang: draw, then press the window's close button."""

    def boom(self):
        raise RuntimeError("simulated redraw failure")

    monkeypatch.setattr(MaskDrawer, "_commit_live", boom)
    state = {}

    def factory(ax):
        widget = ax.figure.canvas.get_tk_widget()
        top, root = widget.winfo_toplevel(), widget.winfo_toplevel().master
        drawer = MaskDrawer(shape=(30, 30), ax=ax, verbose=False)
        drawer.add_polygon([(4, 4), (20, 4), (20, 20), (8, 22)])

        def press():
            cmd = top.protocol("WM_DELETE_WINDOW")
            state["cmd"], state["root"], state["top"] = cmd, root, top
            top.tk.call(cmd)  # exactly what the window manager does

        def watchdog():
            state["watchdog"] = True
            root.destroy()

        widget.after(30, press)
        widget.after(4000, watchdog)
        return drawer

    drawer = _run_in_popup(factory)
    try:
        assert not state.get("watchdog"), (
            "the close button did nothing: only the 4 s watchdog ended mainloop"
        )
        assert state["cmd"].endswith("_leave"), (
            f"pressed {state['cmd']!r}: the close button is not wired to the "
            "wrapper that guarantees the teardown"
        )
        assert drawer._closed is True, "mainloop returned but the drawer never closed"
        assert not _alive(state["root"]), "the window is still alive"
        assert not _alive(state["top"]), (
            "mainloop returned but the visible panel is still on screen -- the "
            "reported symptom: a dead window the user cannot get rid of"
        )
        assert drawer.mask().sum() > 100, "polygons drawn before the failure were lost"
    finally:
        monkeypatch.undo()
        drawer.teardown()


@pytest.mark.skipif(not _display_available(), reason="needs a window system")
def test_close_button_survives_a_catastrophic_close(monkeypatch):
    """Belt and braces: even a close() that cannot run must not strand mainloop."""

    def boom(self):
        raise RuntimeError("simulated total failure")

    monkeypatch.setattr(MaskDrawer, "close", boom)
    state = {}

    def factory(ax):
        widget = ax.figure.canvas.get_tk_widget()
        top, root = widget.winfo_toplevel(), widget.winfo_toplevel().master
        drawer = MaskDrawer(shape=(20, 20), ax=ax, verbose=False)

        def watchdog():
            state["watchdog"] = True
            root.destroy()

        def press():
            state["root"], state["top"] = root, top
            top.tk.call(top.protocol("WM_DELETE_WINDOW"))

        widget.after(30, press)
        widget.after(4000, watchdog)
        return drawer

    drawer = _run_in_popup(factory)  # returns instead of hanging
    monkeypatch.undo()               # so teardown() is not the sabotaged close
    assert not state.get("watchdog"), (
        "the close button did nothing: only the 4 s watchdog ended mainloop"
    )
    assert not _alive(state["root"]), "window survived a failed close()"
    assert not _alive(state["top"]), "the visible panel survived"
    drawer.teardown()


@pytest.mark.skipif(not _display_available(), reason="needs a window system")
def test_tk_callback_errors_are_reported_to_the_cell(capsys):
    """Nothing in the window may fail silently -- that silence was the bug."""

    seen = {}

    def factory(ax):
        widget = ax.figure.canvas.get_tk_widget()
        top, root = widget.winfo_toplevel(), widget.winfo_toplevel().master
        drawer = MaskDrawer(shape=(20, 20), ax=ax, verbose=False)

        def probe():
            hook = getattr(root, "report_callback_exception", None)
            seen["qualname"] = getattr(hook, "__qualname__", None)
            if hook is not None:
                hook(RuntimeError, RuntimeError("kaboom"), None)
            drawer.close()

        widget.after(30, probe)
        widget.after(4000, root.destroy)
        return drawer

    drawer = _run_in_popup(factory)
    try:
        assert str(seen.get("qualname", "")).startswith("_run_in_popup"), (
            f"Tk errors go to {seen.get('qualname')!r}: the default handler writes "
            "to stderr, which a notebook never shows, so a crash looks like a hang"
        )
        out = capsys.readouterr().out
        assert "unexpected error in the drawing window" in out and "kaboom" in out
    finally:
        drawer.teardown()


def _alive(widget_or_drawer) -> bool:
    """Whether a Tk toplevel still exists.

    A destroyed application does not return False from winfo_exists(), it raises
    TclError -- which is the success case for every teardown test here.
    """
    import tkinter as tk

    obj = widget_or_drawer
    if hasattr(obj, "fig"):
        try:
            obj = obj.fig.canvas.get_tk_widget().winfo_toplevel()
        except tk.TclError:
            return False
    try:
        return bool(obj.winfo_exists())
    except tk.TclError:
        return False
