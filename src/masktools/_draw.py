"""Draw polygon masks interactively, in a notebook or a popup window.

The prototype this package replaces (``pixsed.utils_interactive``) hand-rolled
its click handling: a list of pending vertices, a ``shift + left click``
binding, and a ``d`` key to close the shape.  That approach has three problems
this module avoids.

1. ``shift`` and ``ctrl`` are already taken by matplotlib's own
   :class:`~matplotlib.widgets.PolygonSelector` (move-all and move-vertex), so
   a hand-rolled binding silently collides with the tool it sits next to.
2. Redrawing the in-progress outline by hand means maintaining artists, and the
   prototype left the "discard the current drawing" path as a ``FIXME``.
3. Mutating the IPython ``%matplotlib`` magic from inside a library makes the
   notebook state unpredictable for everything else in the session.

So this module builds on :class:`matplotlib.widgets.PolygonSelector` and never
touches the shell's magic.  Vertex placement, dragging, deletion and the
bounding box come from matplotlib; this class adds only what it lacks --
multiple polygons, a rasterised mask, and file I/O.

Examples
--------
>>> from masktools._draw import MaskDrawer                       # doctest: +SKIP
>>> drawer = MaskDrawer(image, stretch='asinh')                 # doctest: +SKIP
>>> drawer.polygons                                             # doctest: +SKIP
>>> m = drawer.mask()                                           # doctest: +SKIP
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector as _BasePolygonSelector

from ._geometry import as_polygon_list
from ._plotting import plot_image, plot_mask
from ._vectorize import mask_to_polygons, polygons_to_mask

__all__ = ["MaskDrawer", "draw_mask"]

#: Keys reserved by PolygonSelector itself; :class:`MaskDrawer` never binds these.
RESERVED_KEYS = ("escape", "shift", "control")

_HELP = """\
Draw a mask with the mouse:
  left click          place a vertex
  click the 1st vertex close the polygon (needs >= 3 vertices)
  drag a vertex       move it            (or ctrl+drag)
  shift+drag          move the whole polygon
  right click a vertex  delete it
  esc                 restart the polygon currently being drawn

Keyboard:
  n   finish the current polygon and start a new one
  e   re-open the last finished polygon for editing
  d   delete the last finished polygon
  s   save the mask (needs a save path)
  q   finish and close the window
  Ctrl-C also closes the window, keeping what you drew
"""


def _unanchored() -> tuple[float, float]:
    """The pending vertex to start a polygon from: nowhere in particular.

    :class:`~matplotlib.widgets.PolygonSelector` always keeps one extra point
    at the end of its internal list, which follows the cursor and is dropped
    again by the ``verts`` getter.  Upstream parks that point at data
    ``(0, 0)``.  Until the cursor moves it is an ordinary finite coordinate, so
    the polyline is drawn *through* it: the window opened with the drawing
    tethered to the lower-left corner of the frame, and a gesture that places
    vertices by clicking (a trackpad tap sends no motion event at all) picked
    up a spurious corner at the origin.  NaN ends the line instead of reaching
    for the corner, so the first vertex lands exactly where it was clicked.
    """
    return (float("nan"), float("nan"))


class _RobustPolygonSelector(_BasePolygonSelector):
    """PolygonSelector that never tethers the drawing to the data origin.

    See :func:`_unanchored`: the mandatory pending vertex starts as NaN rather
    than ``(0, 0)``, both when the widget is built and after every ``clear()``,
    so nothing is ever drawn to the frame corner.

    It also cannot be broken by an Esc-then-modifier keypress.  Upstream,
    ``escape`` calls :meth:`clear`, which resets the internal vertex list to a
    single pending point.  Holding a modifier (shift/ctrl) then pops that point,
    and the *next* modifier press hits ``self._xys.pop()`` on an empty list --
    ``IndexError: pop from empty list`` -- which disconnects nothing but leaves
    the widget dead for the rest of the session.  Guarding the pop costs one
    comparison and keeps the drawing usable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._xys = [_unanchored()]

    def _clear_without_update(self):
        # clear() is the only supported reset; it re-parks the pending vertex at
        # the origin, so undo that before the artists are refreshed.
        super()._clear_without_update()
        self._xys = [_unanchored()]
        self._draw_polygon_without_update()

    def _press(self, event):
        # Nothing has been placed yet, so there is no tool handle to grab.
        # Upstream asks for the closest one anyway and takes argmin of an empty
        # array, which crashes the window as soon as a click lands while a move
        # modifier (ctrl/shift) is held down.
        if len(self._xys) < 2:
            self._xys_at_press = list(self._xys)
            return
        super()._press(event)

    def _on_key_press(self, event):
        # Upstream pops the pending vertex when a move modifier goes down.
        # Popping an already-empty list raises ``IndexError: pop from empty
        # list`` and leaves the widget dead, so skip the pop instead.
        if not self._xys:
            return
        super()._on_key_press(event)

    def _on_key_release(self, event):
        # Upstream puts the pending vertex back as ``(event.xdata,
        # event.ydata)``.  With the pointer off the axes those are ``None`` --
        # not a point -- and ``np.asarray(..., float)`` later turns them into
        # NaN, which shapely rejects with "Points of LinearRing do not form a
        # closed linestring".  Releasing Ctrl or Shift with the cursor off the
        # image is enough to do it.  Put back an unusable coordinate instead.
        if event.xdata is None or event.ydata is None:
            if event.key in (
                self._state_modifier_keys.get("move_vertex"),
                self._state_modifier_keys.get("move_all"),
            ):
                if not self._xys or np.isfinite(self._xys[-1]).all():
                    self._xys.append(_unanchored())
            elif not self._xys:
                self._xys = [_unanchored()]
            self._draw_polygon()
            return
        # A release with no matching press (Ctrl or Shift held while the window
        # opened) would leave the placeholder in front of the point upstream
        # appends; a ring that starts at that point cannot be built.
        if len(self._xys) == 1 and not np.isfinite(self._xys[0]).all():
            self._xys = []
        super()._on_key_release(event)
        if not self._xys:
            self._xys = [_unanchored()]


class MaskDrawer:
    """Interactively draw one or more polygons and turn them into a mask.

    The drawer keeps a *stash* of finished polygons plus at most one polygon
    still being drawn, and uses a single :class:`PolygonSelector` for both.
    This sidesteps matplotlib's multi-widget contention: ``set_active(False)``
    does not disconnect events, so two live selectors would both answer every
    click.

    Parameters
    ----------
    data : array-like, 2-D, optional
        Background image.  Pass it to draw against real pixels; without it the
        canvas is blank but still usable.
    mask : array-like, 2-D, optional
        An existing mask, shown for reference and used as the starting stash.
        Its polygons are imported, so you can refine an auto-generated mask by
        hand.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes instead of making a new figure.  Use this to
        embed the drawer in a multi-panel figure.
    figsize : tuple
        Figure size when creating one.  Ignored when *ax* is given.
    shape : (int, int), optional
        ``(nrow, ncol)`` assumed by :meth:`mask`.  Defaults to the *data* or
        *mask* shape; pass it to draw on a blank canvas with no array to look
        at.  Raises if it disagrees with *data* or *mask*.
    stretch, percent, vmin, vmax, asinh_a, min_percent, max_percent, cmap
        Passed to :func:`masktools._plotting.plot_image` for the background.
        The defaults are the plotting module's, i.e. an ``asinh`` stretch
        clipped at the 16th and 99.9th percentiles, which stays readable on a
        background-dominated frame.
    norm : matplotlib.colors.Normalize, optional
        Pre-built normalisation for the background.
    mask_color, poly_color : str
        Tint of the reference mask and of the polygon outlines.
    alpha : float
        Transparency of the reference mask.
    savepath : str, optional
        Where ``s`` writes the mask.  Also used by :meth:`save` when no path is
        given.
    verbose : bool
        Print a line to stdout whenever a polygon is committed or deleted.
    """

    def __init__(
        self,
        data=None,
        mask=None,
        shape=None,
        ax=None,
        figsize=(9, 9),
        stretch: str = "asinh",
        percent: float | None = None,
        vmin=None,
        vmax=None,
        asinh_a: float = 0.1,
        min_percent: float | None = 16.0,
        max_percent: float | None = 99.9,
        cmap: str = "Greys_r",
        norm=None,
        mask_color: str = "red",
        poly_color: str = "cyan",
        alpha: float = 0.35,
        savepath: str | None = None,
        verbose: bool = True,
    ):
        self._data = None if data is None else np.asarray(data, dtype=float)
        self._ref_mask = None if mask is None else np.asarray(mask) != 0
        self._shape = None if shape is None else _check_shape(shape)
        if self._shape is not None:
            for name, arr in (("data", self._data), ("mask", self._ref_mask)):
                if arr is not None and arr.shape != self._shape:
                    raise ValueError(
                        f"shape={self._shape} disagrees with the {name} array "
                        f"{arr.shape}; pass one or the other"
                    )
        self._mask_color = mask_color
        self._poly_color = poly_color
        self._alpha = alpha
        self._verbose = verbose
        self.savepath = savepath

        #: Finished polygons, as ``(N, 2)`` float arrays in pixel space.
        self._committed: list[np.ndarray] = []
        self._closed = False
        self._on_close_callbacks: list = []

        if mask is not None and np.any(self._ref_mask):
            # Import the reference mask as editable polygons.
            for poly in mask_to_polygons(self._ref_mask, space="pixel"):
                # Already pixel space, so no half-pixel adjustment here.
                xy = np.asarray(poly.exterior.coords, dtype=float)
                self._committed.append(xy[:-1] if _is_closed(xy) else xy)

        # --- figure / axes -------------------------------------------------
        created = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        self.ax = ax
        self.fig = fig

        if self._data is not None:
            plot_image(
                self._data,
                ax=ax,
                stretch=stretch,
                percent=percent,
                min_percent=min_percent,
                max_percent=max_percent,
                vmin=vmin,
                vmax=vmax,
                asinh_a=asinh_a,
                cmap=cmap,
                norm=norm,
            )
        elif created:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.set_title("click to add vertices; 'n' for a new polygon", fontsize=11)

        if self._ref_mask is not None:
            plot_mask(self._ref_mask, ax=ax, color=mask_color, alpha=alpha)

        # Keep the artists of committed polygons so we can redraw cheaply.
        self._stash_lines = []

        # --- the selector --------------------------------------------------
        useblit = _blit_ok(fig)
        self._selector = _RobustPolygonSelector(
            ax,
            onselect=self._on_select,
            useblit=useblit,
            props=dict(color=poly_color, linestyle="-", linewidth=1.5, alpha=0.9),
            handle_props=dict(
                marker="o", markersize=5, markeredgecolor=poly_color,
                markerfacecolor=poly_color,
            ),
        )
        self._latest: list[tuple[float, float]] | None = None

        # --- keys ----------------------------------------------------------
        self._cids = [
            fig.canvas.mpl_connect("key_press_event", self._on_key),
            fig.canvas.mpl_connect("close_event", self._on_window_close),
        ]
        # matplotlib garbage-collects widgets that nothing references; holding
        # the only reference from user code is the usual footgun, so pin ourselves
        # to the figure too.
        fig._masktools_drawer = self

    # ------------------------------------------------------------------ view
    @property
    def polygons(self) -> list:
        """All polygons: the finished ones plus the one being drawn, if any.

        Returns a list of shapely ``Polygon`` in ``"pixel"`` space (pixel
        centres on integers), ready for
        :func:`masktools._vectorize.polygons_to_mask`.
        """
        out = as_polygon_list(self.vertices, space="pixel")
        return out

    @property
    def vertices(self) -> list[np.ndarray]:
        """Raw vertex arrays of every polygon, finished and in-progress."""
        arrays = [_finite_xy(xy) for xy in self._committed]
        live = self._selector.verts
        if live:
            xy = _finite_xy(live)
            if len(xy) >= 2 and _is_closed(xy):
                xy = xy[:-1]
            if len(xy) >= 3:
                arrays.append(xy)
        return arrays

    def mask(self, shape=None) -> np.ndarray:
        """Rasterise the drawn polygons into a boolean mask.

        Parameters
        ----------
        shape : (int, int), optional
            ``(nrow, ncol)`` of the result.  Defaults to the background image's
            shape, or the reference mask's shape if no image was given.

        Raises
        ------
        ValueError
            If no shape is available and none is passed.
        """
        if shape is None:
            shape = self.default_shape
        if shape is None:
            raise ValueError(
                "no mask shape: this drawer has no background image or "
                "reference mask, so pass shape=(nrow, ncol) explicitly"
            )
        return polygons_to_mask(self.vertices, tuple(shape), space="pixel")

    @property
    def default_shape(self) -> tuple | None:
        """Shape assumed by :meth:`mask`.

        An explicit ``shape=`` wins, then the background image, then the
        reference mask.
        """
        if self._shape is not None:
            return self._shape
        if self._data is not None:
            return tuple(self._data.shape)
        if self._ref_mask is not None:
            return tuple(self._ref_mask.shape)
        return None

    # ----------------------------------------------------------- mutation
    def add_polygon(self, xy):
        """Append a polygon programmatically.

        Lets a script (or a test) build the same result a mouse would, with no
        event loop involved.  Polygons are implicitly closed on rasterisation.

        Parameters
        ----------
        xy : array-like, ``(N, 2)``
            Vertices in pixel space.  ``N >= 3``.
        """
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"xy must be (N, 2), got shape {xy.shape}")
        if _is_closed(xy):
            xy = xy[:-1]
        if len(xy) < 3:
            raise ValueError(f"a polygon needs >= 3 vertices, got {len(xy)}")
        self._committed.append(xy.copy())
        self._redraw_stash()
        return self

    def clear(self):
        """Discard every polygon, finished and in-progress."""
        self._committed = []
        self._reset_selector()
        self._redraw_stash()
        return self

    def delete_last(self):
        """Remove the most recently finished polygon.  Returns it or `None`."""
        if not self._committed:
            self._log("nothing to delete")
            return None
        gone = self._committed.pop()
        self._redraw_stash()
        self._log(f"deleted polygon with {len(gone)} vertices")
        return gone

    def edit_last(self):
        """Re-open the most recent finished polygon in the selector.

        Its vertices move back into the live selector so matplotlib's own
        drag/delete handles apply; the stash entry is removed.
        """
        if not self._committed:
            self._log("nothing to edit")
            return None
        xy = self._committed.pop()
        self._selector.verts = [tuple(p) for p in xy]  # type: ignore[assignment]
        self._redraw_stash()
        self._log("re-opened the last polygon for editing")
        return xy

    # ------------------------------------------------------------- I/O
    def save(self, path=None, shape=None, header=None, overwrite: bool = True):
        """Write the mask to *path* as ``.npy`` or FITS.

        Parameters
        ----------
        path : str, optional
            Destination; defaults to the ``savepath`` given at construction.
            A ``.npy`` extension writes a numpy file (no extra dependency);
            anything else is FITS and needs the optional ``astropy`` extra.
        shape : (int, int), optional
            Mask shape; defaults to :attr:`default_shape`.
        header : `~astropy.io.fits.Header`, optional
            Header to attach to the FITS mask.  Pass the image's own header
            (``fits.getheader(image)``) and the WCS travels with the mask, so
            it stays aligned with the frame you drew on.
        overwrite : bool
            Replace an existing file.

        Returns
        -------
        str
            The path written.
        """
        path = path or self.savepath
        if path is None:
            raise ValueError("no save path: pass one or set savepath=...")
        if not overwrite and os.path.exists(path):
            raise FileExistsError(f"{path} exists (pass overwrite=True)")
        arr = self.mask(shape)
        if str(path).endswith(".npy"):
            np.save(path, arr)
        else:
            try:
                from astropy.io import fits
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "saving FITS needs astropy: pip install 'astro-mask-tools[fits]' "
                    "or use a .npy path"
                ) from exc
            hd = fits.Header(header) if header is not None else None
            fits.PrimaryHDU(arr.astype(np.uint8), header=hd).writeto(
                path, overwrite=True
            )
        self._log(f"saved mask to {path}")
        return path

    def on_close(self, func):
        """Register *func(drawer)* to run when the window closes."""
        self._on_close_callbacks.append(func)
        return func

    # -------------------------------------------------------- event loop
    def show(self, block: bool = True):
        """Display the figure, blocking until the window closes.

        In a Jupyter notebook with ``%matplotlib widget`` this returns
        immediately, because the inline figure is already live; call
        :meth:`mask` once you are done drawing.  In a popup backend
        (``macosx``, ``TkAgg``, ``QtAgg``) this blocks, which is what makes the
        drawer usable from a plain script.

        On a non-interactive backend (``Agg``, the default when no display is
        available) there is no window to draw in, so this warns and returns
        straight away rather than appearing to hang.  Pick a GUI backend first::

            matplotlib.use("TkAgg")   # or "macosx" / "QtAgg"

        or, in a notebook, run ``%matplotlib widget`` before constructing the
        drawer.
        """
        if not _is_interactive(self.fig):
            warnings.warn(
                "the current matplotlib backend "
                f"({self.fig.canvas.__class__.__name__}) is not interactive, so "
                "no drawing window will appear. Use a GUI backend "
                "('TkAgg', 'macosx', 'QtAgg') or run '%matplotlib widget' in a "
                "notebook; the polygons added with add_polygon() still work.",
                stacklevel=2,
            )
            return self
        plt.show(block=block)
        return self

    #: Alias, so ``drawer.join()`` reads naturally in scripts.
    wait = show

    def close(self):
        """Finish the drawing: commit the live polygon and run close callbacks.

        This never raises.  Committing the last polygon is ordinary drawing work
        -- it mutates the selector and redraws artists -- and it used to run
        unprotected ahead of the callbacks below.  When it failed, ``_closed``
        stayed `False` and the close callbacks never ran, so the popup's
        ``root.destroy()`` was never scheduled: ``mainloop()`` kept spinning, the
        window refused to close, and because the exception had happened inside a
        Tk event callback it went to the kernel's stderr, which a notebook never
        shows.  The symptom was a silent, permanent hang.  So the commit is
        best-effort now, and the callbacks are unconditional.

        The figure window itself is left for :meth:`teardown` (or the caller's
        own ``plt.close``) so that a callback can still read from the axes.
        """
        if self._closed:
            return
        self._closed = True  # first, so re-entry cannot commit twice
        try:
            self._commit_live()
        except Exception as exc:
            print(f"[masktools] could not finish the polygon in progress: {exc!r}")
        try:
            for cid in self._cids:
                self.fig.canvas.mpl_disconnect(cid)
            self._cids = []
            self._selector.disconnect_events()
        except Exception as exc:  # pragma: no cover - backend dependent
            print(f"[masktools] while releasing event handlers: {exc!r}")
        for func in list(self._on_close_callbacks):
            try:
                func(self)
            except Exception as exc:  # pragma: no cover - interactive feedback
                print(f"[masktools] on_close callback failed: {exc!r}")

    def teardown(self):
        """Release every artist and connection, then close the figure.

        Safe to call twice, and on a drawer that was already closed.  Tests and
        long-running scripts should call it to avoid leaking figures.
        """
        self.close()
        for line in self._stash_lines:
            try:
                line.remove()
            except Exception:  # pragma: no cover
                pass
        self._stash_lines = []
        try:
            self.fig._masktools_drawer = None
        except Exception:  # pragma: no cover
            pass
        plt.close(self.fig)

    # ------------------------------------------------------------ internals
    def _on_select(self, verts):
        """Called by PolygonSelector on completion *and* on every later edit."""
        self._latest = list(verts)
        # The selector keeps ownership of these vertices until 'n' commits
        # them, so we only refresh the display here.
        self._redraw_stash()
        n = len(verts) - 1 if _is_closed(np.asarray(verts, dtype=float)) else len(verts)
        self._log(f"polygon live with {n} vertices -- press 'n' to finish it")

    def _on_key(self, event):
        """Handle the drawer's own keys, then defer to matplotlib's."""
        if event.key is None:
            return
        # Accept keys aimed at our own axes, and untargeted ones (a bare
        # keypress has inaxes=None whenever the cursor is off the axes, and
        # gating on that would make 'q'/'s' unreliable).  Only keys aimed at a
        # *different* axes on a shared figure are ignored.
        if event.inaxes is not None and event.inaxes is not self.ax:
            return
        key = event.key
        if key in RESERVED_KEYS:
            # escape/shift/control belong to PolygonSelector.
            return
        if key == "n":
            self._commit_live()
        elif key == "e":
            self.edit_last()
        elif key in ("d", "backspace", "delete"):
            self.delete_last()
        elif key == "s":
            if self.savepath is None:
                print(
                    "[masktools] no save path was set for this window; save the "
                    "mask from your own code instead:\n"
                    "    m = mask_manual(image)\n"
                    "    np.save('mask.npy', m)"
                )
                return
            try:
                self.save()
            except Exception as exc:  # pragma: no cover - interactive feedback
                print(f"[masktools] save failed: {exc}")
        elif key == "h":
            print(_HELP)
        elif key in ("q", "c"):
            self.close()  # close() commits the polygon in progress
        else:
            key_press_handler(event, event.canvas, getattr(event, "manager", None))

    def _reset_selector(self):
        """Put the selector back into the "nothing drawn" state.

        ``selector.verts = []`` looks like the obvious way but raises
        IndexError, because the setter builds a *closed* polygon from its
        argument and indexes ``xys[0]``.  ``clear()`` is the supported reset.
        """
        self._selector.clear()

    def _commit_live(self):
        """Move the selector's polygon into the stash and reset it."""
        verts = self._selector.verts
        if not verts:
            self._log("no polygon to finish")
            return
        xy = _finite_xy(verts)
        if _is_closed(xy):
            xy = xy[:-1]
        if len(xy) < 3:
            self._log(f"need >= 3 vertices, have {len(xy)}; polygon discarded")
            self._reset_selector()
            return
        self._committed.append(xy)
        self._reset_selector()
        self._redraw_stash()
        self._log(f"finished polygon {len(self._committed)} ({len(xy)} vertices)")

    def _redraw_stash(self):
        """Redraw the committed polygons."""
        for line in self._stash_lines:
            try:
                line.remove()
            except Exception:  # pragma: no cover - canvas may be going away
                pass
        self._stash_lines = []
        if not self._committed:
            try:
                self.fig.canvas.draw_idle()
            except Exception:  # pragma: no cover
                pass
            return
        try:
            for xy in self._committed:
                (line,) = self.ax.plot(
                    np.r_[xy[:, 0], xy[0, 0]],
                    np.r_[xy[:, 1], xy[0, 1]],
                    color=self._poly_color,
                    lw=1.2,
                    ls=":",
                    alpha=0.9,
                )
                self._stash_lines.append(line)
            self.fig.canvas.draw_idle()
        except Exception:  # pragma: no cover - headless oddities
            pass

    def _on_window_close(self, event):
        """Window manager asked us to go away: finish the drawing.

        This runs from matplotlib's ``close_event``, i.e. *while the canvas is
        being torn down*.  An exception escaping here aborts that teardown, so
        the Tk window survives and ``mainloop()`` never returns -- another way
        to reach the same silent hang.  Report and carry on.
        """
        try:
            self.close()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[masktools] error while closing the window: {exc!r}")

    def _log(self, message):
        if self._verbose:
            print(f"[masktools] {message}")

    def __repr__(self):
        live = len(self._selector.verts or [])
        return (
            f"<MaskDrawer polygons={len(self._committed)} live_vertices={live} "
            f"shape={self.default_shape}>"
        )

    def __del__(self):  # pragma: no cover - interpreter shutdown
        try:
            if hasattr(self, "fig") and self.fig is not None:
                self.fig._masktools_drawer = None
        except Exception:
            pass


def _check_shape(shape) -> tuple:
    """Validate and normalise a user-supplied ``(nrow, ncol)``."""
    arr = tuple(int(v) for v in shape)
    if len(arr) != 2 or any(v <= 0 for v in arr):
        raise ValueError(f"shape must be (nrow, ncol) with positive ints, got {shape!r}")
    return arr


def _is_interactive(fig) -> bool:
    """Whether *fig* is backed by a real window we can draw a widget in.

    Uses matplotlib's own test (``FigureManagerBase.show`` checks the same
    attribute before warning), so the drawer's idea of "interactive" cannot
    drift from the library's.  Note that ``fig.manager`` is *not* the test: it
    is set even under Agg.
    """
    manager = getattr(fig, "manager", None)
    return manager is not None and hasattr(fig.canvas, "switch_backends")


def _is_closed(xy) -> bool:
    """True if the first and last vertex of *xy* coincide."""
    xy = np.asarray(xy, dtype=float)
    return len(xy) > 2 and np.array_equal(xy[0], xy[-1])


def _finite_xy(xy) -> np.ndarray:
    """Return *xy* as an ``(N, 2)`` float array without unusable vertices.

    A vertex can end up as ``(None, None)`` -- a modifier key released while the
    pointer is off the axes -- which becomes NaN on conversion.  A ring that
    starts with NaN is rejected by shapely ("Points of LinearRing do not form a
    closed linestring"), so such vertices are dropped rather than passed on.
    """
    arr = np.asarray(xy, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 2)
    return arr[np.isfinite(arr).all(axis=1)]


def _blit_ok(fig) -> bool:
    """Whether blitting is safe on this figure's canvas.

    Blitting breaks the notebook backends (ipympl redraws the figure itself)
    and does nothing on the non-interactive Agg, so it is enabled only for real
    GUI canvases.
    """
    from matplotlib.backend_bases import FigureManagerBase

    name = fig.canvas.__class__.__name__
    if "NbAgg" in name or name == "FigureCanvasAgg":
        return False
    # fig.manager is absent (not merely None) until a manager is created, which
    # happens on the first plt.show()/plt.subplots() in a GUI process.
    manager = getattr(fig, "manager", None)
    return isinstance(manager, FigureManagerBase)


_POPUP_TITLE = "masktools -- draw a mask (q when done, h for help)"

#: The one Tk interpreter this process uses, created on first use.
_TCL_ROOT = None


def _report_callback_error(exc, val, tb) -> None:
    """Report an exception raised inside a Tk callback.

    Tk's own handler writes to stderr, which a Jupyter notebook does not show,
    so a failure while closing used to look exactly like a hang.  Print into
    the cell's own stream instead: visible, and still not fatal.
    """
    import traceback

    print("[masktools] unexpected error in the drawing window:", flush=True)
    traceback.print_exception(exc, val, tb, file=sys.stdout)
    sys.stdout.flush()


def _tcl_root(tk):
    """Return this process's Tk interpreter, creating it on first use.

    Creating and destroying a Tk interpreter on every call is what makes a
    long-lived kernel die: on macOS the interpreter is torn down while
    matplotlib still holds canvas objects for it, and Tcl aborts
    (``Tcl_AsyncDelete: async handler deleted by the wrong thread``).  One
    interpreter per process, with a fresh ``Toplevel`` per window, is the
    supported pattern; ``mainloop`` is ended with ``quit()`` and the
    interpreter is left alone.
    """
    global _TCL_ROOT
    root = _TCL_ROOT
    if root is not None:
        try:
            if root.winfo_exists():
                return root
        except Exception:  # pragma: no cover - interpreter went away
            pass
    root = tk.Tk()
    root.withdraw()  # the visible window is a Toplevel child of this root
    root.report_callback_exception = _report_callback_error
    _TCL_ROOT = root
    return root



def _run_in_popup(factory, figsize=(8, 8), title: str = _POPUP_TITLE):
    """Run *factory(ax)* inside a dedicated window and block until it closes.

    The window is deliberately **not** a pyplot figure: it is a bare
    :class:`~matplotlib.figure.Figure` hung on a
    :class:`~matplotlib.backends.backend_tkagg.FigureCanvasTkAgg`.  That is
    what lets the drawing work from a Jupyter kernel whatever backend or
    ``%matplotlib`` magic the session uses -- the kernel cannot process widget
    events while a cell is blocked, but a Tk window runs its own event loop, so
    one call can wait for the mouse and still return a finished mask.  The
    caller's own inline plots are left untouched.

    There is no navigation toolbar, on purpose: its pan/zoom bindings fight
    :class:`~matplotlib.widgets.PolygonSelector` for the left mouse button.

    Parameters
    ----------
    factory : callable
        ``factory(ax)`` is called once the canvas exists, and must build a
        drawer (or anything else) that accepts an ``ax`` and exposes
        :meth:`MaskDrawer.close` and :meth:`MaskDrawer.on_close`.
    figsize : (float, float)
        Window size in inches.
    title : str
        Window title.

    Returns
    -------
    object
        Whatever *factory* returned, after the window has closed.

    Raises
    ------
    RuntimeError
        If tkinter is missing or no window system is reachable.
    """
    try:
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except Exception as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            f"the drawing window needs tkinter, which could not be loaded ({exc!r}). "
            "Install tkinter, or build the mask without a window: "
            "MaskDrawer(...).add_polygon(...)."
        ) from exc

    fig = Figure(figsize=figsize)
    root = None
    window = None
    drawer = None
    try:
        root = _tcl_root(tk)
        window = tk.Toplevel(root)
        window.title(title)
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        ax = fig.add_subplot(111)
        drawer = factory(ax)

        loop_running = [False]

        def _end_loop():
            # Deferred: Tk forbids tearing down a window from inside its own
            # event callback.  quit() ends mainloop() without destroying the
            # interpreter, which is reused by the next call.  Only schedule it
            # while a loop is actually running: a quit left pending from a
            # window that was already closed would end the *next* window's
            # loop the moment it starts.
            if loop_running[0]:
                root.after(0, root.quit)

        def _shutdown(_drawer):
            _end_loop()

        def _leave():
            # The window-manager close button.  close() is now exception-safe,
            # but the teardown must not depend on that holding for every future
            # edit, so ending the loop is unconditional.
            try:
                drawer.close()
            finally:
                _end_loop()

        drawer.on_close(_shutdown)
        # The window-manager close button means the same thing as 'q'.
        window.protocol("WM_DELETE_WINDOW", _leave)

        # Ctrl-C must not freeze the window.  A KeyboardInterrupt raised while
        # the main thread sits inside Tcl's event loop does not unwind it: Tk
        # takes the exception in the middle of its own dispatch, stops
        # servicing events, and mainloop() never returns -- the window goes
        # dead (it ignores the mouse, 'q' and its own close button) and the
        # cell stays blocked for the rest of the kernel's life.  So SIGINT is
        # intercepted for the duration and treated like the close button,
        # which still returns the polygons finished so far.
        stop_requested = []
        previous_sigint = None

        def _request_stop(*_args):
            stop_requested.append(True)

        if threading.current_thread() is threading.main_thread():
            try:
                previous_sigint = signal.signal(signal.SIGINT, _request_stop)
            except (ValueError, OSError):  # pragma: no cover - odd platforms
                previous_sigint = None

        def _poll_for_stop():
            if stop_requested:
                _leave()
                return
            if getattr(drawer, "_closed", False):
                return  # the window is already on its way out
            root.after(100, _poll_for_stop)

        root.after(100, _poll_for_stop)
        canvas.draw()
        loop_running[0] = True
        try:
            root.mainloop()
        finally:
            loop_running[0] = False
            if previous_sigint is not None:
                try:
                    signal.signal(signal.SIGINT, previous_sigint)
                except (ValueError, OSError):  # pragma: no cover
                    pass
    except tk.TclError as exc:  # pragma: no cover - needs a real display
        raise RuntimeError(
            f"could not open a drawing window ({exc}). This needs a window "
            "system, so on a remote or headless kernel build the mask without "
            "one: MaskDrawer(...).add_polygon(...)."
        ) from exc
    finally:
        # Tear down the window, never the interpreter: the interpreter is
        # shared with every later call in this process (see _tcl_root).
        if window is not None:
            try:
                window.destroy()
            except tk.TclError:  # pragma: no cover - already gone
                pass
    return drawer


def draw_mask(
    data=None,
    mask=None,
    shape=None,
    block: bool = True,
    **kwargs,
) -> np.ndarray | None:
    """One-call convenience: open a drawer, let the user draw, return a mask.

    Parameters
    ----------
    data : array-like, 2-D, optional
        Background image.
    mask : array-like, 2-D, optional
        Existing mask to refine.
    shape : (int, int), optional
        Shape of the returned mask; defaults to *data* or *mask*.  Pass it to
        draw on a blank canvas; it becomes the drawer's ``default_shape``, so
        the non-blocking path returns a drawer that can rasterise on its own.
    block : bool
        Wait for the window to close before returning.  Set `False` in a
        notebook with ``%matplotlib widget``, in which case the drawer is
        returned instead of a mask so you can poll it yourself.
    **kwargs
        Forwarded to :class:`MaskDrawer`.

    Returns
    -------
    numpy.ndarray or MaskDrawer or None
        The boolean mask when *block* is `True` and something was drawn;
        the :class:`MaskDrawer` itself when *block* is `False`; `None` if the
        user closed the window without drawing anything.

    Examples
    --------
    >>> from masktools import draw_mask              # doctest: +SKIP
    >>> m = draw_mask(image)                         # doctest: +SKIP
    """
    drawer = MaskDrawer(data=data, mask=mask, shape=shape, **kwargs)
    if not block:
        return drawer
    drawer.show(block=True)
    if not drawer.vertices:
        return None
    return drawer.mask(shape)
