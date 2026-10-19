"""Display helpers: images with astronomical stretches, masks, and polygons.

Everything here is built on matplotlib only, so the package keeps no hard
dependency on astropy.  The stretch options mirror what astronomers expect from
``astropy.visualization.simple_norm`` (linear / sqrt / asinh / log with a
percentile clip) because a raw ``imshow`` of an astronomical frame is almost
always useless: the sky background dominates and every source vanishes.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize

from ._geometry import HALF_PIXEL, as_polygon_list, to_space
from .vectorize import mask_to_polygons

__all__ = [
    "make_norm",
    "plot_image",
    "plot_mask",
    "plot_polygons",
    "plot_mask_contours",
    "show_mask",
]

STRETCHES = ("linear", "sqrt", "asinh", "log")


def _as_ax(ax, figsize=(7, 7)) -> Axes:
    """Return *ax*, creating a figure first if it is `None`."""
    if ax is not None:
        return ax
    fig, ax = plt.subplots(figsize=figsize)
    return ax


def make_norm(
    data,
    stretch: str = "asinh",
    percent: float | None = None,
    vmin=None,
    vmax=None,
    asinh_a: float = 0.1,
    min_percent: float | None = 16.0,
    max_percent: float | None = 99.9,
) -> Normalize | LogNorm:
    """Build a matplotlib normalisation for an astronomical image.

    Parameters
    ----------
    data : array-like
        The image, used to derive default limits from its percentiles.
        Non-finite pixels are ignored.
    stretch : {'linear', 'sqrt', 'asinh', 'log'}
        Intensity mapping applied between *vmin* and *vmax*.
    percent : float, optional
        Symmetric percentile clipping, *identical to*
        ``astropy.visualization.simple_norm(percent=...)``: ``vmin`` becomes the
        ``[(100 - percent) / 2]``-ile and ``vmax`` the
        ``[100 - (100 - percent) / 2]``-ile.  Given the choice, prefer
        *min_percent*/*max_percent* -- see their notes for why symmetric
        clipping fails on real frames.  ``percent=99.9`` reproduces astropy
        exactly if you need a pixel-for-pixel match with an existing notebook.
    min_percent, max_percent : float, optional
        Asymmetric clipping: ``vmin`` = the *min_percent*-ile, ``vmax`` = the
        *max_percent*-ile.  Defaults to ``16`` and ``99.9`` (roughly "sky minus
        one sigma" to "sky plus 3.5 sigma"), which is what makes faint sources
        readable on a background-dominated frame.  Same keyword names as
        astropy's ``simple_norm``, but here they are the default rather than an
        alternative, because symmetric clipping is actively misleading on
        astronomical data: a few hundred pathological pixels -- a zeroed or
        saturated region, a bad column, a masked island -- sit far below sky and
        drag a symmetric low cut down to themselves, flattening every source in
        the frame.  On the shipped cutout, 582 zero pixels (0.22% of the frame)
        pull ``percent=99.9``'s ``vmin`` from 487 to 0, and the image turns grey.

    Precedence
    ----------
    Explicit *vmin*/*vmax* win over *percent*, which wins over
    *min_percent*/*max_percent*, which win over the data's plain min/max.
    vmin, vmax : float, optional
        Explicit limits, overriding *percent*.
    asinh_a : float
        For ``stretch="asinh"``, the width of the linear region expressed as a
        *fraction of the plotted range* -- the same meaning it has in
        ``astropy.visualization.simple_norm``, whose default it matches.

    Returns
    -------
    matplotlib.colors.Normalize
        A ``Normalize``, ``PowerNorm``, ``AsinhNorm`` or ``LogNorm`` as
        appropriate, always with ``clip=True``.

    Notes
    -----
    The asinh stretch is built with ``FuncNorm`` rather than matplotlib's
    ``AsinhNorm``.  ``AsinhNorm`` only exists on matplotlib >= 3.8, is
    documented as provisional, and in 3.10 ignores its own ``linear_width``
    argument (every width gives the same curve), so it cannot express astropy's
    ``asinh_a``.  Going through ``FuncNorm`` matches
    ``astropy.visualization.simple_norm`` exactly on every version.
    """
    stretch = stretch.lower()
    if stretch not in STRETCHES:
        raise ValueError(f"stretch must be one of {STRETCHES}, got {stretch!r}")

    arr = np.asarray(data, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("data has no finite pixels; cannot build a normalisation")
    flat = arr[finite]

    if percent is not None:
        # Same symmetric rule as astropy's simple_norm, so asking for it
        # explicitly reproduces an existing notebook exactly.
        half = (100.0 - float(percent)) / 2.0
        lo = float(np.percentile(flat, half))
        hi = float(np.percentile(flat, 100.0 - half))
    elif min_percent is not None or max_percent is not None:
        lo = float(np.percentile(flat, 0.1 if min_percent is None else min_percent))
        hi = float(np.percentile(flat, 99.9 if max_percent is None else max_percent))
    else:
        lo, hi = float(flat.min()), float(flat.max())
    if vmin is not None:
        lo = float(vmin)
    if vmax is not None:
        hi = float(vmax)
    if not hi > lo:
        # A perfectly flat image (or a pathological clip) would make the
        # normalisation divide by zero; widen by one unit instead.
        hi = lo + 1.0

    if stretch == "linear":
        return Normalize(vmin=lo, vmax=hi, clip=True)
    if stretch == "sqrt":
        return mcolors.PowerNorm(gamma=0.5, vmin=lo, vmax=hi, clip=True)
    if stretch == "log":
        # LogNorm cannot represent zero or negative pixels, so lift the floor to
        # the smallest positive value available.
        positive = flat[flat > 0]
        floor = hi / 1e6 if positive.size == 0 else max(float(positive.min()), 1e-30)
        return mcolors.LogNorm(vmin=max(lo, floor), vmax=hi, clip=True)

    # asinh: arcsinh(u/a) / arcsinh(1/a), with u the value scaled to [0, 1] and
    # a = asinh_a a fraction of the plotted range -- term for term the same
    # curve as astropy's AsinhStretch.
    a = max(float(asinh_a), 1e-12)
    width = a * (hi - lo)
    fwd = lambda v: np.arcsinh((v - lo) / width)
    inv = lambda f: lo + width * np.sinh(f)
    return mcolors.FuncNorm((fwd, inv), vmin=lo, vmax=hi, clip=True)


def plot_image(
    data,
    ax=None,
    figsize=(7, 7),
    stretch: str = "asinh",
    percent: float | None = None,
    vmin=None,
    vmax=None,
    asinh_a: float = 0.1,
    min_percent: float | None = 16.0,
    max_percent: float | None = 99.9,
    cmap: str = "Greys_r",
    origin: str = "lower",
    norm=None,
    **imshow_kwargs,
) -> Axes:
    """Show an image with an astronomical stretch.

    Parameters
    ----------
    data : array-like, 2-D
        The image.  Non-finite pixels are rendered transparent rather than
        breaking the colour map.
    ax : matplotlib.axes.Axes, optional
        Target axes; a new figure is made when absent.
    figsize : tuple
        Figure size, used only when *ax* is `None`.
    stretch, percent, vmin, vmax, asinh_a, min_percent, max_percent
        See :func:`make_norm`.
    cmap : str
        Colour map name.
    origin : {'lower', 'upper'}
        Passed to ``imshow``.  ``"lower"`` (default) puts row 0 at the bottom,
        matching the pixel-coordinate convention the rest of the package uses.
    norm : matplotlib.colors.Normalize, optional
        Pre-built normalisation, bypassing *stretch*/*percent*.
    **imshow_kwargs
        Extra arguments for ``imshow``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"data must be 2-D, got shape {arr.shape}")

    ax = _as_ax(ax, figsize)
    if norm is None:
        norm = make_norm(
            arr,
            stretch,
            percent,
            vmin,
            vmax,
            asinh_a,
            min_percent=min_percent,
            max_percent=max_percent,
        )

    # Masked-out (non-finite) pixels become transparent so an overlay underneath
    # stays visible instead of turning black.
    disp = np.ma.masked_invalid(arr)
    ax.imshow(disp, origin=origin, cmap=cmap, norm=norm, **imshow_kwargs)
    ax.set_aspect("equal", adjustable="box")
    return ax


def _extent_for(shape_, origin="lower"):
    """Extent that draws an array of *shape* with pixel centres on integers."""
    nrow, ncol = shape_
    if origin == "lower":
        return [-0.5, ncol - 0.5, -0.5, nrow - 0.5]
    return [-0.5, ncol - 0.5, nrow - 0.5, -0.5]


def plot_mask(
    mask,
    ax=None,
    data=None,
    color: str = "red",
    alpha: float = 0.35,
    origin: str = "lower",
    cmap=None,
    **imshow_kwargs,
) -> Axes:
    """Overlay a boolean mask, optionally on top of *data*.

    Parameters
    ----------
    mask : array-like, 2-D
        The mask; nonzero is shown.
    ax : matplotlib.axes.Axes, optional
        Target axes.
    data : array-like, 2-D, optional
        Underlying image, shown with :func:`plot_image`.
    color : str
        Tint for the masked pixels, used when *cmap* is `None`.
    alpha : float
        Transparency of the overlay.
    origin : {'lower', 'upper'}
        Same meaning as in :func:`plot_image`.
    cmap : str, optional
        Colour map for the overlay instead of a flat *color*.
    **imshow_kwargs
        Extra arguments for the overlay ``imshow``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    arr = np.asarray(mask)
    ax = _as_ax(ax)
    if data is not None:
        plot_image(data, ax=ax, origin=origin)

    tint = mcolors.to_rgba(color)
    if cmap is None:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "masktools", [(0, 0, 0, 0), (*tint[:3], 1.0)]
        )
    shown = np.ma.masked_where(arr == 0, arr != 0)
    kwargs = dict(imshow_kwargs)
    kwargs.setdefault("vmin", 0)
    kwargs.setdefault("vmax", 1)
    ax.imshow(
        shown,
        origin=origin,
        extent=_extent_for(arr.shape, origin),
        cmap=cmap,
        alpha=alpha,
        interpolation="nearest",
        **kwargs,
    )
    return ax


def plot_polygons(
    polygons,
    ax=None,
    space: str = "pixel",
    edgecolor: str = "cyan",
    facecolor: str | None = None,
    fill: bool = False,
    lw: float = 1.0,
    labels: bool = False,
    **plot_kwargs,
) -> Axes:
    """Draw polygons on *ax*.

    Parameters
    ----------
    polygons : Polygon-like or list of it
        As accepted by
        :func:`masktools.vectorize.polygons_to_mask`.
    ax : matplotlib.axes.Axes, optional
        Target axes.
    space : {'pixel', 'corner'}
        Coordinate convention of *polygons*; the outline is shifted by 0.5 when
        needed so that it always lands on pixel centres on screen.
    edgecolor : str
        Outline colour.
    facecolor : str, optional
        Fill colour; only used when *fill* is `True`.
    fill : bool
        Fill the interiors (with ``fill=False`` the holes stay visible as gaps).
    lw : float
        Linewidth.
    labels : bool
        Annotate each polygon with its index at its centroid.
    **plot_kwargs
        Extra arguments for ``plot`` / ``add_patch``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from matplotlib.patches import PathPatch

    ax = _as_ax(ax)
    # as_polygon_list resolves *space* into canonical corner space; each ring is
    # then converted for drawing, so outlines sit on pixel centres.
    polys = as_polygon_list(polygons, space=space)

    for i, poly in enumerate(polys):
        for j, ring in enumerate([poly.exterior, *poly.interiors]):
            xy = to_space(np.asarray(ring.coords, dtype=float), "pixel")
            style = dict(color=edgecolor, lw=lw, **plot_kwargs)
            if j > 0:
                style["ls"] = "--"  # holes, so they read as gaps not new regions
            ax.plot(xy[:, 0], xy[:, 1], **style)

        if fill:
            ax.add_patch(
                PathPatch(
                    _poly_to_path(poly),
                    facecolor=facecolor if facecolor is not None else edgecolor,
                    alpha=0.3,
                    edgecolor=edgecolor,
                    lw=lw,
                )
            )
        if labels:
            c = poly.centroid
            ax.text(
                c.x - HALF_PIXEL,
                c.y - HALF_PIXEL,
                str(i),
                color=edgecolor,
                fontsize=9,
                ha="center",
                va="center",
            )
    return ax


def _poly_to_path(poly):
    """Convert a shapely Polygon (corner space) to a matplotlib Path."""
    from matplotlib.path import Path

    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        c = np.asarray(ring.coords, dtype=float) - 0.5
        verts.append(c)
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(c) - 2) + [Path.CLOSEPOLY])
    return Path(np.vstack(verts), codes)


def plot_mask_contours(
    mask,
    ax=None,
    connectivity: int = 8,
    min_area: float = 0.0,
    **plot_kwargs,
) -> Axes:
    """Vectorise *mask* and draw its outlines (no raster overlay).

    Equivalent to ``plot_polygons(mask_to_polygons(mask, ...), ...)`` and a
    drop-in for the prototype's ``pixsed.utils.plot_mask_contours``, except the
    half-pixel alignment is handled by the coordinate contract instead of a
    bare ``- 0.5`` at the plot call.
    """
    polys = mask_to_polygons(mask, connectivity=connectivity, min_area=min_area)
    return plot_polygons(polys, ax=ax, space="pixel", **plot_kwargs)


def show_mask(
    mask,
    data=None,
    figsize=(14, 7),
    overlay: bool = True,
    contour_color: str = "cyan",
    title=None,
    **kwargs,
):
    """One-call figure: the mask alone, and (optionally) overlaid on the image.

    Parameters
    ----------
    mask : array-like, 2-D
        The mask to display.
    data : array-like, 2-D, optional
        Background image.  When given, a two-panel figure is produced: mask on
        the left, overlay on the right.
    figsize : tuple
        Figure size.
    overlay : bool
        Draw the overlay panel when *data* is given.
    contour_color : str
        Colour of the vectorised outline in the overlay panel.
    title : str, optional
        Super-title for the figure.
    **kwargs
        Forwarded to :func:`plot_image` (stretch, percent, cmap, ...).

    Returns
    -------
    matplotlib.figure.Figure
    """
    mask = np.asarray(mask)
    if data is None:
        fig, ax = plt.subplots(figsize=(figsize[0] // 2, figsize[1]))
        plot_mask(mask, ax=ax)
        ax.set_title("mask")
        if title:
            fig.suptitle(title)
        return fig

    ncols = 1 if not overlay else 2
    fig, axs = plt.subplots(1, ncols, figsize=figsize, sharex=True, sharey=True)
    axs = np.atleast_1d(axs)

    plot_image(data, ax=axs[0], **kwargs)
    axs[0].set_title("data")

    if overlay:
        plot_image(data, ax=axs[1], **kwargs)
        plot_mask(mask, ax=axs[1])
        plot_mask_contours(mask, ax=axs[1], edgecolor=contour_color, lw=0.8)
        axs[1].set_title("data + mask")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
