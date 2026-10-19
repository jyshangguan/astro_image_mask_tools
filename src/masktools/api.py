"""The two public functions of :mod:`masktools`.

Everything the package needs underneath -- the polygon conversion, the
rasteriser, the drawing widget -- lives in private modules and is reached only
through the two functions defined here.
"""

from __future__ import annotations

import numpy as np

from ._draw import _HELP, MaskDrawer, _run_in_popup
from ._transform import scale_mask

__all__ = ["mask_poly_expand", "mask_manual"]


def mask_poly_expand(mask, factor: float) -> np.ndarray:
    """Grow or shrink every region of a pixel mask by a linear *factor*.

    Each masked region is turned into a polygon, scaled about its own centre,
    and painted back onto the same pixel grid.  Because the scaling is linear,
    the number of pixels goes as ``factor ** 2``: ``factor=2`` makes a region
    twice as wide *and* twice as tall, i.e. four times as many pixels.
    ``factor=1`` returns the mask unchanged.

    Parameters
    ----------
    mask : array-like, 2-D
        The mask.  Nonzero means masked, so boolean, ``uint8`` and labelled
        integer arrays all work.
    factor : float
        Positive scaling factor.  ``> 1`` grows, ``< 1`` shrinks.

    Returns
    -------
    numpy.ndarray
        Boolean mask with the same shape as *mask*.

    Warns
    -----
    UserWarning
        When a grown region no longer fits inside the array: the part outside
        is dropped.  Pad the input with :func:`numpy.pad` first if you need
        that room.

    See Also
    --------
    mask_manual : draw or add regions by hand.

    Examples
    --------
    >>> import numpy as np
    >>> from masktools import mask_poly_expand
    >>> m = np.zeros((40, 40), bool)
    >>> m[10:20, 10:20] = True              # a 10 x 10 square, 100 pixels
    >>> mask_poly_expand(m, 2.0).sum()      # twice as wide -> 4x the area
    400
    >>> mask_poly_expand(m, 0.5).sum()      # half as wide -> a quarter
    25
    """
    return scale_mask(mask, factor)


def mask_manual(image=None, mask=None, *, shape=None, figsize=(8, 8),
                block: bool = True, **kwargs):
    """Draw mask regions with the mouse and return the new mask.

    Opens a window showing *image*, with *mask* (if given) tinted red, and
    waits for you to trace polygons.  Press ``q`` -- or close the window --
    and the call returns the union of everything in there as a boolean array.

    * From scratch: ``m = mask_manual(image)``.
    * Added to an existing mask: ``m = mask_manual(image, mask=old)``.
    * Refining an auto-generated mask: the same call -- the old regions come
      in as polygons you can drag, delete, and add to.

    Aborting is safe: closing the window without drawing returns *mask*
    unchanged, or an all-``False`` mask when none was given.

    Parameters
    ----------
    image : array-like, 2-D, optional
        Background to draw against.  Without it the window is a blank canvas,
        which is only useful together with *shape*.
    mask : array-like, 2-D, optional
        Existing mask to add regions to.
    shape : (int, int), optional
        ``(nrow, ncol)`` of the result, when there is no *image* or *mask* to
        take it from.
    figsize : (float, float)
        Size of the drawing window in inches.
    block : bool
        `True` (default) waits for the mouse and returns a mask.  `False`
        returns the drawer straight away, for scripts and tests that add
        polygons with ``add_polygon()`` instead of drawing them.
    **kwargs
        Passed to the drawer: ``verbose=False`` silences its progress lines;
        ``cmap``, ``stretch``, ``vmin``/``vmax`` tune the background;
        ``mask_color``/``poly_color`` the overlays.

    Returns
    -------
    numpy.ndarray or MaskDrawer
        The boolean mask, same shape as *image*; or, with ``block=False``,
        the drawer to work with programmatically.

    Notes
    -----
    In the window:

    ============================== ==========================================
    gesture                        action
    ============================== ==========================================
    left click                     place a vertex
    click the first vertex again   close the polygon (needs >= 3 vertices)
    drag a placed vertex           move it
    ``shift`` + drag               move the whole polygon
    right click a vertex           delete it
    ``n``                          finish the current polygon, start the next
    ``d``                          delete the last finished polygon
    ``q``                          done -- return the mask
    ``h``                          print this list
    ============================== ==========================================

    The window is a plain Tk window, so the call behaves the same from a
    Jupyter notebook, a Qt console, or a script: it blocks until you are
    done, and it never changes your matplotlib backend.
    """
    if image is None and mask is None and shape is None:
        raise ValueError(
            "nothing to draw on: pass an image, an existing mask, or a shape "
            "like shape=(512, 512)"
        )
    if not block:
        return MaskDrawer(data=image, mask=mask, shape=shape, **kwargs)
    if kwargs.get("verbose", True):
        print(_HELP)

    def _build(ax):
        drawer = MaskDrawer(data=image, mask=mask, shape=shape, ax=ax, **kwargs)
        if image is None and mask is None:
            # A blank canvas: give the axes the pixel extent the mask will be
            # rasterised onto, or every click lands inside a default 0..1 box.
            nrow, ncol = shape
            ax.set_xlim(-0.5, ncol - 0.5)
            ax.set_ylim(-0.5, nrow - 0.5)
        return drawer

    drawer = _run_in_popup(_build, figsize=figsize)
    return drawer.mask(shape)
