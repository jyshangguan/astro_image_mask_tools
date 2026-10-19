"""masktools -- pixel masks: expand them by a factor, draw them by hand.

Two functions make the whole package:

- :func:`mask_poly_expand` -- grow or shrink every region of a mask by a
  linear factor (a factor of 2 quadruples a region's pixels).
- :func:`mask_manual` -- trace regions on the image with the mouse and get a
  boolean mask back, from scratch or added to an existing one.

Everything they need underneath lives in private modules.
"""

from .api import mask_manual, mask_poly_expand

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "mask_poly_expand",
    "mask_manual",
]
