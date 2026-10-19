"""Pytest fixtures for the masktools suite."""

from __future__ import annotations

import pytest

from ._shapes import EXAMPLE_FITS, block, load_example, ring, staircase


@pytest.fixture(scope="session")
def example_image():
    """The shipped 512x512 NTT/EFOSC cutout: ``(data, mask, header)``."""
    return load_example()
