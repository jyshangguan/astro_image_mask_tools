"""Executable checks for every number and call quoted in README.md.

The README used to carry ~40 specific quantities, three of which were wrong when
first written -- each a number carried over from a different shape or a
different ``join_style`` than the one being described.  The package is now two
functions and the README is short, so this file can stay exhaustive: every claim
below is one a reader could act on and be misled by if it drifted.

Run directly for a flat report::

    PYTHONPATH=src python3 tests/test_readme_claims.py
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import masktools as mt

from ._shapes import EXAMPLE_FITS, ROOT, load_example

README = (ROOT / "README.md").read_text()


# ---------------------------------------------------------------------------
# the shape of the public surface
# ---------------------------------------------------------------------------


def test_the_package_exports_exactly_two_functions():
    public = {n for n in dir(mt) if not n.startswith("_")}
    assert public == {"mask_poly_expand", "mask_manual", "api"}
    assert sorted(mt.__all__) == ["__version__", "mask_manual", "mask_poly_expand"]


def test_readme_never_names_an_internal_function():
    """The README is the whole manual, so internals must not leak into it."""
    internals = (
        "mask_to_polygons", "polygons_to_mask", "scale_mask", "grow_mask",
        "scale_polygons", "grow_polygons", "MaskDrawer", "draw_mask",
        "plot_image", "plot_mask", "plot_polygons", "make_norm", "show_mask",
        "mask_area", "corner_to_pixel", "pixel_to_corner", "space=",
        "connectivity=", "join_style=", "all_touched=", "min_percent=",
    )
    found = [name for name in internals if name in README]
    assert not found, f"internal API leaked into the README: {found}"


def test_version_is_stated_once():
    assert mt.__version__ == "0.1.0"


# ---------------------------------------------------------------------------
# "Expanding a mask"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factor", "pixels"), [(0.5, 25), (1.0, 100), (1.5, 225), (2.0, 400), (3.0, 900)]
)
def test_square_table_rows(factor, pixels):
    """The 10x10 square (100 pixels) and the pixel counts the table lists."""
    m = np.zeros((40, 40), bool)
    m[10:20, 10:20] = True
    assert m.sum() == 100
    assert mt.mask_poly_expand(m, factor).sum() == pixels


def test_factor_two_is_four_times_the_pixels():
    """'twice as wide -> four times the pixels', on the demo frame's mask."""
    image, mask, header = frame_values()
    assert mask.sum() == 582
    assert mt.mask_poly_expand(mask, 2.0).sum() == 4 * 582 == 2328


@pytest.mark.parametrize(("factor", "pixels"), [(0.5, 149), (1.5, 1303), (2.0, 2328)])
def test_demo_frame_counts(factor, pixels):
    """'149 px at 0.5, 1303 px at 1.5, 2328 px at 2.0'."""
    image, mask, header = frame_values()
    assert mt.mask_poly_expand(mask, factor).sum() == pixels


def test_factor_one_returns_the_mask_bit_for_bit():
    """'the same mask back, bit for bit'."""
    image, mask, header = frame_values()
    assert np.array_equal(mt.mask_poly_expand(mask, 1.0), mask)


def test_separate_regions_keep_their_separation():
    """'nothing slides toward the middle of the frame'."""
    m = np.zeros((90, 90), bool)
    m[10:16, 10:16] = True
    m[60:68, 20:26] = True
    out = mt.mask_poly_expand(m, 2.0)
    for label, sl in (("upper", np.s_[0:40, :]), ("lower", np.s_[40:, :])):
        before = np.array(np.nonzero(m[sl])).mean(axis=1)
        after = np.array(np.nonzero(out[sl])).mean(axis=1)
        assert np.allclose(before, after, atol=0.5), f"{label} region: {before} -> {after}"


def test_edge_clipping_warns_and_drops():
    """'You get a UserWarning and the overflowing part is dropped'."""
    m = np.zeros((20, 20), bool)
    m[15:20, 15:20] = True
    with pytest.warns(UserWarning, match="outside"):
        out = mt.mask_poly_expand(m, 4.0)
    assert out.sum() < 16 * m.sum()


def test_padding_the_frame_avoids_the_loss():
    """'numpy.pad the frame first if you need that room'."""
    m = np.zeros((20, 20), bool)
    m[15:20, 15:20] = True                     # touches the upper/right edge
    with pytest.warns(UserWarning):
        clipped = mt.mask_poly_expand(m, 4.0)
    padded = np.pad(m, 20)
    big = mt.mask_poly_expand(padded, 4.0)     # room to spare: no warning
    assert big.sum() == 16 * m.sum() > clipped.sum()
    assert np.array_equal(big[20:-20, 20:-20], clipped)


def test_shrink_floor_leaves_one_pixel():
    """'factor=0.01 on a 3x3 square still leaves 1 pixel'."""
    m = np.zeros((40, 40), bool)
    m[18:21, 18:21] = True
    assert mt.mask_poly_expand(m, 0.01).sum() == 1


def test_output_is_a_plain_boolean_array():
    """'a 2-D boolean array where True means "this pixel is masked"'."""
    image, mask, header = frame_values()
    out = mt.mask_poly_expand(mask, 1.7)
    assert out.dtype == bool and out.shape == mask.shape


# ---------------------------------------------------------------------------
# "How to choose the factor" -- the flux-coverage sentence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("factor", "percent"), [(1.0, 0.0), (2.0, 96.0), (3.0, 100.0)])
def test_starlight_coverage(factor, percent):
    """'0% of the star's light at factor=1, 96% at 2.0, and all of it at 3.0'.

    Recomputed here with the same window and threshold the demo notebook's
    section 3 uses, so the README and the notebook cannot drift apart.
    """
    image, mask, header = frame_values()
    sky = np.median(image)
    sigma = 1.4826 * np.median(np.abs(image - sky))
    rows, cols = np.nonzero(mask)                      # the notebook's ZOOM
    window = np.zeros_like(mask)
    window[rows.min() - 25 : rows.max() + 25, cols.min() - 25 : cols.max() + 25] = True
    bright = ((image - sky) > 3 * sigma) & window
    excess = np.clip(image - sky, 0, None)
    covered = excess[bright & mt.mask_poly_expand(mask, factor)].sum() / excess[bright].sum()
    assert 100 * covered == pytest.approx(percent, abs=0.05)


# ---------------------------------------------------------------------------
# "Drawing a mask by hand"
# ---------------------------------------------------------------------------


def test_mask_manual_signature_matches_the_readme_snippets():
    """mask_manual(image), mask_manual(image, mask=...), ..., cmap=..."""
    import inspect

    sig = inspect.signature(mt.mask_manual)
    assert list(sig.parameters)[:2] == ["image", "mask"]
    assert all(sig.parameters[n].default is None for n in ("image", "mask"))
    assert "kwargs" in str(sig)
    for name in ("shape", "figsize", "block"):
        assert name in sig.parameters, name


@pytest.mark.parametrize("key", ["n", "d", "q", "h"])
def test_every_key_in_the_readme_table_is_actually_bound(key, capsys):
    """The gesture table is documentation the user acts on; drive it."""
    from masktools._draw import MaskDrawer

    from .test_draw import Mouse

    image, mask, header = frame_values()
    drawer = MaskDrawer(data=image, mask=mask, verbose=False)
    mouse = Mouse(drawer)
    try:
        n_before = len(drawer.vertices)
        mouse.key(key)
        printed = capsys.readouterr().out
        if key == "h":
            assert "left click" in printed and "finish and close" in printed
        elif key == "q":
            assert drawer._closed is True
        elif key == "n":
            assert n_before == len(drawer.vertices)  # nothing live to commit
        else:  # 'd' deletes the last finished region
            assert len(drawer.vertices) == n_before - 1
    finally:
        drawer.teardown()


def test_reserved_keys_stay_with_the_selector():
    """The table gives shift/ctrl/escape to the widget, not to the drawer."""
    from masktools._draw import RESERVED_KEYS

    assert RESERVED_KEYS == ("escape", "shift", "control")


def test_abort_loses_nothing(monkeypatch):
    """'Closing the window without drawing loses nothing'."""
    import masktools.api as api

    def close_at_once(factory, figsize=(8, 8), title=""):
        import matplotlib

        drawer = factory(matplotlib.figure.Figure().add_subplot(111))
        drawer.close()
        return drawer

    monkeypatch.setattr(api, "_run_in_popup", close_at_once)
    image, mask, header = frame_values()
    assert np.array_equal(mt.mask_manual(image, mask=mask, verbose=False), mask)
    empty = mt.mask_manual(shape=(16, 24), verbose=False)
    assert empty.sum() == 0 and empty.shape == (16, 24) and empty.dtype == bool


def test_existing_regions_come_in_as_editable_outlines():
    """'its regions arrive as editable outlines, so you are refining'."""
    image, mask, header = frame_values()
    drawer = mt.mask_manual(image, mask=mask, block=False, verbose=False)
    try:
        assert len(drawer.vertices) == 1, "the island is one region"
        assert len(drawer.vertices[0]) > 20, "a real outline, not a bounding box"
        drawer.add_polygon([(5, 5), (20, 5), (20, 20)])
        out = drawer.mask()
        assert np.all(out[mask]), "the imported region survives"
        assert out.sum() > mask.sum(), "and the new one adds to it"
    finally:
        drawer.teardown()


def test_cmap_reaches_the_background():
    """'mask_manual(image, mask=..., cmap="inferno")  # tune the background'."""
    image, mask, header = frame_values()
    drawer = mt.mask_manual(
        image, mask=mask, block=False, cmap="inferno", verbose=False
    )
    try:
        assert drawer.ax.images[0].cmap.name == "inferno"
    finally:
        drawer.teardown()


def test_the_window_is_tk_and_the_library_never_switches_backends():
    """'a plain Tk window ... never changes your matplotlib backend'."""
    source = (ROOT / "src" / "masktools" / "_draw.py").read_text()
    assert "backend_tkagg" in source, "the window must really be Tk"

    # A string search would hit the docstrings that *advise* the user to pick a
    # backend, so walk the syntax tree for actual calls instead.
    calls = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                calls.add(func.attr)
            elif isinstance(func, ast.Name):
                calls.add(func.id)
    forbidden = {"use", "ion", "switch_backend", "get_ipython",
                 "run_line_magic", "InteractiveShell"}
    assert not calls & forbidden, f"backend-mutating call in _draw.py: {calls & forbidden}"


# ---------------------------------------------------------------------------
# install, data, notebook, tests
# ---------------------------------------------------------------------------


def test_demo_frame_exists_at_the_quoted_size():
    """'data/example_cutout.fits.gz (820 KB)' and '512 x 512'."""
    kb = EXAMPLE_FITS.stat().st_size / 1024
    assert EXAMPLE_FITS.exists()
    assert 810 < kb < 830, f"{kb:.0f} KB"
    image, mask, header = frame_values()
    assert image.shape == (512, 512)


def test_readme_extras_all_exist_in_pyproject():
    """The install table promises fits / notebook / test / all."""
    text = (ROOT / "pyproject.toml").read_text()
    block = text.split("[project.optional-dependencies]")[1]
    block = re.split(r"^\[", block, maxsplit=1, flags=re.M)[0]
    names = set(re.findall(r"^(\w+) =", block, re.M))
    assert names == {"fits", "notebook", "test", "all"}


def test_python_floor_is_as_documented():
    assert 'requires-python = ">=3.10"' in (ROOT / "pyproject.toml").read_text()


def test_notebook_exists_with_the_quoted_cell_count():
    """'the whole package in 20 cells'."""
    import nbformat

    nb = ROOT / "examples" / "masktools_demo.ipynb"
    doc = nbformat.read(nb, as_version=4)
    assert len(doc.cells) == 20
    assert sum(len(c.get("outputs", [])) for c in doc.cells) == 0, "commit it bare"
    assert "DRAW_BY_HAND = True" in "".join(c.source for c in doc.cells)


def test_the_notebook_only_uses_the_two_functions():
    import nbformat

    nb = ROOT / "examples" / "masktools_demo.ipynb"
    source = "".join(c.source for c in nbformat.read(nb, as_version=4).cells)
    assert "from masktools import mask_manual, mask_poly_expand" in source
    for name in ("mask_to_polygons", "polygons_to_mask", "MaskDrawer", "grow_mask",
                 "scale_mask", "draw_mask", "make_norm", "plot_image", "space="):
        assert name not in source, f"{name} leaked into the notebook"


def test_every_notebook_heading_the_readme_points_at_exists():
    """Cross-references rot silently; numbers at least fail loudly.

    The README first sent the reader to "section 3" for the flux measurement,
    which actually lives in section 1.  Nothing in the suite would have noticed.
    """
    import nbformat

    nb = ROOT / "examples" / "masktools_demo.ipynb"
    heads = []
    for c in nbformat.read(nb, as_version=4).cells:
        if c.cell_type == "markdown":
            heads += re.findall(r"^#{1,4}\s+(.+)$", c.source, re.M)
    # Whitespace-normalised on both sides: prose wraps across lines.
    squash = lambda t: re.sub(r"\s+", " ", t).strip().lower()
    joined = squash(" ".join(heads))
    # Every italic/bold phrase the README attributes to the notebook.
    for phrase in re.findall(r"The demo notebook's \*\"([^\"]+)\"\*", README):
        assert squash(phrase) in joined, (
            f"README points at {phrase!r}; notebook headings are: {heads}"
        )
    # And the headings it describes in prose.
    for phrase in ("Expanding a mask", "Drawing a mask by hand"):
        assert squash(phrase) in joined, f"missing notebook heading: {phrase}"


def test_the_embedded_figure_exists():
    for rel in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", README):
        assert (ROOT / rel).exists(), f"missing figure {rel}"


def test_readme_links_all_resolve():
    """Markdown links, minus the anchors GitHub makes up itself."""
    bad = []
    for text, rel in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", README):
        if rel.startswith(("http", "#")):
            continue
        path = ROOT / rel.split("#")[0]
        if not path.exists():
            bad.append(f"{text} -> {rel}")
    assert not bad, f"broken README links: {bad}"


def test_readme_test_count_matches_the_suite():
    """The README's "N passed" line, which a reader uses to spot a broken
    install.  The number is deliberately not written here: it is read out of the
    README and compared against a live collection, so hardcoding it in this
    docstring would only rot."""
    stated = int(re.search(r"(\d+) passed", README).group(1))
    env = {**os.environ, "PYTHONPATH": "src:.testdeps", "MPLBACKEND": "Agg"}
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--collect-only"],
        capture_output=True, text=True, cwd=ROOT, env=env,
    )
    collected = sum(int(n) for n in re.findall(r"^tests/\S+: (\d+)$", out.stdout, re.M))
    assert collected > 100, f"cannot parse the collection report:\n{out.stdout[-400:]}"
    assert stated == collected, f"README says {stated}, suite collects {collected}"


def test_licence_file_is_mit():
    assert "MIT" in (ROOT / "LICENSE").read_text()[:200]


# --------------------------------------------------------------------------
def frame_values():
    """Cached load of the demo frame (the tests are read-only on it)."""
    global _FRAME
    if _FRAME is None:
        _FRAME = load_example()
    return _FRAME


_FRAME = None


if __name__ == "__main__":  # pragma: no cover - manual report
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))
