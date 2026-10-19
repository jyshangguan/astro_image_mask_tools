"""Run the drawing window in a process of its own.

``mask_manual`` opens the window here rather than in the caller's process, for
one concrete reason: Tk in a Jupyter kernel shares the main thread with
ipykernel's asyncio loop, and on macOS its run loop is then not pumped at all.
The window shows up but never draws or answers the mouse, and macOS reports
``error messaging the mach port for IMKCFRunLoopWakeUpReliable``.  A child
process gets an ordinary main thread, so the window behaves exactly as it does
in a script; the parent only waits for it, and stopping the cell kills it.

Everything crosses the process boundary through a temporary directory: the
arrays and settings are pickled into a spec, and the finished mask is written
back as ``.npy``.
"""

import os
import pickle
import sys

import numpy as np


def main(argv=None) -> None:
    argv = sys.argv if argv is None else argv
    with open(argv[1], "rb") as fh:
        spec = pickle.load(fh)

    data = np.load(spec["data"]) if spec.get("data") else None
    mask = np.load(spec["mask"]) if spec.get("mask") else None
    shape = spec.get("shape")

    from ._draw import _build_drawer, _run_in_popup

    def factory(ax):
        return _build_drawer(data, mask, shape, ax, dict(spec.get("kwargs") or {}))

    drawer = _run_in_popup(
        factory,
        figsize=tuple(spec.get("figsize", (8, 8))),
        parent_pid=os.getppid(),
    )
    np.save(spec["out"], drawer.mask(shape))


if __name__ == "__main__":
    main()
