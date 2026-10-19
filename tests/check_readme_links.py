"""Check that every path and image referenced by README.md actually exists.

Run:  python tests/check_readme_links.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    text = (ROOT / "README.md").read_text()
    refs = set(re.findall(r"!?\[[^\]]*\]\(([^)#][^)]*)\)", text))
    refs |= set(re.findall(r"`(examples/[\w./-]+|data/[\w./-]+|src/[\w./-]+)`", text))

    missing, checked = [], 0
    for ref in sorted(refs):
        if ref.startswith(("http://", "https://", "mailto:")):
            continue
        target = (ROOT / ref).resolve()
        checked += 1
        if not target.exists():
            missing.append(ref)
        else:
            size = target.stat().st_size
            print(f"  ok   {ref:52s} {size / 1024:7.0f} KiB")

    # code fences should at least be balanced
    fences = text.count("```")
    if fences % 2:
        missing.append(f"unbalanced code fences ({fences})")

    # tables: a blank *leading* cell is the conventional row-label column, but
    # any other empty cell means a mistyped row.
    for i, line in enumerate(text.splitlines(), 1):
        if not (line.startswith("|") and line.rstrip().endswith("|")):
            continue
        row = [c.strip() for c in line.strip("|").split("|")]
        if all(c == "" for c in row):
            missing.append(f"entirely empty table row on line {i}: {line[:60]}")
        elif any(c == "" for c in row[1:]):
            missing.append(f"empty table cell on line {i}: {line[:60]}")

    print(f"\n{checked} references checked, {len(missing)} problems")
    for m in missing:
        print("  MISSING", m)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
