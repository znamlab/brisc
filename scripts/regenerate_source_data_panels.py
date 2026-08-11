#!/usr/bin/env python3
"""
Redraw one figure per Source Data panel, reading only the exported Excel workbooks.

This is a round-trip check on the Source Data deposit: if a panel cannot be drawn
from its worksheet alone, the worksheet is missing something. PNGs are written to
``<source-data-dir>/panels/<Figure_N>/<Sheet_Name>.png``.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from brisc.manuscript_analysis.utils import get_output_folder  # noqa: E402
from brisc.source_data import panels  # noqa: E402

DATA_ROOT = Path("/Volumes/BlackPasspo/brisc/brisc")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        default=None,
        help="Directory holding the Source_Data_*.xlsx files. Defaults to "
        "<figure output folder>/Source_Data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the panels. Defaults to <source-data-dir>/panels.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Resolution of the PNGs.")
    parser.add_argument(
        "--pattern",
        default="Source_Data_*.xlsx",
        help="Glob used to find the workbooks.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    source_data_dir = args.source_data_dir
    if source_data_dir is None:
        source_data_dir = get_output_folder(DATA_ROOT) / "Source_Data"
    if not source_data_dir.is_dir():
        print(f"[Panels] Source Data directory not found: {source_data_dir}")
        return 1

    output_dir = args.output_dir or source_data_dir / "panels"
    print(f"Reading Source Data from {source_data_dir}")
    print(f"Writing panels to {output_dir}")

    results = panels.plot_all_workbooks(
        source_data_dir, output_dir=output_dir, dpi=args.dpi, pattern=args.pattern
    )
    n_panels = sum(len(v) for v in results.values())
    print(f"\n[Panels] Wrote {n_panels} panels from {len(results)} workbooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
