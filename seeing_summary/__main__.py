"""CLI: python -m seeing_summary PERIOD [options]."""
import argparse
import sys
import warnings
from pathlib import Path

from .periods import parse_period
from . import data, plots


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="seeing_summary")
    parser.add_argument("period", help="period spec, e.g. 2026q2 or 2025")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("images"))
    parser.add_argument("--no-cyclop", action="store_true",
                        help="skip the seeing-monitor figures")
    args = parser.parse_args(argv)

    try:
        period = parse_period(args.period)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        df = data.load_wfs(args.data_dir, period)
    except data.NoDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    data.warn_coverage(df, period)

    out_dir = args.out_dir / str(period.year)
    written = plots.render_wfs_figures(df, period, out_dir)

    if not args.no_cyclop:
        try:
            cyclop = data.load_cyclop(period)
        except data.CyclopUnavailable as exc:
            warnings.warn(f"skipping cyclop figures: {exc}")
        else:
            written += plots.render_cyclop_figures(df, cyclop, period, out_dir)

    for path in written:
        print(f"wrote {path}")
    print(f"{len(written)} figures written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
