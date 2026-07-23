"""Discover, load, and filter per-night WFS CSVs; load the MiniCyclop log."""
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .periods import Period

_DIR_RE = re.compile(r"^\d{8}$")


class NoDataError(Exception):
    """No usable WFS data for the requested period."""


class CyclopUnavailable(Exception):
    """The MiniCyclop seeing log could not be loaded."""


def discover_csvs(data_dir: Path, period: Period) -> list[Path]:
    data_dir = Path(data_dir)
    out = []
    for csv in sorted(data_dir.glob("*/reanalyze_results.csv")):
        name = csv.parent.name
        if not _DIR_RE.match(name):
            continue
        day = pd.Timestamp(datetime.strptime(name, "%Y%m%d"))
        if period.start <= day < period.end:
            out.append(csv)
    return out


def load_wfs(data_dir: Path, period: Period) -> pd.DataFrame:
    csvs = discover_csvs(data_dir, period)
    if not csvs:
        raise NoDataError(
            f"no reanalyze_results.csv found under {Path(data_dir)} for "
            f"{period.spec} ({period.date_range_str})"
        )
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df = df[np.isfinite(df["seeing"])]
    df = df[df["fwhm"] > 0.0]
    df = df[(df["seeing"] > 0.0) & (df["seeing"] < 4.0)]
    if df.empty:
        raise NoDataError(f"all rows filtered out for {period.spec}; no usable data")
    df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["time"]), name="ut"))
    return df.sort_index()


def warn_coverage(df: pd.DataFrame, period: Period) -> None:
    last_day = (period.end - pd.Timedelta(days=1)).normalize()
    max_night = df.index.max().normalize()
    if max_night < last_day:
        short = (last_day - max_night).days
        warnings.warn(
            f"data for {period.spec} ends {max_night.date()}, {short} days short "
            f"of the period end {last_day.date()}; figures cover a partial period.",
            UserWarning,
            stacklevel=2,
        )
    first_day = period.start.normalize()
    min_night = df.index.min().normalize()
    if min_night > first_day:
        print(f"note: data for {period.spec} starts {min_night.date()}, "
              f"after the period start {first_day.date()}.")


def load_cyclop(period: Period, path: Path | None = None) -> pd.DataFrame:
    if path is None:
        path = Path.home() / "MMT/minicyclop/data/MiniCyclop/Data/Seeing_Data.txt"
    path = Path(path)
    if not path.exists():
        raise CyclopUnavailable(f"cyclop log not found: {path}")
    try:
        from minicyclop.io import read_seeing_data
    except ImportError as exc:
        raise CyclopUnavailable(f"minicyclop not importable: {exc}") from exc
    cyc = read_seeing_data(path)
    cyc = cyc[(cyc.index >= period.start) & (cyc.index < period.end)]
    if cyc.empty:
        raise CyclopUnavailable(f"no cyclop data in {period.date_range_str}")
    return cyc
