import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seeing_summary.periods import parse_period
from seeing_summary import data


def _write_night(root: Path, day: str, n: int = 5, time_format: str = "%Y-%m-%dT%H:%M:%S.%f"):
    d = root / day
    d.mkdir(parents=True)
    ts = pd.date_range(f"{day[:4]}-{day[4:6]}-{day[6:]}T02:00:00", periods=n, freq="min")
    df = pd.DataFrame({
        "time": ts.strftime(time_format),
        "wfs": ["f5"] * n,
        "el": np.linspace(60, 70, n),
        "seeing": np.linspace(0.5, 1.5, n),
        "vlt_seeing": np.linspace(0.5, 1.4, n),
        "raw_seeing": np.linspace(0.6, 1.6, n),
        "ellipticity": np.linspace(0.1, 0.2, n),
        "fwhm": np.linspace(3.0, 5.0, n),
    })
    df.to_csv(d / "reanalyze_results.csv", index=False)


def test_discover_filters_by_date(tmp_path):
    _write_night(tmp_path, "20251015")   # in 2025q4
    _write_night(tmp_path, "20260105")   # out (2026q1)
    period = parse_period("2025q4")
    found = data.discover_csvs(tmp_path, period)
    assert [p.parent.name for p in found] == ["20251015"]


def test_load_wfs_ok(tmp_path):
    _write_night(tmp_path, "20251015")
    df = data.load_wfs(tmp_path, parse_period("2025q4"))
    assert df.index.name == "ut"
    assert len(df) == 5
    assert df["vlt_seeing"].notna().all()


def test_load_wfs_mixed_time_formats(tmp_path):
    # Real per-night CSVs have mixed `time` formats within a single quarter:
    # some rows have microseconds, some don't. pd.to_datetime must not choke
    # on the mix (regression for the ValueError seen against real data).
    _write_night(tmp_path, "20251001", time_format="%Y-%m-%dT%H:%M:%S.%f")
    _write_night(tmp_path, "20251020", time_format="%Y-%m-%dT%H:%M:%S")
    df = data.load_wfs(tmp_path, parse_period("2025q4"))
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "ut"
    assert len(df) == 10


def test_load_wfs_no_data_raises(tmp_path):
    with pytest.raises(data.NoDataError):
        data.load_wfs(tmp_path, parse_period("2025q4"))


def test_warn_coverage_short(tmp_path):
    _write_night(tmp_path, "20251005")  # only early-October data for a full quarter
    df = data.load_wfs(tmp_path, parse_period("2025q4"))
    with pytest.warns(UserWarning, match="short of the period end"):
        data.warn_coverage(df, parse_period("2025q4"))


def test_warn_coverage_full_no_warning(tmp_path):
    _write_night(tmp_path, "20251231")
    df = data.load_wfs(tmp_path, parse_period("2025q4"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data.warn_coverage(df, parse_period("2025q4"))  # must not warn


def test_load_cyclop_missing_log_raises(tmp_path):
    with pytest.raises(data.CyclopUnavailable):
        data.load_cyclop(parse_period("2025q4"), path=tmp_path / "nope.txt")
