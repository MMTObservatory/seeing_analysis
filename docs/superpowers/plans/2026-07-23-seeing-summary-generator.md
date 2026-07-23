# Seeing-Summary Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the copy-a-notebook-per-quarter workflow with a `seeing_summary` package (run via `make 2026q2`) that generates the full seeing figure set for any quarter or year, and reorganize the repo's images and notebooks.

**Architecture:** A small importable package: `periods` (parse a spec into a date range + labels), `data` (auto-discover and load per-night CSVs by date; load the MiniCyclop log), `plots` (one function per figure, driven off a DatetimeIndex so quarter vs year is automatic), and `__main__` (argparse CLI). A `Makefile` wraps the CLI so a period is generated with just its spec. A one-time reorg moves images into `images/<year>/` and notebooks into `notebooks/`.

**Tech Stack:** Python 3 (numpy, pandas, scipy, matplotlib, astropy), `minicyclop` (optional, for the seeing-monitor comparison), GNU make, pytest.

## Global Constraints

- Run all Python in the `mmtwfs` conda env: `/Users/tim/conda/envs/mmtwfs/bin/python`. All `pytest`/`python` commands below use it.
- The generator reads only `data/YYYYMMDD/reanalyze_results.csv`; it needs **no** `mmtwfs` import. The only optional dependency is `minicyclop.io.read_seeing_data`.
- Seeing plotted is the `vlt_seeing` column (zenith-corrected). Standard filter: `seeing` finite, `fwhm > 0`, `0 < seeing < 4`.
- Night split (UT): first half `00:00`–`07:00`, second half `07:00`–`14:00`.
- Period specs: `YYYY` or `YYYYqN` only. Date ranges are half-open `[start, end)`, tz-naive.
- Output figures: `images/<year>/{tag}_{figure}.png`, where `tag` is `2025_q4` for a quarter and `2025` for a year. matplotlib uses the `Agg` backend and `ggplot` style.
- Commit after each task. We are on branch `seeing-summary-generator`.

---

### Task 1: `periods.py` — period parsing

**Files:**
- Create: `seeing_summary/__init__.py` (empty)
- Create: `seeing_summary/periods.py`
- Test: `tests/test_periods.py`

**Interfaces:**
- Produces: `parse_period(spec: str) -> Period`. `Period` is a frozen dataclass with fields `spec: str`, `start: pd.Timestamp`, `end: pd.Timestamp`, `tag: str`, `title: str`, `year: int`, and properties `date_range_str -> str` and `month_keys -> list[str]`. `parse_period` raises `ValueError` on an unrecognized spec.

- [ ] **Step 1: Write the failing test**

Create `tests/test_periods.py`:

```python
import pandas as pd
import pytest

from seeing_summary.periods import parse_period


def test_quarter():
    p = parse_period("2025q4")
    assert p.start == pd.Timestamp(2025, 10, 1)
    assert p.end == pd.Timestamp(2026, 1, 1)
    assert p.tag == "2025_q4"
    assert p.title == "2025 Q4"
    assert p.year == 2025
    assert p.date_range_str == "2025-10-01 through 2025-12-31"
    assert p.month_keys == ["2025-10", "2025-11", "2025-12"]


def test_year():
    p = parse_period("2025")
    assert p.start == pd.Timestamp(2025, 1, 1)
    assert p.end == pd.Timestamp(2026, 1, 1)
    assert p.tag == "2025"
    assert p.title == "2025"
    assert p.year == 2025
    assert len(p.month_keys) == 12
    assert p.month_keys[0] == "2025-01"
    assert p.month_keys[-1] == "2025-12"


def test_q1_bounds():
    p = parse_period("2026q1")
    assert p.start == pd.Timestamp(2026, 1, 1)
    assert p.end == pd.Timestamp(2026, 4, 1)


def test_bad_spec_raises():
    with pytest.raises(ValueError):
        parse_period("2022_2ndhalf")
    with pytest.raises(ValueError):
        parse_period("2025q5")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest tests/test_periods.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'seeing_summary'`.

- [ ] **Step 3: Write minimal implementation**

Create `seeing_summary/__init__.py` as an empty file.

Create `seeing_summary/periods.py`:

```python
"""Parse period specs (YYYY, YYYYqN) into date ranges and labels."""
import re
from dataclasses import dataclass

import pandas as pd

_QUARTER_RE = re.compile(r"^(\d{4})q([1-4])$")
_YEAR_RE = re.compile(r"^(\d{4})$")


@dataclass(frozen=True)
class Period:
    spec: str
    start: pd.Timestamp
    end: pd.Timestamp  # exclusive
    tag: str
    title: str
    year: int

    @property
    def date_range_str(self) -> str:
        last = (self.end - pd.Timedelta(days=1)).date()
        return f"{self.start.date()} through {last}"

    @property
    def month_keys(self) -> list[str]:
        last = self.end - pd.Timedelta(days=1)
        return [p.strftime("%Y-%m") for p in pd.period_range(self.start, last, freq="M")]


def parse_period(spec: str) -> Period:
    key = spec.strip().lower()
    m = _QUARTER_RE.match(key)
    if m:
        year, q = int(m.group(1)), int(m.group(2))
        start = pd.Timestamp(year, (q - 1) * 3 + 1, 1)
        end = start + pd.DateOffset(months=3)
        return Period(key, start, end, f"{year}_q{q}", f"{year} Q{q}", year)
    m = _YEAR_RE.match(key)
    if m:
        year = int(m.group(1))
        return Period(key, pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1),
                      f"{year}", f"{year}", year)
    raise ValueError(f"unrecognized period spec {spec!r}; expected YYYY or YYYYqN")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest tests/test_periods.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add seeing_summary/__init__.py seeing_summary/periods.py tests/test_periods.py
git commit -m "feat: add period parsing for seeing_summary"
```

---

### Task 2: `data.py` — CSV discovery, loading, coverage, cyclop

**Files:**
- Create: `seeing_summary/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `Period` from `seeing_summary.periods`.
- Produces:
  - `discover_csvs(data_dir: Path, period: Period) -> list[Path]`
  - `load_wfs(data_dir: Path, period: Period) -> pd.DataFrame` (filtered, `ut` DatetimeIndex, sorted); raises `NoDataError` if nothing usable.
  - `warn_coverage(df: pd.DataFrame, period: Period) -> None` (emits a `warnings.warn` if data ends before the period end).
  - `load_cyclop(period: Period, path: Path | None = None) -> pd.DataFrame`; raises `CyclopUnavailable`.
  - Exceptions `NoDataError`, `CyclopUnavailable`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_data.py`:

```python
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seeing_summary.periods import parse_period
from seeing_summary import data


def _write_night(root: Path, day: str, n: int = 5):
    d = root / day
    d.mkdir(parents=True)
    ts = pd.date_range(f"{day[:4]}-{day[4:6]}-{day[6:]}T02:00:00", periods=n, freq="min")
    df = pd.DataFrame({
        "time": ts.strftime("%Y-%m-%dT%H:%M:%S.%f"),
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest tests/test_data.py -q`
Expected: FAIL — `AttributeError`/`ImportError` (no `seeing_summary.data`).

- [ ] **Step 3: Write minimal implementation**

Create `seeing_summary/data.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest tests/test_data.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add seeing_summary/data.py tests/test_data.py
git commit -m "feat: add CSV discovery, loading, coverage checks, cyclop loader"
```

---

### Task 3: `plots.py` — WFS figures

**Files:**
- Create: `seeing_summary/plots.py`
- Test: `tests/test_plots.py`

**Interfaces:**
- Consumes: `Period`; a WFS DataFrame with `vlt_seeing`, `seeing`, `ellipticity`, `el`, `wfs` columns and a `ut` DatetimeIndex.
- Produces: `render_wfs_figures(df: pd.DataFrame, period: Period, out_dir: Path) -> list[Path]`, plus private helpers `_month_groups`, `_daily_groups`, `_violin`. Writes 11 PNGs named `{period.tag}_{figure}.png` into `out_dir`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_plots.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seeing_summary.periods import parse_period
from seeing_summary import plots


@pytest.fixture
def wfs_df():
    idx = pd.date_range("2025-10-02T02:00:00", "2025-12-20T09:00:00", periods=600)
    rng = np.random.default_rng(0)
    n = len(idx)
    wfs = rng.choice(["binospec", "mmirs", "f5", "newf9"], size=n)
    df = pd.DataFrame({
        "vlt_seeing": np.abs(rng.normal(0.9, 0.3, n)) + 0.2,
        "seeing": np.abs(rng.normal(1.0, 0.3, n)) + 0.2,
        "ellipticity": np.abs(rng.normal(0.15, 0.05, n)),
        "el": rng.uniform(30, 85, n),
        "wfs": wfs,
    }, index=pd.DatetimeIndex(idx, name="ut"))
    return df


def test_render_wfs_figures(wfs_df, tmp_path):
    period = parse_period("2025q4")
    written = plots.render_wfs_figures(wfs_df, period, tmp_path)
    names = {p.name for p in written}
    assert names == {
        "2025_q4_hist.png", "2025_q4_monthly.png", "2025_q4_1st2nd.png",
        "2025_q4_nightly.png", "2025_q4_violin.png", "2025_q4_violin_monthly.png",
        "2025_q4_ellip_violin.png", "2025_q4_per_instrument.png",
        "2025_q4_ellipticity.png", "2025_q4_ellip_vs_inst.png",
        "2025_q4_bino_ellip_vs_el.png",
    }
    for p in written:
        assert p.exists() and p.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest tests/test_plots.py -q`
Expected: FAIL — no `render_wfs_figures`.

- [ ] **Step 3: Write minimal implementation**

Create `seeing_summary/plots.py`:

```python
"""Figure generation for seeing summaries. Headless (Agg backend)."""
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import hist as astro_hist
from scipy.stats import lognorm

from .periods import Period

_STYLE = "ggplot"
_WFS_ORDER = [("binospec", "Binospec"), ("mmirs", "MMIRS"), ("f5", "F/5"), ("newf9", "F/9")]


def _month_groups(series):
    """Ordered {'YYYY-MM': array} for non-empty months."""
    groups = {}
    for key in sorted(set(series.index.strftime("%Y-%m"))):
        arr = np.asarray(series.loc[key].dropna(), dtype=float)
        if arr.size:
            groups[key] = arr
    return groups


def _daily_groups(series):
    """Ordered {'YYYY-MM-DD': array} for non-empty nights."""
    groups = {}
    for key in sorted(set(series.index.strftime("%Y-%m-%d"))):
        arr = np.asarray(series.loc[key].dropna(), dtype=float)
        if arr.size:
            groups[key] = arr
    return groups


def _violin(ax, groups, key_fmt, date_fmt, widths, points, ylim=None):
    labels = [datetime.strptime(k, key_fmt).date() for k in groups]
    x = mdates.date2num(labels)
    ax.violinplot(list(groups.values()), x, points=points, widths=widths,
                  showextrema=False, showmedians=True, bw_method="silverman")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    if ylim is not None:
        ax.set_ylim(*ylim)


def _lognorm_hist(values, title, out, xlabel="Seeing (arcsec)"):
    values = np.asarray(values, dtype=float)
    sigma, loc, exp_mu = lognorm.fit(values)
    x = np.arange(0.0, 4.0, 0.01)
    p = lognorm.pdf(x, sigma, loc=loc, scale=exp_mu)
    mode = np.exp(np.log(exp_mu) - sigma ** 2) + loc
    fit_median = exp_mu + loc
    median = np.nanmedian(values)
    fig = plt.figure(figsize=(8, 5))
    with plt.style.context(_STYLE):
        plt.hist(values, density=True, bins=100, range=(0.0, 4.0), alpha=0.6)
        plt.plot(x, p)
        plt.xlabel(xlabel)
        plt.ylabel("Number Density")
        plt.title(title)
        plt.legend([f'median={fit_median:.2f}", mode={mode:.2f}"', f'median={median:.2f}"'])
        fig.savefig(out)
    plt.close(fig)


def _monthly_hist(series, out):
    fig = plt.figure(figsize=(8, 5))
    legends = []
    for key, arr in _month_groups(series).items():
        label = datetime.strptime(key, "%Y-%m").strftime("%B")
        plt.hist(arr, bins=100, range=(0.0, 4.0), label=label, alpha=0.6)
        legends.append(f'{label}: {np.median(arr):.2f}"')
    plt.legend(legends)
    plt.xlabel("Seeing (arcsec)")
    plt.ylabel("N")
    fig.savefig(out)
    plt.close(fig)


def _first_second(series, out):
    first = series.between_time("00:00", "07:00")
    second = series.between_time("07:00", "14:00")
    fig = plt.figure(figsize=(8, 5))
    plt.hist(first, bins=100, range=(0.0, 4.0), alpha=0.6)
    plt.hist(second, bins=100, range=(0.0, 4.0), alpha=0.6)
    plt.legend([f'1st Half: {np.median(first):.2f}"', f'2nd Half: {np.median(second):.2f}"'])
    plt.xlabel("Seeing (arcsec)")
    plt.ylabel("N")
    fig.savefig(out)
    plt.close(fig)


def _nightly(series, out):
    fig, ax = plt.subplots()
    med = series.resample("D").median()
    lo = med - series.resample("D").min()
    hi = series.resample("D").max() - med
    ax.errorbar(med.index, med, yerr=[lo, hi], fmt="o")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%Y"))
    ax.set_ylim(0.0, 3.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    ax.set_ylabel("Seeing (arcsec)")
    fig.savefig(out)
    plt.close(fig)


def _per_instrument(df, out):
    with plt.style.context(_STYLE):
        fig = plt.figure(figsize=(8, 5))
        for key, label in _WFS_ORDER:
            vals = df["vlt_seeing"][df["wfs"] == key]
            if len(vals) == 0:
                continue
            plt.hist(vals, density=True, bins=100, range=(0.0, 4.0), alpha=0.6,
                     label=f'{label}: {np.median(vals):.2f}"')
        plt.legend()
        plt.title("Seeing by Instrument")
        plt.xlabel("Seeing (arcsec)")
        plt.ylabel("Probability Density")
        fig.savefig(out)
    plt.close(fig)


def _ellipticity_hist(df, out):
    fig = plt.figure(figsize=(8, 5))
    plt.hist(df["ellipticity"], bins=100, range=(0.0, 0.5), alpha=0.6,
             label=f'Median: {np.median(df["ellipticity"]):.2f}')
    plt.xlabel("Ellipticity")
    plt.ylabel("N")
    plt.legend()
    fig.savefig(out)
    plt.close(fig)


def _ellip_vs_inst(df, out):
    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.5, 6), sharex=True, sharey=True)
        axes = axes.flat
        fig.subplots_adjust(hspace=0)
        for ax, (key, label) in zip(axes, _WFS_ORDER):
            vals = df["ellipticity"][df["wfs"] == key]
            if len(vals):
                astro_hist(np.asarray(vals), bins="scott", ax=ax,
                           histtype="stepfilled", alpha=0.6, density=True)
                ax.legend([f'{label}: {np.median(vals):.2f}'])
            ax.set_xlim(0, 0.5)
        axes[0].set_ylabel("Probability Density")
        axes[2].set_ylabel("Probability Density")
        axes[2].set_xlabel("Ellipticity")
        axes[3].set_xlabel("Ellipticity")
        fig.tight_layout()
        fig.savefig(out)
    plt.close(fig)


def _bino_ellip_vs_el(df, out):
    bino = df[df["wfs"] == "binospec"]
    with plt.style.context(_STYLE):
        fig = plt.figure()
        if len(bino):
            els = list(range(30, 90, 5))
            e_meds = [np.median(bino["ellipticity"][(bino["el"] >= e - 2.5) & (bino["el"] <= e + 2.5)])
                      for e in els]
            plt.hist2d(bino["el"], bino["ellipticity"], bins=100, cmap="viridis",
                       norm=mcolors.PowerNorm(0.3))
            plt.scatter(els, e_meds, color="w")
        plt.xlabel("Elevation (deg)")
        plt.ylabel("Ellipticity")
        plt.title("Binospec Ellipticity vs. Elevation")
        fig.savefig(out)
    plt.close(fig)


def render_wfs_figures(df, period: Period, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = period.tag
    seeing = df["vlt_seeing"]
    ellip = df["ellipticity"]
    written = []

    def path(name):
        p = out_dir / f"{tag}_{name}.png"
        written.append(p)
        return p

    _lognorm_hist(seeing, period.date_range_str, path("hist"))
    _monthly_hist(seeing, path("monthly"))
    _first_second(seeing, path("1st2nd"))
    _nightly(seeing, path("nightly"))

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(11, 5))
        _violin(ax, _daily_groups(seeing), "%Y-%m-%d", "%m-%d-%Y", widths=1.5, points=50)
        ax.set_ylabel("Seeing (arcsec)")
        fig.autofmt_xdate()
        fig.savefig(path("violin"))
    plt.close(fig)

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(11, 5))
        _violin(ax, _month_groups(seeing), "%Y-%m", "%b", widths=15, points=100, ylim=(0.0, 3.5))
        ax.set_ylabel("Seeing (arcsec)")
        ax.set_title(f"{period.title} Monthly WFS Seeing Statistics")
        fig.autofmt_xdate()
        fig.savefig(path("violin_monthly"))
    plt.close(fig)

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(11, 5))
        _violin(ax, _daily_groups(ellip), "%Y-%m-%d", "%m-%d-%Y", widths=1.5, points=50, ylim=(0, 0.5))
        ax.set_ylabel("Ellipticity")
        fig.autofmt_xdate()
        fig.savefig(path("ellip_violin"))
    plt.close(fig)

    _per_instrument(df, path("per_instrument"))
    _ellipticity_hist(df, path("ellipticity"))
    _ellip_vs_inst(df, path("ellip_vs_inst"))
    _bino_ellip_vs_el(df, path("bino_ellip_vs_el"))
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest tests/test_plots.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add seeing_summary/plots.py tests/test_plots.py
git commit -m "feat: add WFS figure rendering"
```

---

### Task 4: `plots.py` — cyclop figures

**Files:**
- Modify: `seeing_summary/plots.py` (append cyclop functions + `render_cyclop_figures`)
- Modify: `tests/test_plots.py` (add cyclop test)

**Interfaces:**
- Consumes: WFS `df` (for the per-instrument comparison), a `cyclop` DataFrame with a `seeing` column and a DatetimeIndex, and `Period`. Reuses `_month_groups`, `_daily_groups`, `_violin`, `_lognorm_hist` from Task 3.
- Produces: `render_cyclop_figures(df, cyclop, period, out_dir) -> list[Path]`. Writes 7 PNGs named `{tag}_cyclop_{figure}.png`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_plots.py`:

```python
@pytest.fixture
def cyclop_df():
    idx = pd.date_range("2025-10-01T02:00:00", "2025-12-30T10:00:00", periods=2000)
    rng = np.random.default_rng(1)
    return pd.DataFrame({"seeing": np.abs(rng.normal(1.0, 0.35, len(idx))) + 0.2},
                        index=idx)


def test_render_cyclop_figures(wfs_df, cyclop_df, tmp_path):
    period = parse_period("2025q4")
    written = plots.render_cyclop_figures(wfs_df, cyclop_df, period, tmp_path)
    names = {p.name for p in written}
    assert names == {
        "2025_q4_cyclop_hist.png", "2025_q4_cyclop_monthly.png",
        "2025_q4_cyclop_1st2nd.png", "2025_q4_cyclop_nightly.png",
        "2025_q4_cyclop_violin.png", "2025_q4_cyclop_violin_monthly.png",
        "2025_q4_cyclop_vs_inst.png",
    }
    for p in written:
        assert p.exists() and p.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest tests/test_plots.py::test_render_cyclop_figures -q`
Expected: FAIL — no `render_cyclop_figures`.

- [ ] **Step 3: Write minimal implementation**

Append to `seeing_summary/plots.py`:

```python
def _cyclop_vs_inst(df, cyclop, out):
    pairs = [
        (df[df["wfs"] == "binospec"], "Binospec"),
        (df[df["wfs"] == "f5"], "F/5"),
        (df[df["wfs"] == "mmirs"], "MMIRS"),
        (df[df["wfs"] == "newf9"], "F/9"),
    ]
    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.5, 6), sharex=True, sharey=True)
        axes = axes.flat
        fig.subplots_adjust(hspace=0)
        for ax, (sub, label) in zip(axes, pairs):
            if len(sub):
                nights = sorted(set(sub.index.strftime("%Y-%m-%d")))
                cyc_nights = [np.asarray(cyclop.loc[n]["seeing"]) for n in nights
                              if n in cyclop.index.strftime("%Y-%m-%d")]
                cyc = np.hstack(cyc_nights) if cyc_nights else np.array([])
                astro_hist(np.asarray(sub["vlt_seeing"]), bins="scott", ax=ax,
                           histtype="stepfilled", alpha=0.6, density=True)
                legend = [f'{label}: {np.median(sub["vlt_seeing"]):.2f}']
                if cyc.size:
                    astro_hist(cyc, bins="scott", ax=ax, histtype="stepfilled",
                               alpha=0.6, density=True)
                    legend.append(f"Cyclop: {np.median(cyc):.2f}")
                ax.legend(legend)
            ax.set_xlim(0, 4)
        axes[0].set_ylabel("Probability Density")
        axes[2].set_ylabel("Probability Density")
        axes[2].set_xlabel("Seeing (arcsec)")
        axes[3].set_xlabel("Seeing (arcsec)")
        fig.tight_layout()
        fig.savefig(out)
    plt.close(fig)


def render_cyclop_figures(df, cyclop, period: Period, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = period.tag
    seeing = cyclop["seeing"]
    written = []

    def path(name):
        p = out_dir / f"{tag}_cyclop_{name}.png"
        written.append(p)
        return p

    _lognorm_hist(seeing, f"Seeing Monitor: {period.date_range_str}", path("hist"))
    _monthly_hist(seeing, path("monthly"))
    _first_second(seeing, path("1st2nd"))
    _nightly(seeing, path("nightly"))

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(11, 5))
        _violin(ax, _daily_groups(seeing), "%Y-%m-%d", "%m-%d-%Y", widths=1.5, points=50, ylim=(0.0, 3.5))
        ax.set_ylabel("Seeing (arcsec)")
        fig.autofmt_xdate()
        fig.savefig(path("violin"))
    plt.close(fig)

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(11, 5))
        _violin(ax, _month_groups(seeing), "%Y-%m", "%b", widths=15, points=100, ylim=(0.0, 3.5))
        ax.set_ylabel("Seeing (arcsec)")
        ax.set_title(f"{period.title} Monthly Seeing Monitor Statistics")
        fig.autofmt_xdate()
        fig.savefig(path("violin_monthly"))
    plt.close(fig)

    _cyclop_vs_inst(df, cyclop, path("vs_inst"))
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest tests/test_plots.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add seeing_summary/plots.py tests/test_plots.py
git commit -m "feat: add cyclop (seeing monitor) figure rendering"
```

---

### Task 5: `__main__.py` — CLI wiring

**Files:**
- Create: `seeing_summary/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `parse_period`, `data.load_wfs`, `data.warn_coverage`, `data.load_cyclop`, `data.NoDataError`, `data.CyclopUnavailable`, `plots.render_wfs_figures`, `plots.render_cyclop_figures`.
- Produces: `main(argv: list[str] | None = None) -> int`. CLI: `python -m seeing_summary PERIOD [--data-dir DIR] [--out-dir DIR] [--no-cyclop]`. Writes figures to `<out-dir>/<year>/`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seeing_summary.__main__ import main


def _write_night(root, day, n=40):
    d = root / day
    d.mkdir(parents=True)
    ts = pd.date_range(f"{day[:4]}-{day[4:6]}-{day[6:]}T02:00:00", periods=n, freq="min")
    rng = np.random.default_rng(int(day))
    pd.DataFrame({
        "time": ts.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "wfs": rng.choice(["binospec", "mmirs", "f5", "newf9"], n),
        "el": rng.uniform(30, 85, n),
        "seeing": np.abs(rng.normal(1.0, 0.3, n)) + 0.2,
        "vlt_seeing": np.abs(rng.normal(0.9, 0.3, n)) + 0.2,
        "ellipticity": np.abs(rng.normal(0.15, 0.05, n)),
        "fwhm": np.abs(rng.normal(4.0, 0.5, n)) + 1.0,
    }).to_csv(d / "reanalyze_results.csv", index=False)


def test_cli_writes_wfs_figures(tmp_path):
    data_dir = tmp_path / "data"
    for day in ["20251005", "20251115", "20251231"]:
        _write_night(data_dir, day)
    out = tmp_path / "images"
    rc = main(["2025q4", "--data-dir", str(data_dir), "--out-dir", str(out), "--no-cyclop"])
    assert rc == 0
    produced = list((out / "2025").glob("2025_q4_*.png"))
    assert len(produced) == 11


def test_cli_no_data_exits_nonzero(tmp_path):
    rc = main(["2025q4", "--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "img"), "--no-cyclop"])
    assert rc != 0


def test_cli_bad_period(tmp_path):
    with pytest.raises(SystemExit):
        main(["2022_2ndhalf", "--data-dir", str(tmp_path)])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest tests/test_cli.py -q`
Expected: FAIL — no `seeing_summary.__main__.main`.

- [ ] **Step 3: Write minimal implementation**

Create `seeing_summary/__main__.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest tests/test_cli.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full suite and commit**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest tests/ -q`
Expected: PASS (all tests).

```bash
git add seeing_summary/__main__.py tests/test_cli.py
git commit -m "feat: add seeing_summary CLI entry point"
```

---

### Task 6: Makefile

**Files:**
- Create: `Makefile`

**Interfaces:**
- Produces: `make <period>` (e.g. `make 2026q2`), `make test`, `make help` (default).

- [ ] **Step 1: Create the Makefile**

Create `Makefile`:

```make
PYTHON ?= /Users/tim/conda/envs/mmtwfs/bin/python

.PHONY: help
help:
	@echo "Usage:"
	@echo "  make <period>   generate figures, e.g. make 2026q2  or  make 2025"
	@echo "  make test       run the unit tests"

.PHONY: test
test:
	$(PYTHON) -m pytest tests/

# Period specs (YYYY or YYYYqN) all start with "20".
# No file named after the target is ever created, so this re-runs every time.
20%:
	$(PYTHON) -m seeing_summary $@
```

- [ ] **Step 2: Verify `make test` runs the suite**

Run: `make test`
Expected: pytest runs and all tests pass.

- [ ] **Step 3: Verify `make help` and default target**

Run: `make`
Expected: prints the usage lines (does not error).

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "feat: add Makefile wrapper (make <period>, make test)"
```

---

### Task 7: End-to-end smoke test against real data

**Files:** none (verification only)

- [ ] **Step 1: Try to make minicyclop importable (optional)**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -c "import minicyclop" 2>/dev/null && echo HAVE || /Users/tim/conda/envs/mmtwfs/bin/pip install -e ~/MMT/minicyclop`
Expected: either already importable, or installs cleanly. If the install fails, continue — the smoke test uses `--no-cyclop`.

- [ ] **Step 2: Generate a fully-covered past quarter**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m seeing_summary 2025q4 --no-cyclop`
Expected: prints 11 `wrote images/2025/2025_q4_*.png` lines, no warning about coverage (2025q4 is complete).

- [ ] **Step 3: Verify the 11 WFS figures exist and are non-empty**

Run:
```bash
ls -l images/2025/2025_q4_{hist,monthly,1st2nd,nightly,violin,violin_monthly,ellip_violin,per_instrument,ellipticity,ellip_vs_inst,bino_ellip_vs_el}.png
```
Expected: all 11 files listed, each > 0 bytes. (Compare names against the archived `Seeing_2025_q4.ipynb` output to confirm parity.)

- [ ] **Step 4: If cyclop is available, generate the full set and the current quarter**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m seeing_summary 2025q4`
Expected: 18 figures if cyclop is available; otherwise a `skipping cyclop figures` warning and 11.

Run: `make 2026q2`
Expected: figures written to `images/2026/`; a `WARNING: data for 2026q2 ... short of the period end` if the quarter is not yet complete.

- [ ] **Step 5: Commit the generated figures for the smoke-tested quarter**

```bash
git add images/2025/2025_q4_*.png
git commit -m "test: regenerate 2025q4 figures via seeing_summary generator"
```

---

### Task 8: Reorganize images into `images/<year>/`

**Files:**
- Move: all tracked root-level `*.png` / `*.pdf` into `images/<year>/` (year-less into `images/misc/`) via `git mv`.

- [ ] **Step 1: Move tracked PNG/PDF files by year**

Run:
```bash
git ls-files -z '*.png' '*.pdf' | while IFS= read -r -d '' f; do
  case "$f" in images/*) continue;; esac
  base=$(basename "$f")
  yr=$(printf '%s' "$base" | grep -oE '(19|20)[0-9]{2}' | head -1)
  dest="images/${yr:-misc}"
  mkdir -p "$dest"
  git mv "$f" "$dest/$base"
done
```
Expected: no errors. (Files already under `images/` — e.g. the 2025q4 figures from Task 7 — are skipped.)

- [ ] **Step 2: Verify the repo root has no stray images**

Run: `ls *.png *.pdf 2>/dev/null | wc -l`
Expected: `0`.

- [ ] **Step 3: Spot-check the year buckets and the misc catch-all**

Run:
```bash
ls images/ | head
ls images/misc/ | head
git ls-files 'images/**' | wc -l
```
Expected: year directories (`2014`…`2026`) plus `misc`; `misc` holds year-less files (`all_hist.png`, `airmass_corr.png`, `seeing_ambient.png`, `bino_vs_mmirs.pdf`, `hdimm_cyclop_wfs.png`, …); the tracked image count matches the pre-move total.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: move images into images/<year>/ (year-less into images/misc/)"
```

---

### Task 9: Archive notebooks into `notebooks/`

**Files:**
- Move: all tracked root-level `*.ipynb` into `notebooks/` via `git mv`.

- [ ] **Step 1: Move tracked notebooks**

Run:
```bash
mkdir -p notebooks
git ls-files -z '*.ipynb' | while IFS= read -r -d '' f; do
  case "$f" in notebooks/*) continue;; esac
  git mv "$f" "notebooks/$(basename "$f")"
done
```
Expected: no errors (filenames with spaces such as `Seeing Summary - 2017 Q2.ipynb` are handled by the null-delimited loop).

- [ ] **Step 2: Verify no notebooks remain in the root**

Run: `git ls-files '*.ipynb' | grep -v '^notebooks/' | wc -l`
Expected: `0`.

- [ ] **Step 3: Verify the count is preserved**

Run: `git ls-files 'notebooks/*.ipynb' | wc -l`
Expected: `51`.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: archive notebooks into notebooks/"
```

---

### Task 10: Update CLAUDE.md and README

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Add a "Generating summaries" section to CLAUDE.md**

Insert the following section into `CLAUDE.md` immediately after the `## Data flow` section:

```markdown
## Generating quarterly / yearly summaries

Use the `seeing_summary` package instead of copying a notebook. Run everything
in the `mmtwfs` conda env.

```
make 2026q2          # a quarter
make 2025            # a full year
make test            # run the unit tests
```

`make <period>` calls `python -m seeing_summary <period>`. The generator:
- auto-discovers `data/YYYYMMDD/reanalyze_results.csv` files whose date falls in
  the period (no more manual `reanalyze_csvs_*.txt` list files),
- applies the standard filter (`seeing` finite, `fwhm > 0`, `0 < seeing < 4`) and
  plots the `vlt_seeing` column,
- writes standardized figures to `images/<year>/{tag}_{figure}.png`
  (`tag` is `2025_q4` for quarters, `2025` for years),
- warns if the data does not reach the end of the period (in-progress quarter)
  and errors if the period has no data at all,
- appends MiniCyclop seeing-monitor comparison figures when `minicyclop` is
  importable (pass `--no-cyclop`, or it is skipped with a warning otherwise).

Package layout: `seeing_summary/periods.py` (spec parsing),
`data.py` (discovery/loading/coverage/cyclop), `plots.py` (one function per
figure), `__main__.py` (CLI).
```

- [ ] **Step 2: Update the repo-layout notes in CLAUDE.md**

In `CLAUDE.md`, update the notebook/hygiene guidance to reflect the new layout. Replace the `## Editing notebooks` section body's first sentence and the "Notebook hygiene" note about PNG output location with:

```markdown
- Committed figures now live in `images/<year>/` (year-less legacy figures in
  `images/misc/`); the generator writes there. Historical per-quarter/year
  notebooks are archived under `notebooks/`.
```

(Keep the existing NotebookEdit guidance for the archived notebooks.)

- [ ] **Step 3: Update README.md**

Replace `README.md` with:

```markdown
# MMT seeing analysis

Notebooks, scripts, and figures for analyzing MMT WFS seeing measurements.

Generate a quarterly or yearly summary (in the `mmtwfs` conda env):

    make 2026q2
    make 2025

Figures are written to `images/<year>/`. Historical notebooks are in
`notebooks/`. See `CLAUDE.md` for details.
```

- [ ] **Step 4: Verify the docs render sanely**

Run: `git diff --stat CLAUDE.md README.md`
Expected: both files modified.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document seeing_summary generator and new repo layout"
```

---

## Self-Review notes

- **Spec coverage:** period parsing (Task 1) ✓; auto-discovery + filter + `ut`
  index (Task 2) ✓; coverage warning + no-data exception (Task 2, wired in
  Task 5) ✓; cyclop loader with skip-on-unavailable (Tasks 2 & 5) ✓; all 18
  standardized figures — 11 WFS (Task 3) + 7 cyclop (Task 4) ✓; CLI with
  `--data-dir/--out-dir/--no-cyclop` (Task 5) ✓; Makefile `make <period>` +
  `make test` + `help` (Task 6) ✓; smoke test vs a real quarter (Task 7) ✓;
  images → `images/<year>/` incl. `misc` catch-all (Task 8) ✓; notebooks →
  `notebooks/` (Task 9) ✓; CLAUDE.md/README updates (Task 10) ✓.
- **Standardized names:** the plan uses `{tag}_hist.png` (was bare `2025_q4.png`
  / `2024_allyear.png`) and `{tag}_cyclop_nightly.png` (was the misnamed
  `*_cyclop_plot_monthly.png`), matching the spec's naming decision.
- **Type consistency:** `Period` fields/properties, `render_wfs_figures` /
  `render_cyclop_figures` signatures, and `NoDataError` / `CyclopUnavailable`
  names are used identically across tasks.
```
