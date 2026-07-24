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
# Apply ggplot globally so every figure is consistent. The per-figure
# `plt.style.context(_STYLE)` blocks below restore to this same style on exit,
# so figures rendered without an explicit context are ggplot too.
plt.style.use(_STYLE)

_WFS_ORDER = [("binospec", "Binospec"), ("mmirs", "MMIRS"), ("f5", "F/5"), ("newf9", "F/9")]


def _instrument_grid(n):
    """Return ``(fig, axes_list)`` for ``n`` per-instrument panels.

    Four instruments use the familiar 2x2 grid; any other count is laid out as
    a single vertical column (e.g. three instruments -> three stacked panels).
    Axes share x and y.
    """
    n = max(n, 1)
    if n == 4:
        fig, axes = plt.subplots(2, 2, figsize=(7.5, 6), sharex=True, sharey=True)
        axes = list(axes.flat)
    else:
        fig, axes = plt.subplots(n, 1, figsize=(6, 2.8 * n), sharex=True, sharey=True)
        axes = [axes] if n == 1 else list(axes)
    fig.subplots_adjust(hspace=0)
    return fig, axes


def _label_panels(axes, xlabel, ylabel):
    """Label the bottom-row axes with ``xlabel`` and left-column axes with
    ``ylabel``, for either the 2x2 grid or a vertical column."""
    if len(axes) == 4:
        left, bottom = (axes[0], axes[2]), (axes[2], axes[3])
    else:
        left, bottom = tuple(axes), (axes[-1],)
    for ax in left:
        ax.set_ylabel(ylabel)
    for ax in bottom:
        ax.set_xlabel(xlabel)


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
    present = [(k, l) for k, l in _WFS_ORDER if len(df["ellipticity"][df["wfs"] == k])]
    if not present:
        present = [_WFS_ORDER[0]]
    fig, axes = _instrument_grid(len(present))
    for ax, (key, label) in zip(axes, present):
        vals = df["ellipticity"][df["wfs"] == key]
        astro_hist(np.asarray(vals), bins="scott", ax=ax,
                   histtype="stepfilled", alpha=0.6, density=True)
        ax.legend([f'{label}: {np.median(vals):.2f}'])
        ax.set_xlim(0, 0.5)
    _label_panels(axes, "Ellipticity", "Probability Density")
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
    seeing = df["vlt_seeing"]
    ellip = df["ellipticity"]
    written = []

    def path(name):
        # Figures are written bare (e.g. hist.png); the period is encoded by the
        # containing images/<year>/<subdir>/ directory, not the filename.
        p = out_dir / f"{name}.png"
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


_CYCLOP_ORDER = [("binospec", "Binospec"), ("f5", "F/5"), ("mmirs", "MMIRS"), ("newf9", "F/9")]


def _cyclop_vs_inst(df, cyclop, out):
    present = [(k, l) for k, l in _CYCLOP_ORDER if len(df[df["wfs"] == k])]
    if not present:
        present = [_CYCLOP_ORDER[0]]
    cyclop_days = set(cyclop.index.strftime("%Y-%m-%d"))
    fig, axes = _instrument_grid(len(present))
    for ax, (key, label) in zip(axes, present):
        sub = df[df["wfs"] == key]
        nights = sorted(set(sub.index.strftime("%Y-%m-%d")))
        cyc_nights = [np.asarray(cyclop.loc[n]["seeing"]) for n in nights if n in cyclop_days]
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
    _label_panels(axes, "Seeing (arcsec)", "Probability Density")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def render_cyclop_figures(df, cyclop, period: Period, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeing = cyclop["seeing"]
    written = []

    def path(name):
        p = out_dir / f"cyclop_{name}.png"
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
