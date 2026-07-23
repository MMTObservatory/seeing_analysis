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
