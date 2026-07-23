from pathlib import Path

import matplotlib.pyplot as plt
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


def test_module_applies_ggplot_style():
    # Importing the module must put matplotlib into the ggplot style so every
    # figure is consistent (not a mix of ggplot and the default style).
    assert plt.rcParams["axes.facecolor"] == plt.style.library["ggplot"]["axes.facecolor"]


def test_instrument_grid_layouts():
    # 4 instruments -> 2x2 grid; fewer -> a single vertical column.
    fig, axes = plots._instrument_grid(4)
    assert len(axes) == 4
    assert axes[0].get_subplotspec().get_gridspec().get_geometry() == (2, 2)
    plt.close(fig)

    fig, axes = plots._instrument_grid(3)
    assert len(axes) == 3
    assert axes[0].get_subplotspec().get_gridspec().get_geometry() == (3, 1)
    plt.close(fig)

    fig, axes = plots._instrument_grid(1)
    assert len(axes) == 1
    assert axes[0].get_subplotspec().get_gridspec().get_geometry() == (1, 1)
    plt.close(fig)


@pytest.fixture
def three_wfs_df():
    # Only three instruments present (no newf9), to exercise the vertical layout.
    idx = pd.date_range("2025-10-02T02:00:00", "2025-12-20T09:00:00", periods=600)
    rng = np.random.default_rng(2)
    n = len(idx)
    wfs = rng.choice(["binospec", "mmirs", "f5"], size=n)
    return pd.DataFrame({
        "vlt_seeing": np.abs(rng.normal(0.9, 0.3, n)) + 0.2,
        "seeing": np.abs(rng.normal(1.0, 0.3, n)) + 0.2,
        "ellipticity": np.abs(rng.normal(0.15, 0.05, n)),
        "el": rng.uniform(30, 85, n),
        "wfs": wfs,
    }, index=pd.DatetimeIndex(idx, name="ut"))


def test_render_wfs_figures_three_instruments(three_wfs_df, tmp_path):
    # The per-instrument panels use a vertical layout here; all 11 figures must
    # still render without error and be non-empty.
    period = parse_period("2025q4")
    written = plots.render_wfs_figures(three_wfs_df, period, tmp_path)
    assert len(written) == 11
    for p in written:
        assert p.exists() and p.stat().st_size > 0


def test_render_cyclop_figures_three_instruments(three_wfs_df, cyclop_df, tmp_path):
    period = parse_period("2025q4")
    written = plots.render_cyclop_figures(three_wfs_df, cyclop_df, period, tmp_path)
    assert len(written) == 7
    for p in written:
        assert p.exists() and p.stat().st_size > 0
