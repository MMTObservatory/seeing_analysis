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
