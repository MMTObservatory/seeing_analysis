# Seeing-summary generalization & repo cleanup — design

Date: 2026-07-23

## Problem

Atmospheric-seeing analysis for the MMT WFS systems is currently done by copying
a per-quarter (or per-year) Jupyter notebook, editing hardcoded dates, month
names, and output filenames, and re-running it. This produces ~18 figures per
period. The repo root has accumulated ~419 PNGs/PDFs and ~51 notebooks with
inconsistent naming.

Goals:
1. Replace the copy-a-notebook workflow with a single script that generates the
   full figure set for any quarter or year.
2. Organize all existing images into `images/<year>/`.
3. Archive all notebooks into `notebooks/`.

## Environment

All Python runs in the `mmtwfs` conda env
(`/Users/tim/conda/envs/mmtwfs/bin/python`). `minicyclop` may need
`pip install -e ~/MMT/minicyclop` into that env before cyclop features work.

## Key insight

The per-night `data/YYYYMMDD/reanalyze_results.csv` files already contain every
column the figures need: `time, wfs, el, seeing, vlt_seeing, ellipticity, fwhm,
exptime, ...`. So the generator needs **no** `mmtwfs` dependency. The only
external dependency is `minicyclop.io.read_seeing_data` for the seeing-monitor
(cyclop) comparison, isolated to one function so it can be swapped/mocked.

## Architecture

A small importable package, run with no install step:

```
seeing_summary/
  __init__.py
  periods.py    # parse a period spec -> date range, month list, labels, filename tag
  data.py       # discover + load + filter WFS CSVs; load cyclop for a range
  plots.py      # one function per figure; each takes prepared data + out path
  __main__.py   # argparse CLI
```

Invocation:

```
/Users/tim/conda/envs/mmtwfs/bin/python -m seeing_summary 2025q4
/Users/tim/conda/envs/mmtwfs/bin/python -m seeing_summary 2025
```

Chosen over a single flat script (keeps ~18 plot functions readable) and over a
full pyproject/console-script (repo has no packaging today — YAGNI).

### periods.py

`parse_period(spec) -> Period` where `Period` carries:
- `start`, `end` — half-open `[start, end)` datetimes (UTC).
  - `2025q4` -> `2025-10-01` .. `2026-01-01`
  - `2025`   -> `2025-01-01` .. `2026-01-01`
- `months` — list of `(YYYY-MM, label)` for month grouping (3 for a quarter, 12
  for a year). Month labels come from the timestamps, not hardcoded.
- `tag` — filename stem component: `2025_q4` for quarters, `2025` for years.
- `title` — human title: `"2025 Q4"`, `"2025"`.
- `date_range_str` — e.g. `"2025-10-01 through 2025-12-31"` for plot titles.

Supported specs: `YYYYqN` and `YYYY`. Anything else -> clear error. (Half-year
specs like the one-off `2022_2ndhalf` are out of scope; that notebook stays
archived.)

### data.py

- `discover_csvs(data_dir, period)` — glob `data_dir/YYYYMMDD/reanalyze_results.csv`,
  parse the date from the directory name, keep those with date in `[start, end)`.
- `load_wfs(data_dir, period)` — concat discovered CSVs, apply the standard
  filter (`seeing` finite, `fwhm > 0`, `0 < seeing < 4`), set a
  `DatetimeIndex` named `ut` from the `time` column. Returns the DataFrame.
- `load_cyclop(period)` — `read_seeing_data(...)` then slice to `[start, end)`.
  Path defaults to `~/MMT/minicyclop/data/MiniCyclop/Data/Seeing_Data.txt`.

### plots.py

One function per figure. Each takes already-loaded/filtered data plus an output
`Path` and writes a PNG. The `seeing` value plotted is `vlt_seeing`
(zenith-corrected). Month grouping, nightly resampling, violins, and the
first/second-half split (`between_time` 00:00–07:00 vs 07:00–14:00 UT) all derive
from the DatetimeIndex, so they adapt to quarter vs year automatically.

Figure set (the current quarterly set, standardized names under `images/<year>/`
with stem `{tag}`):

WFS figures:
- `{tag}_hist.png` — normalized hist + log-normal fit (median/mode in legend)
- `{tag}_monthly.png` — per-month histograms
- `{tag}_1st2nd.png` — first vs second half of night
- `{tag}_nightly.png` — nightly median errorbars (min/max whiskers)
- `{tag}_violin.png` — nightly violin (seeing)
- `{tag}_violin_monthly.png` — monthly violin (seeing)
- `{tag}_ellip_violin.png` — nightly violin (ellipticity)
- `{tag}_per_instrument.png` — per-instrument seeing hist (binospec/mmirs/f5/newf9)
- `{tag}_ellipticity.png` — ellipticity histogram
- `{tag}_ellip_vs_inst.png` — 2x2 per-instrument ellipticity
- `{tag}_bino_ellip_vs_el.png` — binospec ellipticity vs elevation (2d hist + medians)

Cyclop figures:
- `{tag}_cyclop_hist.png` — cyclop hist + log-normal fit
- `{tag}_cyclop_monthly.png` — per-month cyclop histograms
- `{tag}_cyclop_1st2nd.png` — cyclop first vs second half
- `{tag}_cyclop_nightly.png` — cyclop nightly errorbars
  (replaces the misnamed `*_cyclop_plot_monthly.png`)
- `{tag}_cyclop_violin.png` — cyclop nightly violin
- `{tag}_cyclop_violin_monthly.png` — cyclop monthly violin
- `{tag}_cyclop_vs_inst.png` — 2x2 WFS-vs-cyclop comparison (matched nights)

Naming is **standardized** (consistent `{tag}_{figure}.png` for both quarters
and years). This intentionally diverges from a few legacy quirks (bare
`2025_q4.png`, `2024_allyear.png`, `2025q4_bino_ellip_vs_el.png`); legacy images
keep their original names in the archive.

### __main__.py (CLI)

```
python -m seeing_summary PERIOD [--data-dir DIR] [--out-dir DIR] [--no-cyclop]
```

- `PERIOD` — e.g. `2025q4` or `2025`.
- `--data-dir` — default `./data`.
- `--out-dir` — default `./images`; figures land in `<out-dir>/<year>/`.
- `--no-cyclop` — skip the seeing-monitor figures (e.g. when the log is missing
  or minicyclop isn't installed).

Behavior: parse period -> load WFS -> ensure `images/<year>/` exists -> write WFS
figures -> if cyclop available, write cyclop figures. Prints each file written
and a final summary line.

### Makefile wrapper

Primary user-facing entry point so a period can be generated with just its spec:

```
make 2026q2
make 2025
```

Implementation uses a pattern rule on the `20` prefix (every supported period
spec starts with `20`), which pins the wildcard so it does not shadow named
targets like `help`:

```make
PYTHON ?= /Users/tim/conda/envs/mmtwfs/bin/python

.PHONY: help
help:
	@echo "Usage: make <period>   e.g. make 2026q2  or  make 2025"

# Period specs (YYYY or YYYYqN) all start with "20".
# No file named after the target is ever created, so this re-runs every time.
20%:
	$(PYTHON) -m seeing_summary $@
```

The target file (`2026q2`) is never created, so the rule re-runs every time.
(`.PHONY` is intentionally not used here — it does not accept pattern targets;
the never-created-file behavior gives the same always-run effect.)
Extra CLI flags remain available via the underlying `python -m seeing_summary`
call for one-off cases; `make` covers the common path.

## Error handling

- Unknown period spec -> `SystemExit` with a clear message.
- **Period not covered at all** -> raise an exception. If `discover_csvs` finds
  no `reanalyze_results.csv` with a date in `[start, end)` (or all discovered
  files are empty / filtered away to zero rows), abort with a message naming the
  requested range and the data dir. This is a hard error, not a warning.
- **Data does not reach the end of the period** -> emit a warning and continue.
  After loading, compute the max night present in the data. The period's last
  covered day is `end - 1 day` (the range is half-open). If
  `max_night < last_day`, warn, e.g.:
  `WARNING: data for 2026q2 ends 2026-05-14, 47 days short of the period end
  2026-06-30; figures cover a partial period.` This is the expected signal when
  generating an in-progress quarter. (A symmetric note is logged if the first
  night is later than the period start, but only the end-coverage case is a
  warning per the requirement.)
- Cyclop log missing / `minicyclop` not importable -> warn and skip cyclop
  figures (do not abort the WFS figures); `--no-cyclop` forces this.
- Empty per-night slices (nights with no data) are dropped, matching current
  notebook behavior.

## Testing

Lightweight, since this is an analysis repo with no existing test suite:
- Unit-test `periods.parse_period` for `2025q4` and `2025` (date bounds, tag,
  months, title).
- Unit-test `discover_csvs` date filtering against a small temp tree of empty
  dated dirs.
- Smoke test: run the CLI against a real recent quarter in the `mmtwfs` env and
  confirm the expected PNG files appear in `images/<year>/` and are non-empty.
  Compare figure count/names to the archived quarterly notebook's output.

## Repo reorganization (one-time, via `git mv`)

- **Images** -> `images/<year>/` for any tracked PNG/PDF with a 4-digit year
  anywhere in its name (`ellipticity_2024.png`, `bino_vs_mmirs_2022q3.png`,
  `per_instrument_2024_q1.png` -> their respective years). Year-less files
  (`all_*.png/pdf`, `airmass_corr.png`, `seeing_ambient.*`, `hdimm_*`,
  `bino_vs_mmirs.pdf`, etc.) -> `images/misc/`. Basenames unchanged.
- **Notebooks** -> all `.ipynb` into `notebooks/` (flat).
- **Kept in root**: the pipeline `.py` scripts (`dome_seeing.py`,
  `parse_clear.py`, `spot_reduce.py`, `spot_seeing.py`, `fix_wfs.py`), `README`,
  `LICENSE`, `CLAUDE.md`, and the new `seeing_summary/` package.
- **CSV list files** (`data/reanalyze_csvs_*.txt`) are kept (gitignored anyway);
  auto-discovery makes them unnecessary but they are harmless to leave.
- Large data CSVs in root (`all_seeing.csv*`, `seeing_2017*.csv`, etc.) are out
  of scope for this change.

## Documentation

Update `CLAUDE.md` to describe:
- the new `seeing_summary` package and how to run it (mmtwfs env, period specs),
- the new `images/<year>/` and `notebooks/` layout,
- that auto-discovery replaces the manual `reanalyze_csvs_*.txt` list files for
  the summary generator.

## Out of scope

- The reduction step that produces `reanalyze_results.csv` (upstream, not in repo).
- All-time aggregate (`all_seeing.ipynb`) and special-purpose notebooks
  (`spie_seeing`, `fit WFS spots`, `wfs_vs_seeing_monitor`) — archived, not ported.
- Half-year and other non-standard period specs.
- Regenerating or relinking the external QSUM PDFs in `~/iCloudDrive/MMTO/QSUMs`.
