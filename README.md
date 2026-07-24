# MMT seeing analysis

Notebooks, scripts, and figures for analyzing MMT WFS seeing measurements.

Generate a quarterly or yearly summary (in the `mmtwfs` conda env):

    make 2026q2
    make 2025

Figures are written to `images/<year>/<subdir>/` — `q1`..`q4` for a quarter,
`annual` for a full year (e.g. `images/2026/q2/hist.png`). Historical notebooks
are in `notebooks/`. See `CLAUDE.md` for details.
