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
