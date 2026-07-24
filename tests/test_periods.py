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
    assert p.quarter == 4
    assert p.subdir == "q4"
    assert p.date_range_str == "2025-10-01 through 2025-12-31"
    assert p.month_keys == ["2025-10", "2025-11", "2025-12"]


def test_year():
    p = parse_period("2025")
    assert p.start == pd.Timestamp(2025, 1, 1)
    assert p.end == pd.Timestamp(2026, 1, 1)
    assert p.tag == "2025"
    assert p.title == "2025"
    assert p.year == 2025
    assert p.quarter is None
    assert p.subdir == "annual"
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
