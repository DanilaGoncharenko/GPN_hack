import pandas as pd

from oilhack.data_io import asof_lookup, mask_sentinel
from oilhack.config import SENTINEL_BAD_VALUE


def test_asof_lookup_never_looks_into_the_future():
    df = pd.DataFrame({
        "tag": ["S", "S", "S"],
        "timestamp": pd.to_datetime(["2023-01-01 00:00", "2023-01-02 00:00", "2023-01-05 00:00"]),
        "value": [1.0, 2.0, 3.0],
    })
    res = asof_lookup(df, pd.Timestamp("2023-01-03 00:00"), filters={"tag": "S"})
    assert res.value == 2.0
    assert res.age_hours == 24.0


def test_asof_lookup_unavailable_before_first_reading():
    df = pd.DataFrame({
        "tag": ["S"],
        "timestamp": pd.to_datetime(["2023-01-05 00:00"]),
        "value": [3.0],
    })
    res = asof_lookup(df, pd.Timestamp("2023-01-01 00:00"), filters={"tag": "S"})
    assert not res.is_available


def test_mask_sentinel_replaces_bad_value_with_nan():
    df = pd.DataFrame({"A": [1.0, SENTINEL_BAD_VALUE, 3.0]})
    out = mask_sentinel(df)
    assert out["A"].isna().sum() == 1
    assert out["A"].iloc[0] == 1.0
    assert out["A"].iloc[2] == 3.0
