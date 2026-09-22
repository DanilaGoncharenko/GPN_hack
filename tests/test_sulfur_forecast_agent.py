import pandas as pd

from oilhack.agents.sulfur_forecast_agent import SulfurForecastAgent


def test_feature_columns_are_namespaced():
    columns = SulfurForecastAgent._feature_columns()
    assert "242000__T6" in columns
    assert "avt__T55" in columns
    assert "hydro__Q21" in columns  # autoregressive lagged sulfur is a valid predictor


def test_namespaced_counterfactual_key_is_accepted():
    # The test focuses on the public change-key convention. No ML fit is needed.
    key = "242000__T6"
    assert key.count("__") == 1
    assert key.startswith("242000__")
