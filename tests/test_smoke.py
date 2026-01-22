import pandas as pd
from stockpicker.data.universe import build_universe

def test_universe_builds():
    tickers, t2s = build_universe()
    assert len(tickers) > 20
    assert len(tickers) == len(t2s)
