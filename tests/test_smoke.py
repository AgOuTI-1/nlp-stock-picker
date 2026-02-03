import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import pandas as pd
from stockpicker.data.universe import build_universe

def test_universe_builds():
    tickers, t2s = build_universe()
    assert len(tickers) > 20
    assert len(tickers) == len(t2s)
    
