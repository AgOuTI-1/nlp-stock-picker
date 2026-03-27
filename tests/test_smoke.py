import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockpicker.data.universe import build_universe


def test_universe_builds():
    tickers, t2name, t2sector = build_universe()
    assert len(tickers) > 20
    assert len(tickers) == len(t2name)
