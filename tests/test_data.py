import datetime as dt

import pandas as pd
import pytest

from opeval import data

START = dt.date(2021, 1, 4)
END = dt.date(2021, 6, 30)


def _rows(first="2020-01-01", last="2021-12-31"):
    days = pd.bdate_range(first, last)
    return [
        {"date": f"{d:%Y-%m-%d}T00:00:00.000Z", "close": 100.0 + i, "adjClose": 99.0 + i}
        for i, d in enumerate(days)
    ]


@pytest.fixture
def tiingo(monkeypatch, tmp_path):
    """Stub Tiingo: records each call, serves QQQ, and treats index symbols as unknown (404)."""
    monkeypatch.setenv("OPEVAL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    data._MEMO.clear()
    state = {"day": dt.date(2026, 11, 13), "calls": [], "fail_next": False}

    def fake_request(symbol, token):
        assert token == "test-token"
        state["calls"].append(symbol)
        if state["fail_next"]:
            state["fail_next"] = False
            raise ValueError("Could not reach Tiingo")
        if symbol.startswith("^"):
            return None
        return _rows()

    monkeypatch.setattr(data, "_tiingo_request", fake_request)
    monkeypatch.setattr(data, "_today", lambda: state["day"])
    return state


def test_same_day_same_ticker_calls_tiingo_once(tiingo):
    data.fetch_market_data("QQQ", START, END)
    data.fetch_market_data("QQQ", dt.date(2020, 3, 2), dt.date(2020, 9, 1))  # different range, same day
    data.fetch_market_data("QQQ", START, END)
    assert tiingo["calls"].count("QQQ") == 1


def test_disk_cache_serves_later_processes_same_day(tiingo):
    data.fetch_market_data("QQQ", START, END)
    data._MEMO.clear()  # simulate a fresh process
    data.fetch_market_data("QQQ", START, END)
    assert tiingo["calls"].count("QQQ") == 1


def test_new_day_fetches_again(tiingo):
    data.fetch_market_data("QQQ", START, END)
    tiingo["day"] = dt.date(2026, 11, 14)
    data.fetch_market_data("QQQ", START, END)
    assert tiingo["calls"].count("QQQ") == 2


def test_unknown_index_is_not_retried_same_day(tiingo):
    data.fetch_market_data("QQQ", START, END)
    data.fetch_market_data("QQQ", START, END)
    assert tiingo["calls"].count("^VIX") == 1
    assert tiingo["calls"].count("^VXN") == 1
    raw = data.fetch_market_data("QQQ", START, END)
    assert raw["vix"].isna().all()
    assert raw["irx"].isna().all()


def test_failed_request_is_not_cached(tiingo):
    tiingo["fail_next"] = True
    with pytest.raises(ValueError):
        data.fetch_market_data("QQQ", START, END)
    raw = data.fetch_market_data("QQQ", START, END)  # retried, and succeeds
    assert not raw.empty
    assert tiingo["calls"].count("QQQ") == 2


def test_range_is_sliced_with_warmup(tiingo):
    raw = data.fetch_market_data("QQQ", START, END, warmup_days=120)
    assert raw.index.min() < pd.Timestamp(START)
    assert raw.index.min() >= pd.Timestamp(START) - pd.Timedelta(days=121)
    assert raw.index.max() <= pd.Timestamp(END)
    assert {"close", "adj_close", "vix", "vxn", "irx"} <= set(raw.columns)
    assert raw["adj_close"].iloc[0] == raw["close"].iloc[0] - 1.0


def test_missing_api_key_without_cache_raises(tiingo, monkeypatch):
    monkeypatch.delenv("TIINGO_API_KEY")
    with pytest.raises(ValueError, match="TIINGO_API_KEY"):
        data.fetch_market_data("QQQ", START, END)


def test_cache_hit_needs_no_api_key(tiingo, monkeypatch):
    data.fetch_market_data("QQQ", START, END)
    monkeypatch.delenv("TIINGO_API_KEY")
    data._MEMO.clear()
    assert not data.fetch_market_data("QQQ", START, END).empty
