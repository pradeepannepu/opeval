"""Market data from Yahoo Finance (via yfinance) and model inputs built from it."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

VIX_SYMBOL = "^VIX"
VXN_SYMBOL = "^VXN"
TBILL_SYMBOL = "^IRX"  # 13-week T-bill yield, in percent

PERIODS = {
    "1 Year": 1,
    "2 Years": 2,
    "3 Years": 3,
    "5 Years": 5,
    "10 Years": 10,
    "15 Years": 15,
    "20 Years": 20,
}


def period_start(years: int, end: dt.date) -> dt.date:
    try:
        return end.replace(year=end.year - years)
    except ValueError:  # Feb 29
        return end.replace(year=end.year - years, day=28)


def _download(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start,
        end=end + dt.timedelta(days=1),  # yfinance's end is exclusive
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        # Recent yfinance versions return (field, ticker) columns even for one symbol.
        df = df.droplevel(-1, axis=1)
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def _align(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    return series.reindex(index.union(series.index)).ffill().reindex(index)


def fetch_market_data(ticker: str, start: dt.date, end: dt.date, warmup_days: int = 120) -> pd.DataFrame:
    """Daily close/adjusted close of ``ticker`` plus VIX, VXN and T-bill yield.

    Starts ``warmup_days`` before ``start`` so rolling volatility is available
    from the first backtest day.
    """
    fetch_start = start - dt.timedelta(days=warmup_days)
    und = _download(ticker, fetch_start, end)
    if und.empty or "Close" not in und:
        raise ValueError(f"No price data returned for {ticker!r}. Check the symbol or try again later.")

    out = pd.DataFrame(index=und.index)
    out["close"] = und["Close"].astype(float)
    out["adj_close"] = und["Adj Close"].astype(float) if "Adj Close" in und else out["close"]
    for col, symbol in (("vix", VIX_SYMBOL), ("vxn", VXN_SYMBOL), ("irx", TBILL_SYMBOL)):
        try:
            series = _download(symbol, fetch_start, end)
        except Exception:  # an index failing should not kill the whole backtest
            series = pd.DataFrame()
        out[col] = _align(series["Close"].astype(float), out.index) if "Close" in series else np.nan
    return out.dropna(subset=["close"])


def prepare_inputs(
    raw: pd.DataFrame,
    start: dt.date,
    end: dt.date,
    iv_source: str,
    hv_window: int = 30,
    constant_iv: float = 0.20,
    use_tbill_rate: bool = True,
    constant_rate: float = 0.04,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the daily model inputs: close, adj_close, vix, iv (decimal), rate (decimal)."""
    notes: list[str] = []
    df = raw.copy()

    vix = df["vix"] / 100.0
    if iv_source.startswith("VXN"):
        iv = df["vxn"] / 100.0
        if iv.isna().all():
            notes.append("VXN data unavailable; using VIX as the implied-volatility proxy.")
            iv = vix
        elif iv.isna().any():
            iv = iv.fillna(vix)
    elif iv_source == "VIX":
        iv = vix
    elif iv_source == "Historical volatility":
        log_ret = np.log(df["close"]).diff()
        iv = log_ret.rolling(hv_window, min_periods=max(5, hv_window // 2)).std() * np.sqrt(252)
        iv = iv.bfill()
    else:
        iv = pd.Series(constant_iv, index=df.index)

    if iv.isna().all():
        notes.append(f"No volatility data for '{iv_source}'; using constant {constant_iv:.0%}.")
        iv = pd.Series(constant_iv, index=df.index)
    df["iv"] = iv.ffill().bfill()

    if use_tbill_rate and df["irx"].notna().any():
        df["rate"] = (df["irx"] / 100.0).ffill().bfill()
    else:
        if use_tbill_rate:
            notes.append(f"T-bill yield unavailable; using constant rate {constant_rate:.2%}.")
        df["rate"] = constant_rate

    if df["vix"].isna().all():
        notes.append("VIX data unavailable; VIX entry/exit filters will block all trades if enabled.")

    df = df.loc[pd.Timestamp(start) : pd.Timestamp(end), ["close", "adj_close", "vix", "iv", "rate"]]
    if len(df) < 2:
        raise ValueError("Not enough trading days in the selected period.")
    return df, notes
