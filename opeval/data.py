"""Market data from Tiingo (cached on disk, at most one API call per ticker per day) and model inputs built from it."""

from __future__ import annotations

import datetime as dt
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

VIX_SYMBOL = "^VIX"
VXN_SYMBOL = "^VXN"
TBILL_SYMBOL = "^IRX"  # 13-week T-bill yield, in percent

TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{symbol}/prices"
TIINGO_START = "1970-01-01"  # full history is fetched once a day and sliced from the cache
TIINGO_TIMEOUT = 30  # seconds

PERIODS = {
    "1 Year": 1,
    "2 Years": 2,
    "3 Years": 3,
    "5 Years": 5,
    "10 Years": 10,
    "15 Years": 15,
    "20 Years": 20,
}

# Per-process memo so repeated calls on the same day never reach the disk cache or Tiingo again.
_MEMO: dict[str, dict] = {}


def period_start(years: int, end: dt.date) -> dt.date:
    try:
        return end.replace(year=end.year - years)
    except ValueError:  # Feb 29
        return end.replace(year=end.year - years, day=28)


# --------------------------------------------------------------------------- cache
def _today() -> dt.date:
    return dt.date.today()


def _cache_dir() -> Path:
    override = os.environ.get("OPEVAL_CACHE_DIR")
    return Path(override) if override else Path.home() / ".cache" / "opeval" / "tiingo"


def _cache_path(symbol: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in symbol)
    return _cache_dir() / f"{safe}.json"


def _read_cache(symbol: str) -> dict | None:
    try:
        return json.loads(_cache_path(symbol).read_text())
    except (OSError, ValueError):
        return None


def _write_cache(symbol: str, payload: dict) -> None:
    """Best effort: a failed write only means the in-process memo is the only cache for today."""
    path = _cache_path(symbol)
    tmp = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, path)
    except OSError:
        pass


# --------------------------------------------------------------------------- Tiingo
def _tiingo_request(symbol: str, token: str) -> list[dict] | None:
    """Full daily price history from Tiingo. Returns None when Tiingo does not know the symbol."""
    query = urllib.parse.urlencode({"startDate": TIINGO_START, "resampleFreq": "daily", "token": token})
    url = TIINGO_URL.format(symbol=urllib.parse.quote(symbol, safe="")) + "?" + query
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIINGO_TIMEOUT) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:  # the message must not echo the URL, which carries the token
        if exc.code == 404:
            return None
        if exc.code in (401, 403):
            raise ValueError("Tiingo rejected the API key. Check TIINGO_API_KEY.") from None
        if exc.code == 429:
            raise ValueError("Tiingo rate limit reached. Try again later.") from None
        raise ValueError(f"Tiingo request for {symbol!r} failed (HTTP {exc.code}).") from None
    except OSError as exc:  # URLError, timeouts
        raise ValueError(f"Could not reach Tiingo: {getattr(exc, 'reason', exc)}") from None


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    """Tiingo rows to a frame with 'Close' and 'Adj Close' columns on a naive daily DatetimeIndex."""
    if not rows:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"), columns=["Close", "Adj Close"], dtype=float)
    idx = pd.DatetimeIndex(pd.to_datetime([r["date"][:10] for r in rows]), name="date")
    df = pd.DataFrame(
        {"Close": [r["close"] for r in rows], "Adj Close": [r["adjClose"] for r in rows]},
        index=idx,
        dtype=float,
    )
    return df[~df.index.duplicated(keep="last")].sort_index()


def _load_history(symbol: str) -> pd.DataFrame:
    """Full daily history for ``symbol``. Tiingo is called at most once per symbol per calendar day.

    A symbol Tiingo does not know is cached as empty for the day, so it is not retried either.
    Failed requests are not cached, so they can be retried.
    """
    today = _today().isoformat()
    cached = _MEMO.get(symbol) or _read_cache(symbol)
    if cached is None or cached.get("fetched_on") != today:
        token = os.environ.get("TIINGO_API_KEY", "").strip()
        if not token:
            raise ValueError("TIINGO_API_KEY is not set. Add your Tiingo API key to the environment or Streamlit secrets.")
        rows = _tiingo_request(symbol, token) or []
        cached = {"fetched_on": today, "rows": rows}
        _write_cache(symbol, cached)
    _MEMO[symbol] = cached
    return _to_frame(cached["rows"])


def _slice(df: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    return df.loc[pd.Timestamp(start) : pd.Timestamp(end)]


def _align(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    return series.reindex(index.union(series.index)).ffill().reindex(index)


def fetch_market_data(ticker: str, start: dt.date, end: dt.date, warmup_days: int = 120) -> pd.DataFrame:
    """Daily close/adjusted close of ``ticker`` plus VIX, VXN and T-bill yield.

    Starts ``warmup_days`` before ``start`` so rolling volatility is available
    from the first backtest day. Tiingo does not serve the index series, so
    VIX, VXN and the T-bill are usually missing and the inputs fall back to
    their notes in ``prepare_inputs``.
    """
    fetch_start = start - dt.timedelta(days=warmup_days)
    und = _slice(_load_history(ticker), fetch_start, end)
    if und.empty:
        raise ValueError(f"No price data returned for {ticker!r}. Check the symbol or try again later.")

    out = pd.DataFrame(index=und.index)
    out["close"] = und["Close"].astype(float)
    out["adj_close"] = und["Adj Close"].astype(float)
    for col, symbol in (("vix", VIX_SYMBOL), ("vxn", VXN_SYMBOL), ("irx", TBILL_SYMBOL)):
        try:
            series = _slice(_load_history(symbol), fetch_start, end)
        except ValueError:  # an index failing should not kill the whole backtest
            series = pd.DataFrame()
        out[col] = _align(series["Close"], out.index) if not series.empty else np.nan
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
