"""Black-Scholes-Merton pricing and strike selection.

Free data sources have no historical option chains, so every option price in
the backtest is modelled: an implied-volatility proxy (VXN, VIX, realized vol
or a constant) feeds a Black-Scholes model with a simple linear skew.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

MIN_SIGMA = 1e-4


def skewed_vol(base_iv, strike, spot, skew: float):
    """Implied vol for a strike given an ATM vol and a linear log-moneyness skew.

    A positive ``skew`` raises the vol of lower strikes (equity-style put skew):
    sigma(K) = base * (1 - skew * ln(K / S)), clipped to [0.5x, 3x] of base.
    """
    base_iv = np.asarray(base_iv, dtype=float)
    if skew == 0.0:
        return np.broadcast_to(base_iv, np.broadcast(base_iv, strike, spot).shape).astype(float)
    m = np.log(np.asarray(strike, dtype=float) / np.maximum(np.asarray(spot, dtype=float), 1e-12))
    return base_iv * np.clip(1.0 - skew * m, 0.5, 3.0)


def intrinsic(spot, strike, is_call):
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    return np.where(is_call, np.maximum(spot - strike, 0.0), np.maximum(strike - spot, 0.0))


def bs_price_delta(spot, strike, t_years, rate, div_yield, sigma, is_call):
    """Vectorized Black-Scholes-Merton price and delta (per share).

    Options with ``t_years <= 0`` are worth intrinsic value.
    """
    spot = np.asarray(spot, dtype=float)
    strike = np.maximum(np.asarray(strike, dtype=float), 1e-12)
    t = np.asarray(t_years, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    sig = np.maximum(np.asarray(sigma, dtype=float), MIN_SIGMA)

    t_safe = np.maximum(t, 1e-8)
    sqrt_t = np.sqrt(t_safe)
    safe_spot = np.maximum(spot, 1e-12)
    d1 = (np.log(safe_spot / strike) + (rate - div_yield + 0.5 * sig**2) * t_safe) / (sig * sqrt_t)
    d2 = d1 - sig * sqrt_t
    dfq = np.exp(-div_yield * t_safe)
    dfr = np.exp(-rate * t_safe)

    nd1 = ndtr(d1)
    call = spot * dfq * nd1 - strike * dfr * ndtr(d2)
    put = strike * dfr * ndtr(-d2) - spot * dfq * ndtr(-d1)
    price = np.where(is_call, call, put)
    delta = np.where(is_call, dfq * nd1, dfq * (nd1 - 1.0))

    expired = t <= 0
    if np.any(expired):
        price = np.where(expired, intrinsic(spot, strike, is_call), price)
        exp_delta = np.where(is_call, (spot > strike).astype(float), -(spot < strike).astype(float))
        delta = np.where(expired, exp_delta, delta)
    return np.maximum(price, 0.0), delta


def round_strike(value: float, increment: float) -> float:
    k = round(value / increment) * increment
    return float(max(k, increment))


def strike_grid(spot: float, increment: float, low: float = 0.2, high: float = 3.0) -> np.ndarray:
    lo = max(increment, np.floor(spot * low / increment) * increment)
    hi = np.ceil(spot * high / increment) * increment
    return np.arange(lo, hi + increment / 2, increment)


def select_strike(
    method: str,
    value: float,
    *,
    spot: float,
    t_years: float,
    rate: float,
    div_yield: float,
    base_iv: float,
    skew: float,
    is_call: bool,
    increment: float,
    ref_strike: float | None = None,
) -> float:
    """Pick a listed strike (multiple of ``increment``) for one option leg.

    Methods (see ``models.STRIKE_METHODS``):
      * Delta           – |delta| closest to value/100 (35 -> 0.35, sign ignored)
      * $ from price    – spot + value
      * % from price    – spot * (1 + value/100)
      * ATM             – nearest strike to spot
      * Premium         – option price closest to value ($ per share)
      * $ from leg      – strike of the referenced leg + value
    """
    if method == "ATM":
        return round_strike(spot, increment)
    if method == "$ from price":
        return round_strike(spot + value, increment)
    if method == "% from price":
        return round_strike(spot * (1.0 + value / 100.0), increment)
    if method == "$ from leg":
        if ref_strike is None:
            raise ValueError("'$ from leg' needs a referenced option leg")
        return round_strike(ref_strike + value, increment)

    grid = strike_grid(spot, increment)
    sig = skewed_vol(base_iv, grid, spot, skew)
    price, delta = bs_price_delta(spot, grid, t_years, rate, div_yield, sig, is_call)
    if method == "Delta":
        target = abs(value) / 100.0
        idx = int(np.argmin(np.abs(np.abs(delta) - target)))
    elif method == "Premium":
        idx = int(np.argmin(np.abs(price - abs(value))))
    else:
        raise ValueError(f"Unknown strike selection method: {method}")
    return float(grid[idx])
