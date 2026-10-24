"""Day-by-day backtest engine.

Each trading day, at the close:
  1. every open trade is marked to model (Black-Scholes) and checked for exits
     (expiration, end of data, VIX, stop loss, take profit, exit DTE, max days);
  2. if the entry rules pass, one new trade is opened.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .fees import option_order_fee, stock_order_fee
from .models import MARGIN_MODELS, BacktestConfig
from .pricing import bs_price_delta, intrinsic, select_strike, skewed_vol

OPTION_MULTIPLIER = 100
NO_EXPIRY = np.iinfo(np.int64).max // 4


def _day_numbers(index: pd.DatetimeIndex) -> np.ndarray:
    return index.values.astype("datetime64[D]").astype(np.int64)


def _to_date(day: int) -> pd.Timestamp:
    return pd.Timestamp(np.datetime64(int(day), "D"))


def build_expirations(trading_days: pd.DatetimeIndex, cycle: str, future_days: int = 900) -> np.ndarray:
    """Listed expiration dates (as day numbers) for the chosen cycle.

    Trading days after the data end are approximated with business days. An
    expiration that lands on a market holiday rolls back to the prior trading day.
    """
    future = pd.bdate_range(trading_days[-1] + pd.Timedelta(days=1), periods=future_days)
    days = trading_days.union(future)
    if cycle == "Daily":
        return _day_numbers(days)

    cal = pd.date_range(days[0], days[-1], freq="D")
    if cycle == "Mon/Wed/Fri":
        targets = cal[cal.weekday.isin([0, 2, 4])]
    elif cycle == "Weekly (Fri)":
        targets = cal[cal.weekday == 4]
    elif cycle == "Monthly (3rd Fri)":
        targets = cal[(cal.weekday == 4) & ((cal.day - 1) // 7 == 2)]
    else:
        raise ValueError(f"Unknown expiration cycle: {cycle}")
    pos = days.searchsorted(targets, side="right") - 1
    pos = np.unique(pos[pos >= 0])
    return _day_numbers(days[pos])


def choose_expiry(expirations: np.ndarray, today: int, target_dte: int, exact: bool) -> int | None:
    """Expiration closest to ``target_dte`` days out (or exactly that many, if ``exact``)."""
    candidates = expirations[np.searchsorted(expirations, today + 1) :]
    if len(candidates) == 0:
        return None
    target = today + target_dte
    j = int(np.searchsorted(candidates, target))
    if exact:
        return int(candidates[j]) if j < len(candidates) and candidates[j] == target else None
    options = [candidates[k] for k in (j - 1, j) if 0 <= k < len(candidates)]
    return int(min(options, key=lambda e: (abs(e - target), e)))


@dataclass
class Trade:
    trade_id: int
    entry_day: int
    entry_spot: float
    entry_vix: float
    entry_iv: float
    strikes: np.ndarray
    is_call: np.ndarray
    is_stock: np.ndarray
    qty: np.ndarray
    weight: np.ndarray  # signed shares controlled per leg: +buy / -sell, x100 for options
    expiry: np.ndarray
    entry_px: np.ndarray
    entry_value: float  # cash paid at entry: + debit / - credit
    premium: float  # option net premium: + credit / - debit
    premium_basis: float
    bpr: float
    entry_fees: float
    first_expiry: int | None
    exit_day: int | None = None
    exit_spot: float = float("nan")
    exit_px: np.ndarray | None = None
    exit_value: float = float("nan")
    exit_fees: float = 0.0
    exit_reason: str = ""
    pnl: float = float("nan")
    max_profit_seen: float = field(default=0.0)
    max_loss_seen: float = field(default=0.0)

    def describe(self) -> str:
        parts = []
        for k in range(len(self.weight)):
            sign = "+" if self.weight[k] > 0 else "-"
            if self.is_stock[k]:
                parts.append(f"{sign}{self.qty[k]} sh")
            else:
                cp = "C" if self.is_call[k] else "P"
                exp = _to_date(self.expiry[k]).strftime("%Y-%m-%d")
                parts.append(f"{sign}{self.qty[k]} {cp} {self.strikes[k]:g} {exp}")
        return " | ".join(parts)


def leg_prices(strikes, is_call, is_stock, expiry, day, spot, rate, div_yield, iv, skew) -> np.ndarray:
    """Model mid price per share of each leg (stock legs = spot)."""
    px = np.full(len(strikes), spot, dtype=float)
    opt = ~is_stock
    if opt.any():
        t = np.maximum(expiry[opt] - day, 0) / 365.0
        sig = skewed_vol(iv, strikes[opt], spot, skew)
        px[opt], _ = bs_price_delta(spot, strikes[opt], t, rate, div_yield, sig, is_call[opt])
    return px


def _fill_prices(mid: np.ndarray, weight: np.ndarray, is_stock: np.ndarray, opening: bool, slippage: float) -> np.ndarray:
    """Mid prices moved against us by ``slippage`` per share on option legs."""
    if slippage <= 0:
        return mid.copy()
    side = np.sign(weight) if opening else -np.sign(weight)  # +1 = we buy, -1 = we sell
    fill = np.where(is_stock, mid, mid + side * slippage)
    return np.maximum(fill, 0.0)


def _order_fees(qty, is_stock, px, mask, cfg: BacktestConfig) -> float:
    opt = mask & ~is_stock
    fee = option_order_fee(
        [(int(q), float(p)) for q, p in zip(qty[opt], px[opt])], cfg.fee_model, cfg.custom_fee_per_contract
    )
    for k in np.flatnonzero(mask & is_stock):
        fee += stock_order_fee(int(qty[k]), float(px[k]), cfg.fee_model)
    return fee


def position_value(strikes, is_call, is_stock, weight, expiry, eval_day, grid, rate, div_yield, iv, skew) -> np.ndarray:
    """Position value on ``eval_day`` for each underlying price in ``grid``.

    Legs expired by then are worth intrinsic; the rest are valued with the model.
    """
    values = np.zeros(len(grid), dtype=float)
    for k in range(len(weight)):
        if is_stock[k]:
            values += weight[k] * grid
        elif expiry[k] <= eval_day:
            values += weight[k] * intrinsic(grid, strikes[k], is_call[k])
        else:
            sig = skewed_vol(iv, strikes[k], grid, skew)
            p, _ = bs_price_delta(grid, strikes[k], (expiry[k] - eval_day) / 365.0, rate, div_yield, sig, is_call[k])
            values += weight[k] * p
    return values


def compute_bpr(strikes, is_call, is_stock, weight, expiry, spot, rate, div_yield, iv, skew, entry_value, margin_model):
    """Approximate buying-power reduction of a position at entry.

    The position is valued at the first expiration over a grid of underlying
    prices (later-dated legs at model value). "Max loss" uses the worst loss
    on the grid; "Reg-T" caps it with a +/-20% move stress (min 10% of spot per
    uncovered short), as brokers do for naked options. Debits are always fully
    paid, and unbounded upside risk always falls back to the Reg-T stress.
    """
    opt = ~is_stock
    pts = [np.linspace(0.0, 3.0 * spot, 301), np.array([0.8 * spot, spot, 1.2 * spot])]
    t1 = int(expiry[opt].min()) if opt.any() else 0
    if opt.any():
        pts.append(strikes[opt])
    grid = np.unique(np.concatenate(pts))
    pnl = position_value(strikes, is_call, is_stock, weight, expiry, t1, grid, rate, div_yield, iv, skew) - entry_value

    full_loss = max(0.0, -pnl.min())
    upside_slope = weight[is_call | is_stock].sum()
    unbounded = upside_slope < -1e-9
    in_stress = (grid >= 0.8 * spot - 1e-9) & (grid <= 1.2 * spot + 1e-9)
    stress_loss = max(0.0, -pnl[in_stress].min())

    shares_long = weight[is_stock].sum()
    net_calls = (weight[opt & is_call].sum() + shares_long) / OPTION_MULTIPLIER
    net_puts = weight[opt & ~is_call].sum() / OPTION_MULTIPLIER
    naked = max(0.0, -net_calls, -net_puts)
    reg_t = max(stress_loss, 0.10 * spot * OPTION_MULTIPLIER * naked)

    if unbounded:
        core = reg_t
    elif margin_model == MARGIN_MODELS[0]:
        core = min(full_loss, reg_t)
    else:
        core = full_loss
    return float(max(entry_value, 0.0, core))


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    daily: pd.DataFrame
    capital: float
    config: BacktestConfig
    skipped_entries: int


class Backtester:
    def __init__(self, data: pd.DataFrame, cfg: BacktestConfig):
        errors = cfg.validate()
        if errors:
            raise ValueError("\n".join(errors))
        self.cfg = cfg
        self.data = data
        self.dates = data.index
        self.days = _day_numbers(data.index)
        self.close = data["close"].to_numpy(float)
        self.vix = data["vix"].to_numpy(float)
        self.iv = data["iv"].to_numpy(float) * cfg.iv_multiplier
        self.rate = data["rate"].to_numpy(float)
        self.expirations = build_expirations(data.index, cfg.expiration_cycle)
        self.skipped_entries = 0

    # ------------------------------------------------------------------ entry
    def _entry_allowed(self, i: int, n_open: int) -> bool:
        en, ex = self.cfg.entry, self.cfg.exit
        vix = self.vix[i]
        if en.weekdays is not None and self.dates[i].weekday() not in en.weekdays:
            return False
        if en.max_open_trades is not None and n_open >= en.max_open_trades:
            return False
        if en.vix_min is not None and not vix >= en.vix_min:
            return False
        if en.vix_max is not None and not vix <= en.vix_max:
            return False
        if ex.vix_exit_above is not None and not vix <= ex.vix_exit_above:
            return False
        return np.isfinite(self.iv[i]) and self.iv[i] > 0

    def _open_trade(self, i: int, trade_id: int) -> Trade | None:
        cfg = self.cfg
        day, spot, iv, rate = int(self.days[i]), self.close[i], self.iv[i], self.rate[i]
        n = len(cfg.legs)
        strikes = np.full(n, np.nan)
        is_call = np.zeros(n, dtype=bool)
        is_stock = np.zeros(n, dtype=bool)
        qty = np.zeros(n, dtype=int)
        weight = np.zeros(n)
        expiry = np.full(n, NO_EXPIRY, dtype=np.int64)

        for k, leg in enumerate(cfg.legs):
            qty[k] = leg.quantity
            if leg.is_stock:
                is_stock[k] = True
                weight[k] = leg.sign * leg.quantity
                continue
            exp = choose_expiry(self.expirations, day, leg.dte, cfg.entry.exact_dte)
            if exp is None:
                return None
            is_call[k] = leg.option_type == "Call"
            weight[k] = leg.sign * leg.quantity * OPTION_MULTIPLIER
            expiry[k] = exp
            ref = strikes[leg.ref_leg - 1] if leg.strike_method == "$ from leg" else None
            strikes[k] = select_strike(
                leg.strike_method,
                leg.strike_value,
                spot=spot,
                t_years=(exp - day) / 365.0,
                rate=rate,
                div_yield=cfg.div_yield,
                base_iv=iv,
                skew=cfg.skew,
                is_call=bool(is_call[k]),
                increment=cfg.strike_increment,
                ref_strike=ref,
            )

        mid = leg_prices(strikes, is_call, is_stock, expiry, day, spot, rate, cfg.div_yield, iv, cfg.skew)
        fill = _fill_prices(mid, weight, is_stock, opening=True, slippage=cfg.slippage_per_share)
        entry_value = float(weight @ fill)
        opt = ~is_stock
        premium = -float(weight[opt] @ fill[opt])
        basis = abs(premium) if abs(premium) > 1e-9 else abs(entry_value)
        fees = _order_fees(qty, is_stock, fill, np.ones(n, dtype=bool), cfg)
        bpr = compute_bpr(
            strikes, is_call, is_stock, weight, expiry, spot, rate, cfg.div_yield, iv, cfg.skew, entry_value,
            cfg.margin_model,
        )
        return Trade(
            trade_id=trade_id,
            entry_day=day,
            entry_spot=spot,
            entry_vix=self.vix[i],
            entry_iv=iv,
            strikes=strikes,
            is_call=is_call,
            is_stock=is_stock,
            qty=qty,
            weight=weight,
            expiry=expiry,
            entry_px=fill,
            entry_value=entry_value,
            premium=premium,
            premium_basis=basis,
            bpr=bpr,
            entry_fees=fees,
            first_expiry=int(expiry[opt].min()) if opt.any() else None,
        )

    # ------------------------------------------------------------------- exit
    def _exit_reason(self, tr: Trade, i: int, value: float, last_day: bool) -> str:
        ex = self.cfg.exit
        day = int(self.days[i])
        if tr.first_expiry is not None and day >= tr.first_expiry:
            return "Expiration"
        if last_day:
            return "End of data"
        if ex.vix_exit_above is not None and self.vix[i] > ex.vix_exit_above:
            return "VIX exit"
        open_pnl = value - tr.entry_value
        if ex.stop_loss_pct is not None and tr.premium_basis > 0 and open_pnl <= -ex.stop_loss_pct / 100 * tr.premium_basis:
            return "Stop loss"
        if ex.take_profit_pct is not None and tr.premium_basis > 0 and open_pnl >= ex.take_profit_pct / 100 * tr.premium_basis:
            return "Take profit"
        if ex.exit_dte is not None and tr.first_expiry is not None and tr.first_expiry - day <= ex.exit_dte:
            return "Exit DTE"
        if ex.max_days_in_trade is not None and day - tr.entry_day >= ex.max_days_in_trade:
            return "Max days in trade"
        return ""

    def _close_trade(self, tr: Trade, i: int, mid: np.ndarray, reason: str) -> None:
        cfg = self.cfg
        day = int(self.days[i])
        n = len(tr.weight)
        # Legs that expire today settle at intrinsic with no fee; the rest are traded out.
        traded = np.ones(n, dtype=bool) if reason != "Expiration" else (tr.is_stock | (tr.expiry > day))
        fill = mid.copy()
        fill[traded] = _fill_prices(mid[traded], tr.weight[traded], tr.is_stock[traded], False, cfg.slippage_per_share)
        tr.exit_day = day
        tr.exit_spot = self.close[i]
        tr.exit_px = fill
        tr.exit_value = float(tr.weight @ fill)
        tr.exit_fees = _order_fees(tr.qty, tr.is_stock, fill, traded, cfg)
        tr.exit_reason = reason
        tr.pnl = tr.exit_value - tr.entry_value - tr.entry_fees - tr.exit_fees

    # -------------------------------------------------------------------- run
    def run(self) -> BacktestResult:
        cfg = self.cfg
        open_trades: list[Trade] = []
        closed: list[Trade] = []
        realized = 0.0
        next_id = 1
        rows = []
        n_days = len(self.days)

        for i in range(n_days):
            day, spot, iv, rate = int(self.days[i]), self.close[i], self.iv[i], self.rate[i]
            last_day = i == n_days - 1
            unrealized = 0.0
            exits_today = 0

            if open_trades:
                strikes = np.concatenate([t.strikes for t in open_trades])
                is_call = np.concatenate([t.is_call for t in open_trades])
                is_stock = np.concatenate([t.is_stock for t in open_trades])
                expiry = np.concatenate([t.expiry for t in open_trades])
                mids = leg_prices(strikes, is_call, is_stock, expiry, day, spot, rate, cfg.div_yield, iv, cfg.skew)
                splits = np.cumsum([len(t.weight) for t in open_trades])[:-1]
                still_open = []
                for tr, mid in zip(open_trades, np.split(mids, splits)):
                    value = float(tr.weight @ mid)
                    open_pnl = value - tr.entry_value
                    tr.max_profit_seen = max(tr.max_profit_seen, open_pnl)
                    tr.max_loss_seen = min(tr.max_loss_seen, open_pnl)
                    reason = self._exit_reason(tr, i, value, last_day)
                    if reason:
                        self._close_trade(tr, i, mid, reason)
                        realized += tr.pnl
                        closed.append(tr)
                        exits_today += 1
                    else:
                        unrealized += open_pnl - tr.entry_fees
                        still_open.append(tr)
                open_trades = still_open

            entered = False
            if not last_day and self._entry_allowed(i, len(open_trades)):
                tr = self._open_trade(i, next_id)
                if tr is None:
                    self.skipped_entries += 1
                else:
                    next_id += 1
                    open_trades.append(tr)
                    entered = True
                    mid = leg_prices(
                        tr.strikes, tr.is_call, tr.is_stock, tr.expiry, day, spot, rate, cfg.div_yield, iv, cfg.skew
                    )
                    unrealized += float(tr.weight @ mid) - tr.entry_value - tr.entry_fees

            rows.append(
                {
                    "date": self.dates[i],
                    "close": spot,
                    "adj_close": self.data["adj_close"].iat[i],
                    "vix": self.vix[i],
                    "iv": self.iv[i],
                    "realized": realized,
                    "unrealized": unrealized,
                    "pnl": realized + unrealized,
                    "open_trades": len(open_trades),
                    "bpr_used": sum(t.bpr for t in open_trades),
                    "entered": entered,
                    "exits": exits_today,
                }
            )

        daily = pd.DataFrame(rows).set_index("date")
        trades = trades_frame(closed)
        capital = cfg.account_size or float(daily["bpr_used"].max())
        return BacktestResult(trades, daily, capital, cfg, self.skipped_entries)


def trades_frame(trades: list[Trade]) -> pd.DataFrame:
    columns = [
        "trade_id", "entry_date", "exit_date", "days", "legs", "entry_spot", "exit_spot", "entry_vix", "entry_iv",
        "premium", "bpr", "fees", "pnl", "return_pct", "max_open_profit", "max_open_loss", "exit_reason", "expiration",
    ]
    rows = []
    for t in sorted(trades, key=lambda t: t.trade_id):
        rows.append(
            {
                "trade_id": t.trade_id,
                "entry_date": _to_date(t.entry_day),
                "exit_date": _to_date(t.exit_day),
                "days": t.exit_day - t.entry_day,
                "legs": t.describe(),
                "entry_spot": t.entry_spot,
                "exit_spot": t.exit_spot,
                "entry_vix": t.entry_vix,
                "entry_iv": t.entry_iv,
                "premium": t.premium,
                "bpr": t.bpr,
                "fees": t.entry_fees + t.exit_fees,
                "pnl": t.pnl,
                "return_pct": t.pnl / t.bpr if t.bpr > 0 else np.nan,
                "max_open_profit": t.max_profit_seen,
                "max_open_loss": t.max_loss_seen,
                "exit_reason": t.exit_reason,
                "expiration": _to_date(t.first_expiry) if t.first_expiry is not None else pd.NaT,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def run_backtest(data: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    return Backtester(data, cfg).run()


def preview_trade(data: pd.DataFrame, cfg: BacktestConfig) -> Trade | None:
    """The trade the strategy would open on the last day of ``data`` (ignores entry filters)."""
    preview_cfg = replace(cfg, entry=replace(cfg.entry, exact_dte=False))
    return Backtester(data, preview_cfg)._open_trade(len(data) - 1, trade_id=0)


def payoff_curve(tr: Trade, rate: float, div_yield: float, iv: float, skew: float):
    """P&L (after entry fees) across underlying prices: at first expiration and on the entry day."""
    opt = ~tr.is_stock
    lo, hi = 0.75 * tr.entry_spot, 1.25 * tr.entry_spot
    if opt.any():
        lo, hi = min(lo, tr.strikes[opt].min() * 0.9), max(hi, tr.strikes[opt].max() * 1.1)
    grid = np.linspace(lo, hi, 301)
    cost = tr.entry_value + tr.entry_fees
    args = (tr.strikes, tr.is_call, tr.is_stock, tr.weight, tr.expiry)
    first = tr.first_expiry if tr.first_expiry is not None else tr.entry_day
    at_expiry = position_value(*args, first, grid, rate, div_yield, iv, skew) - cost
    today = position_value(*args, tr.entry_day, grid, rate, div_yield, iv, skew) - cost
    return grid, at_expiry, today
