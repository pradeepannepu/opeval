"""Performance statistics for the strategy and for buy & hold."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _nan_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def equity_stats(pnl: pd.Series, capital: float) -> dict:
    """Stats for a cumulative P&L curve on a given capital base.

    Equity = capital + P&L. Drawdowns are measured from equity peaks.
    MAR ratio = CAGR / max drawdown %.
    """
    if pnl.empty or capital <= 0:
        return {
            "total_pnl": float(pnl.iloc[-1]) if not pnl.empty else 0.0,
            "capital": capital,
            "return_on_capital": float("nan"),
            "cagr": float("nan"),
            "max_dd_pct": float("nan"),
            "max_dd_dollar": float("nan"),
            "mar": float("nan"),
            "sharpe": float("nan"),
        }
    equity = capital + pnl
    peak = equity.cummax()
    dd_pct = equity / peak - 1.0
    max_dd_pct = float(-dd_pct.min())
    max_dd_dollar = float((peak - equity).max())
    total = float(pnl.iloc[-1])
    years = max((pnl.index[-1] - pnl.index[0]).days / 365.25, 1 / 365.25)
    final = capital + total
    cagr = (final / capital) ** (1.0 / years) - 1.0 if final > 0 else -1.0
    rets = equity.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else float("nan")
    return {
        "total_pnl": total,
        "capital": capital,
        "return_on_capital": total / capital,
        "cagr": cagr,
        "max_dd_pct": max_dd_pct,
        "max_dd_dollar": max_dd_dollar,
        "mar": _nan_div(cagr, max_dd_pct),
        "sharpe": sharpe,
    }


def drawdown_pct(pnl: pd.Series, capital: float) -> pd.Series:
    equity = capital + pnl
    return equity / equity.cummax() - 1.0


def buy_and_hold_pnl(adj_close: pd.Series, capital: float) -> pd.Series:
    """P&L of investing ``capital`` in the underlying (dividends reinvested) on day one."""
    return capital * (adj_close / adj_close.iloc[0] - 1.0)


def trade_stats(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0}
    pnl = trades["pnl"]
    wins = trades[pnl > 0]
    losses = trades[pnl < 0]
    gross_win = float(wins["pnl"].sum())
    gross_loss = float(-losses["pnl"].sum())
    abs_premium = float(trades["premium"].abs().sum())
    return {
        "n_trades": n,
        "n_wins": len(wins),
        "win_rate": len(wins) / n,
        "largest_win": float(wins["pnl"].max()) if len(wins) else 0.0,
        "n_losses": len(losses),
        "loss_rate": len(losses) / n,
        "largest_loss": float(losses["pnl"].min()) if len(losses) else 0.0,
        "avg_return_per_trade": float(trades["return_pct"].mean()),
        "avg_days": float(trades["days"].mean()),
        "avg_bpr": float(trades["bpr"].mean()),
        "avg_premium": float(trades["premium"].mean()),
        "avg_pnl": float(pnl.mean()),
        "avg_win": float(wins["pnl"].mean()) if len(wins) else 0.0,
        "avg_win_pct": float(wins["return_pct"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["pnl"].mean()) if len(losses) else 0.0,
        "avg_loss_pct": float(losses["return_pct"].mean()) if len(losses) else 0.0,
        "premium_capture": _nan_div(float(pnl.sum()), abs_premium),
        "profit_factor": _nan_div(gross_win, gross_loss),
        "total_premium": float(trades["premium"].sum()),
        "total_fees": float(trades["fees"].sum()),
    }


def monthly_pnl(pnl: pd.Series) -> pd.DataFrame:
    """Year x month table of P&L changes."""
    month_end = pnl.resample("ME").last()
    changes = month_end.diff()
    changes.iloc[0] = month_end.iloc[0]  # P&L starts at zero
    table = changes.to_frame("pnl")
    table["year"] = table.index.year
    table["month"] = table.index.month
    return table.pivot(index="year", columns="month", values="pnl").reindex(columns=range(1, 13))
