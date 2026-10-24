import numpy as np
import pandas as pd
import pytest

from opeval.engine import build_expirations, choose_expiry, run_backtest
from opeval.fees import option_order_fee, stock_order_fee
from opeval.metrics import equity_stats, trade_stats
from opeval.models import BacktestConfig, EntryConfig, ExitConfig, Leg
from opeval.pricing import bs_price_delta, select_strike


def synthetic_data(days=400, start_price=100.0, drift=0.0003, iv=0.2, seed=0):
    idx = pd.bdate_range("2022-01-03", periods=days)
    rng = np.random.default_rng(seed)
    rets = drift + rng.normal(0, 0.01, days) if seed is not None else np.full(days, drift)
    close = start_price * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {"close": close, "adj_close": close, "vix": 18.0, "iv": iv, "rate": 0.03}, index=idx
    )


def test_put_call_parity():
    s, k, t, r, q, sig = 100.0, 95.0, 0.25, 0.03, 0.01, 0.25
    call, _ = bs_price_delta(s, k, t, r, q, sig, True)
    put, _ = bs_price_delta(s, k, t, r, q, sig, False)
    assert call - put == pytest.approx(s * np.exp(-q * t) - k * np.exp(-r * t), abs=1e-9)


def test_expired_option_is_intrinsic():
    price, delta = bs_price_delta(np.array([90.0, 110.0]), 100.0, 0.0, 0.03, 0.0, 0.2, True)
    assert price.tolist() == [0.0, 10.0]
    assert delta.tolist() == [0.0, 1.0]


@pytest.mark.parametrize("is_call", [True, False])
def test_delta_strike_selection(is_call):
    k = select_strike("Delta", 30, spot=400, t_years=45 / 365, rate=0.04, div_yield=0.006, base_iv=0.22,
                      skew=0.0, is_call=is_call, increment=1.0)
    _, delta = bs_price_delta(400, k, 45 / 365, 0.04, 0.006, 0.22, is_call)
    assert abs(abs(float(delta)) - 0.30) < 0.01
    assert (k > 400) if is_call else (k < 400)


def test_relative_strikes():
    common = dict(spot=401.3, t_years=0.1, rate=0.04, div_yield=0.0, base_iv=0.2, skew=0.0, is_call=False,
                  increment=1.0)
    assert select_strike("$ from price", -5, **common) == 396.0
    assert select_strike("% from price", -10, **common) == 361.0
    assert select_strike("ATM", 0, **common) == 401.0
    assert select_strike("$ from leg", -5, ref_strike=390.0, **common) == 385.0


def test_monthly_expirations_are_third_fridays():
    idx = pd.bdate_range("2024-01-01", "2024-06-30")
    exps = pd.to_datetime(build_expirations(idx, "Monthly (3rd Fri)").astype("datetime64[D]"))
    first_half = exps[exps <= "2024-06-30"]
    assert [d.strftime("%Y-%m-%d") for d in first_half] == [
        "2024-01-19", "2024-02-16", "2024-03-15", "2024-04-19", "2024-05-17", "2024-06-21",
    ]


def test_choose_expiry_exact_and_nearest():
    exps = np.array([10, 17, 24, 31])
    assert choose_expiry(exps, 0, 17, exact=True) == 17
    assert choose_expiry(exps, 0, 20, exact=True) is None
    assert choose_expiry(exps, 0, 20, exact=False) == 17
    assert choose_expiry(exps, 0, 22, exact=False) == 24


def test_ibkr_fees():
    assert option_order_fee([(1, 2.0)], "IBKR Pro Fixed") == 1.00  # $0.65 lifts to the $1 minimum
    assert option_order_fee([(2, 2.0), (2, 1.0)], "IBKR Pro Fixed") == pytest.approx(2.60)
    assert option_order_fee([(10, 0.03)], "IBKR Pro Fixed") == pytest.approx(2.50)
    assert stock_order_fee(100, 400.0, "IBKR Pro Fixed") == 1.00
    assert option_order_fee([(5, 1.0)], "None") == 0.0


def test_short_put_in_rising_market_keeps_premium():
    data = synthetic_data(seed=None, drift=0.002)  # smooth uptrend, every put expires worthless
    cfg = BacktestConfig(
        legs=[Leg("Option", "Sell", "Put", 1, "Delta", 30, 1, 30)],
        entry=EntryConfig(max_open_trades=1),
        fee_model="None",
    )
    res = run_backtest(data, cfg)
    expired = res.trades[res.trades["exit_reason"] == "Expiration"]
    assert len(expired) > 5
    assert np.allclose(expired["pnl"], expired["premium"])
    assert (res.trades["bpr"] > 0).all()
    assert res.daily["open_trades"].max() == 1
    assert res.capital == pytest.approx(res.daily["bpr_used"].max())


def test_put_spread_loss_capped_by_width():
    data = synthetic_data(seed=None, drift=-0.003)  # smooth downtrend, spreads finish max loss
    cfg = BacktestConfig(
        legs=[Leg("Option", "Sell", "Put", 1, "Delta", 35, 1, 30), Leg("Option", "Buy", "Put", 1, "$ from leg", -5, 1, 30)],
        fee_model="None",
    )
    res = run_backtest(data, cfg)
    t = res.trades[res.trades["exit_reason"] == "Expiration"]
    assert (t["pnl"] >= -500 - 1e-6).all()
    assert np.allclose(t["bpr"], 500 - t["premium"])  # Reg-T for a spread = width - credit


def test_take_profit_and_exit_rules():
    data = synthetic_data(seed=1)
    cfg = BacktestConfig(
        legs=[Leg("Option", "Sell", "Put", 1, "Delta", 30, 1, 45)],
        entry=EntryConfig(weekdays=[4]),
        exit=ExitConfig(take_profit_pct=50, exit_dte=21),
    )
    res = run_backtest(data, cfg)
    assert set(res.trades["exit_reason"]) <= {"Take profit", "Exit DTE", "End of data", "Expiration"}
    assert (res.trades["entry_date"].dt.weekday == 4).all()
    tp = res.trades[res.trades["exit_reason"] == "Take profit"]
    assert (tp["pnl"] + tp["fees"] >= 0.5 * tp["premium"] - 1e-6).all()
    assert (res.trades["fees"] > 0).all()


def test_vix_filter_blocks_entries():
    data = synthetic_data()
    data["vix"] = 40.0
    cfg = BacktestConfig(legs=[Leg()], entry=EntryConfig(vix_max=30))
    assert run_backtest(data, cfg).trades.empty


def test_metrics():
    pnl = pd.Series([0.0, 100.0, 50.0, 200.0], index=pd.bdate_range("2024-01-01", periods=4))
    stats = equity_stats(pnl, 1000.0)
    assert stats["total_pnl"] == 200.0
    assert stats["max_dd_pct"] == pytest.approx(50 / 1100)
    trades = pd.DataFrame({"pnl": [100.0, -50.0], "premium": [150.0, 150.0], "return_pct": [0.1, -0.05],
                           "days": [10, 20], "bpr": [1000.0, 1000.0], "fees": [2.0, 2.0]})
    ts = trade_stats(trades)
    assert ts["win_rate"] == 0.5 and ts["largest_loss"] == -50.0
    assert ts["premium_capture"] == pytest.approx(50 / 300)


def test_validation_rejects_bad_leg_reference():
    cfg = BacktestConfig(legs=[Leg(strike_method="$ from leg", ref_leg=1)])
    assert cfg.validate()
