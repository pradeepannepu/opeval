"""Streamlit UI for the options backtester.  Run with:  uv run streamlit run app.py"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from dataclasses import asdict, fields

import pandas as pd
import streamlit as st

from opeval import charts
from opeval.data import PERIODS, fetch_market_data, period_start, prepare_inputs
from opeval.engine import payoff_curve, preview_trade, run_backtest
from opeval.metrics import buy_and_hold_pnl, drawdown_pct, equity_stats, monthly_pnl, trade_stats
from opeval.models import (
    EXPIRATION_CYCLES,
    FEE_MODELS,
    IV_SOURCES,
    MARGIN_MODELS,
    STRIKE_METHOD_HELP,
    STRIKE_METHODS,
    WEEKDAYS,
    BacktestConfig,
    EntryConfig,
    ExitConfig,
    Leg,
)
from opeval.presets import DEFAULT_PRESET, PRESETS

st.set_page_config(page_title="Options Backtester", page_icon="📈", layout="wide")

LEG_FIELDS = [f.name for f in fields(Leg)]


# --------------------------------------------------------------------------- helpers
@st.cache_data(ttl=6 * 3600, show_spinner="Downloading market data from Yahoo Finance…")
def load_raw(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    return fetch_market_data(ticker, start, end)


def money(x: float, decimals: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"-${abs(x):,.{decimals}f}" if x < 0 else f"${x:,.{decimals}f}"


def pct(x: float, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x * 100:,.{decimals}f}%"


def ratio(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:,.2f}"


def chart_theme() -> charts.Theme:
    try:
        return charts.theme_for(st.context.theme.type)
    except Exception:
        return charts.LIGHT


def show(fig) -> None:
    st.plotly_chart(fig, theme=None, config={"displaylogo": False})


def new_leg_state(leg: Leg) -> dict:
    state = asdict(leg)
    state["id"] = uuid.uuid4().hex[:8]
    return state


def load_preset(name: str) -> None:
    st.session_state.legs = [new_leg_state(leg) for leg in PRESETS[name]]


if "legs" not in st.session_state:
    load_preset(DEFAULT_PRESET)


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Market")
    ticker = st.text_input("Ticker", "QQQ").strip().upper() or "QQQ"
    period = st.selectbox("Interval", [*PERIODS, "Custom"], index=list(PERIODS).index("5 Years"))
    today = dt.date.today()
    if period == "Custom":
        c1, c2 = st.columns(2)
        start = c1.date_input("Start", period_start(3, today), max_value=today)
        end = c2.date_input("End", today, max_value=today)
    else:
        end = today
        start = period_start(PERIODS[period], today)

    with st.expander("Pricing model", expanded=False):
        iv_default = 0 if ticker == "QQQ" else 1
        iv_source = st.selectbox(
            "Implied volatility source", IV_SOURCES, index=iv_default,
            help="Historical option chains are not freely available, so options are priced with Black-Scholes "
                 "using this volatility. VXN is the Nasdaq-100 implied-vol index (best proxy for QQQ).",
        )
        hv_window, constant_iv = 30, 0.20
        if iv_source == "Historical volatility":
            hv_window = st.number_input("Realized vol window (days)", 5, 252, 30)
        if iv_source == "Constant":
            constant_iv = st.number_input("Constant IV (%)", 1.0, 200.0, 20.0) / 100
        iv_multiplier = st.number_input(
            "IV multiplier", 0.1, 5.0, 1.0, step=0.05,
            help="Scale the volatility proxy, e.g. 0.9 if the index overstates the IV of the strikes you trade.",
        )
        skew = st.number_input(
            "Put skew", 0.0, 5.0, 0.0, step=0.1,
            help="IV(K) = IV x (1 - skew x ln(K/S)). 0 = flat. ~0.5-1.5 approximates typical index skew "
                 "(OTM puts priced richer than OTM calls).",
        )
        div_yield = st.number_input("Dividend yield (%)", 0.0, 20.0, 0.6, step=0.1) / 100
        use_tbill = st.checkbox("Risk-free rate from 13-week T-bill (^IRX)", True)
        constant_rate = 0.04
        if not use_tbill:
            constant_rate = st.number_input("Risk-free rate (%)", 0.0, 20.0, 4.0, step=0.25) / 100
        strike_increment = st.number_input("Strike increment ($)", 0.5, 50.0, 1.0, step=0.5)
        expiration_cycle = st.selectbox(
            "Listed expirations", EXPIRATION_CYCLES, index=0,
            help="Which expirations exist. Each leg uses the expiration closest to its DTE "
                 "(or exactly its DTE, with 'Exact DTE match').",
        )

    with st.expander("Costs", expanded=False):
        fee_model = st.selectbox(
            "Commissions", FEE_MODELS, index=0,
            help="IBKR Pro Fixed: $0.65/contract ($1 min/order); stock $0.005/share ($1 min). "
                 "Tiered adds clearing/exchange estimates. Expiring options pay no commission.",
        )
        custom_fee = 0.65
        if fee_model == "Custom per contract":
            custom_fee = st.number_input("Fee per contract ($)", 0.0, 10.0, 0.65, step=0.05)
        slippage = st.number_input(
            "Slippage per option leg ($/share)", 0.0, 5.0, 0.0, step=0.01,
            help="Paid on every option fill vs. the model mid, e.g. 0.02 = $2 per contract per side.",
        )

    with st.expander("Capital", expanded=False):
        margin_model = st.selectbox(
            "Buying power model", MARGIN_MODELS, index=0,
            help="Reg-T (approx): naked shorts ~20% of underlying, spreads = max loss, debits paid in full. "
                 "Max loss: cash-secured (e.g. short put = strike x 100 - credit).",
        )
        account_size = st.number_input(
            "Account size ($, 0 = peak capital used)", 0.0, 1e9, 0.0, step=10_000.0,
            help="Capital base for returns, drawdown % and CAGR. 0 uses the peak buying power the strategy tied up.",
        )


# --------------------------------------------------------------------------- strategy legs
st.title("Options Backtester")
st.caption(
    f"{ticker} · {start:%b %d, %Y} → {end:%b %d, %Y} · daily closes · options priced with Black-Scholes "
    f"from {iv_source.split(' (')[0]}"
)

st.subheader("Strategy legs")
p1, p2, _ = st.columns([3, 1, 3], vertical_alignment="bottom")
preset = p1.selectbox("Preset", list(PRESETS))
if p2.button("Load preset"):
    load_preset(preset)
    st.rerun()


def render_leg(n: int, leg: dict) -> str | None:
    """Draw the inputs for one leg, update ``leg`` in place, return its id if removed."""
    lid = leg["id"]
    removed = None
    with st.container(border=True):
        head, rm = st.columns([8, 1], vertical_alignment="center")
        title = head.empty()
        if rm.button("Remove", key=f"{lid}_rm"):
            removed = lid

        # Asset | Direction | Type | Quantity | Strike selection | Relative to leg | Value | DTE
        cols = st.columns([1.1, 1, 1, 1, 1.5, 1.1, 1.1, 1.1])
        leg["asset"] = cols[0].selectbox("Asset", ["Option", "Stock"], index=["Option", "Stock"].index(leg["asset"]),
                                         key=f"{lid}_asset")
        leg["direction"] = cols[1].selectbox("Direction", ["Buy", "Sell"],
                                             index=["Buy", "Sell"].index(leg["direction"]), key=f"{lid}_dir")
        if leg["asset"] == "Stock":
            leg["quantity"] = int(cols[2].number_input("Shares", 1, 1_000_000, int(leg["quantity"]),
                                                       key=f"{lid}_qty_sh"))
        else:
            leg["option_type"] = cols[2].selectbox("Type", ["Call", "Put"],
                                                   index=["Call", "Put"].index(leg["option_type"]), key=f"{lid}_type")
            leg["quantity"] = int(cols[3].number_input("Quantity", 1, 10_000, int(leg["quantity"]), key=f"{lid}_qty"))
            leg["strike_method"] = cols[4].selectbox(
                "Strike selection", STRIKE_METHODS, index=STRIKE_METHODS.index(leg["strike_method"]),
                key=f"{lid}_method", help="\n\n".join(f"**{k}**: {v}" for k, v in STRIKE_METHOD_HELP.items()),
            )
            method = leg["strike_method"]
            if method == "$ from leg":
                earlier = list(range(1, n))
                if earlier:
                    ref = leg["ref_leg"] if leg["ref_leg"] in earlier else earlier[-1]
                    leg["ref_leg"] = cols[5].selectbox("Relative to leg", earlier, index=earlier.index(ref),
                                                       key=f"{lid}_ref")
                else:
                    cols[5].warning("Needs an earlier leg")
            if method != "ATM":
                label = {"Delta": "Delta", "Premium": "Premium ($)", "% from price": "Offset (%)"}.get(method, "Offset ($)")
                leg["strike_value"] = float(cols[6].number_input(
                    label, value=float(leg["strike_value"]), step=0.05 if method == "Premium" else 1.0,
                    key=f"{lid}_value", help=STRIKE_METHOD_HELP[method],
                ))
            leg["dte"] = int(cols[7].number_input("Expiration (DTE)", 1, 1000, int(leg["dte"] or 45),
                                                  key=f"{lid}_dte"))
        title.markdown(f"**Leg {n}** · {Leg(**{k: leg[k] for k in LEG_FIELDS}).describe()}")
    return removed


removed_ids = [rid for n, leg in enumerate(st.session_state.legs, start=1) if (rid := render_leg(n, leg))]
if removed_ids:
    st.session_state.legs = [leg for leg in st.session_state.legs if leg["id"] not in removed_ids]
    st.rerun()

a1, a2, _ = st.columns([1, 1, 4])
if a1.button("＋ Add option leg"):
    st.session_state.legs.append(new_leg_state(Leg()))
    st.rerun()
if a2.button("＋ Add stock leg"):
    st.session_state.legs.append(new_leg_state(Leg(asset="Stock", direction="Buy", quantity=100)))
    st.rerun()


# --------------------------------------------------------------------------- entry / exit conditions
entry_col, exit_col = st.columns(2)
with entry_col, st.container(border=True):
    st.subheader("Entry conditions")
    st.caption("All optional. Default: open one trade every trading day, no limits.")
    day_mode = st.radio("Enter", ["Every trading day", "Specific weekdays"], horizontal=True)
    weekdays = None
    if day_mode == "Specific weekdays":
        picked = st.multiselect("Weekdays", WEEKDAYS, default=["Fri"])
        weekdays = [WEEKDAYS.index(d) for d in picked]
    exact_dte = st.checkbox(
        "Only enter on an exact DTE match",
        help="Enter only when an expiration exists exactly at each leg's DTE (e.g. 45 days before a monthly expiry).",
    )
    max_open = None
    c1, c2 = st.columns([2, 1], vertical_alignment="center")
    if c1.checkbox("Limit active trades"):
        max_open = int(c2.number_input("Max open trades", 1, 1000, 1, label_visibility="collapsed"))
    vix_min = vix_max = None
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    if c1.checkbox("Enter only if VIX is within range"):
        vix_min = float(c2.number_input("VIX min", 0.0, 200.0, 0.0, step=1.0))
        vix_max = float(c3.number_input("VIX max", 0.0, 200.0, 30.0, step=1.0))

with exit_col, st.container(border=True):
    st.subheader("Exit conditions")
    st.caption("All optional. Default: hold to expiration (the first leg to expire closes the trade).")

    def exit_rule(label: str, default: float, key: str, *, step: float = 1.0, minimum: float = 0.0) -> float | None:
        c1, c2 = st.columns([2, 1], vertical_alignment="center")
        if c1.checkbox(label, key=f"use_{key}"):
            return c2.number_input(label, float(minimum), 100_000.0, float(default), step=float(step), key=key,
                                   label_visibility="collapsed")
        return None

    exit_dte = exit_rule("Exit at DTE", 21, "exit_dte")
    max_days = exit_rule("Exit after days in trade (calendar)", 30, "max_days", minimum=1.0)
    stop_loss = exit_rule("Stop loss (% of premium)", 200, "stop_loss", step=10.0, minimum=1.0)
    take_profit = exit_rule("Take profit (% of premium)", 50, "take_profit", step=5.0, minimum=1.0)
    vix_exit = exit_rule("Exit all & pause entries if VIX above", 35, "vix_exit")
    st.caption("Exits are checked once per day at the close, so stop losses can fill beyond their level on gaps.")


def build_config() -> BacktestConfig:
    legs = [Leg(**{k: leg[k] for k in LEG_FIELDS}) for leg in st.session_state.legs]
    return BacktestConfig(
        legs=legs,
        entry=EntryConfig(weekdays=weekdays, exact_dte=exact_dte, max_open_trades=max_open,
                          vix_min=vix_min, vix_max=vix_max),
        exit=ExitConfig(
            exit_dte=int(exit_dte) if exit_dte is not None else None,
            max_days_in_trade=int(max_days) if max_days is not None else None,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            vix_exit_above=vix_exit,
        ),
        expiration_cycle=expiration_cycle,
        strike_increment=strike_increment,
        iv_multiplier=iv_multiplier,
        skew=skew,
        div_yield=div_yield,
        fee_model=fee_model,
        custom_fee_per_contract=custom_fee,
        slippage_per_share=slippage,
        margin_model=margin_model,
        account_size=account_size or None,
    )


if st.button("Run backtest", type="primary"):
    cfg = build_config()
    errors = cfg.validate()
    if start >= end:
        errors.append("Start date must be before end date.")
    if errors:
        for e in errors:
            st.error(e)
    else:
        try:
            raw = load_raw(ticker, start, end)
            data, notes = prepare_inputs(raw, start, end, iv_source, hv_window, constant_iv, use_tbill, constant_rate)
            with st.spinner("Running backtest…"):
                result = run_backtest(data, cfg)
                preview = preview_trade(data, cfg)
            st.session_state.run = {
                "result": result, "data": data, "notes": notes, "ticker": ticker, "preview": preview,
            }
        except Exception as exc:  # surface data/config problems in the UI
            st.session_state.pop("run", None)
            st.error(f"Backtest failed: {exc}")


# --------------------------------------------------------------------------- results
run = st.session_state.get("run")
if not run:
    st.info("Configure the legs and conditions, then press **Run backtest**.")
    st.stop()

result, data, th = run["result"], run["data"], chart_theme()
daily, trades, capital = result.daily, result.trades, result.capital
for note in run["notes"]:
    st.warning(note)

st.divider()
st.header("Results")
if trades.empty or capital <= 0:
    st.warning("No trades were opened. Loosen the entry conditions or check the leg settings.")
    st.stop()

bh_pnl = buy_and_hold_pnl(daily["adj_close"], capital)
strat = equity_stats(daily["pnl"], capital)
bench = equity_stats(bh_pnl, capital)
ts = trade_stats(trades)

st.caption(
    f"{run['ticker']} · {daily.index[0]:%b %d, %Y} → {daily.index[-1]:%b %d, %Y} · {len(daily):,} trading days · "
    f"capital base {money(capital)} "
    f"({'account size' if result.config.account_size else 'peak buying power used'}) · "
    f"buy & hold invests the same capital on day one, dividends reinvested"
)
if result.skipped_entries:
    st.caption(f"{result.skipped_entries:,} entry days skipped because no expiration matched the DTE.")

s_col, b_col = st.columns(2)
for col, name, stats in ((s_col, "Strategy", strat), (b_col, "Buy & hold", bench)):
    with col, st.container(border=True):
        st.markdown(f"**{name}**")
        m = st.columns(5)
        m[0].metric("Total P/L", money(stats["total_pnl"]))
        m[1].metric("Max drawdown", pct(-stats["max_dd_pct"]), help=money(-stats["max_dd_dollar"]))
        m[2].metric("Return on cap used", pct(stats["return_on_capital"]))
        m[3].metric("CAGR", pct(stats["cagr"]))
        m[4].metric("MAR ratio", ratio(stats["mar"]))

tab_perf, tab_trades, tab_details, tab_log, tab_payoff = st.tabs(
    ["Performance", "Trade analysis", "Details", "Trade log", "Strategy payoff"]
)

with tab_perf:
    st.markdown("##### Cumulative P/L")
    show(charts.equity_chart(daily["pnl"], bh_pnl, th))
    st.markdown("##### Drawdown")
    show(charts.drawdown_chart(drawdown_pct(daily["pnl"], capital), drawdown_pct(bh_pnl, capital), th))
    st.markdown("##### Monthly P/L (strategy)")
    monthly = monthly_pnl(daily["pnl"])
    show(charts.monthly_heatmap(monthly, th))
    with st.expander("Monthly P/L table"):
        table = monthly.copy()
        table.columns = charts.MONTHS
        table["Year"] = table.sum(axis=1, min_count=1)
        st.dataframe(table.style.format(lambda v: "" if pd.isna(v) else money(v)))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Buying power in use")
        show(charts.single_series_chart(daily["bpr_used"], th, name="BPR in use", y_format="$,.0f"))
    with c2:
        st.markdown("##### Open trades")
        show(charts.single_series_chart(daily["open_trades"], th, name="Open trades", y_format=",.0f", shape="hv"))

with tab_trades:
    st.markdown("##### P/L per trade (by exit date)")
    show(charts.trade_pnl_chart(trades, th))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### P/L distribution")
        show(charts.pnl_histogram(trades, th))
    with c2:
        st.markdown("##### Exit reasons")
        show(charts.exit_reason_chart(trades, th))
    st.markdown("##### Volatility used for pricing vs VIX")
    show(charts.vol_chart(daily, th))


def metric_table(rows: list[tuple[str, str]]) -> None:
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value"]), hide_index=True)


with tab_details:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### Trades")
        metric_table([
            ("No. of trades", f"{ts['n_trades']:,}"),
            ("Trades with profit", f"{ts['n_wins']:,}"),
            ("Profit ratio (win rate)", pct(ts["win_rate"])),
            ("Largest individual profit", money(ts["largest_win"])),
            ("Trades with loss", f"{ts['n_losses']:,}"),
            ("Loss rate", pct(ts["loss_rate"])),
            ("Largest individual loss", money(ts["largest_loss"])),
        ])
    with c2:
        st.markdown("##### Averages")
        metric_table([
            ("Avg return per trade (on BPR)", pct(ts["avg_return_per_trade"])),
            ("Avg days in trade", f"{ts['avg_days']:.1f}"),
            ("Avg BPR per trade", money(ts["avg_bpr"])),
            ("Avg premium (credit + / debit −)", money(ts["avg_premium"])),
            ("Avg P/L per trade", money(ts["avg_pnl"])),
            ("Avg win", f"{money(ts['avg_win'])} ({pct(ts['avg_win_pct'])})"),
            ("Avg loss", f"{money(ts['avg_loss'])} ({pct(ts['avg_loss_pct'])})"),
            ("Premium capture rate", pct(ts["premium_capture"])),
            ("Profit factor", ratio(ts["profit_factor"])),
        ])
    with c3:
        st.markdown("##### Totals")
        metric_table([
            ("Total P/L", money(strat["total_pnl"])),
            ("Capital used", money(capital)),
            ("Return on capital used", pct(strat["return_on_capital"])),
            ("CAGR", pct(strat["cagr"])),
            ("MAR ratio", ratio(strat["mar"])),
            ("Max drawdown", f"{pct(-strat['max_dd_pct'])} ({money(-strat['max_dd_dollar'])})"),
            ("Sharpe ratio", ratio(strat["sharpe"])),
            ("Total premium", money(ts["total_premium"])),
            (f"Total fees ({result.config.fee_model})", money(ts["total_fees"], 2)),
        ])

with tab_log:
    log = trades.copy()
    log["return_pct"] = log["return_pct"] * 100
    log["entry_iv"] = log["entry_iv"] * 100
    money_col = st.column_config.NumberColumn
    st.dataframe(
        log,
        hide_index=True,
        column_config={
            "trade_id": st.column_config.NumberColumn("#", format="%d"),
            "entry_date": st.column_config.DateColumn("Entry", format="YYYY-MM-DD"),
            "exit_date": st.column_config.DateColumn("Exit", format="YYYY-MM-DD"),
            "expiration": st.column_config.DateColumn("First expiry", format="YYYY-MM-DD"),
            "days": st.column_config.NumberColumn("Days", format="%d"),
            "legs": st.column_config.TextColumn("Legs (strike, expiry)", width="large"),
            "entry_spot": money_col("Entry spot", format="$%.2f"),
            "exit_spot": money_col("Exit spot", format="$%.2f"),
            "entry_vix": money_col("Entry VIX", format="%.1f"),
            "entry_iv": money_col("Entry IV %", format="%.1f"),
            "premium": money_col("Premium", format="$%.2f"),
            "bpr": money_col("BPR", format="$%.0f"),
            "fees": money_col("Fees", format="$%.2f"),
            "pnl": money_col("P/L", format="$%.2f"),
            "return_pct": money_col("Return % (BPR)", format="%.1f%%"),
            "max_open_profit": money_col("Max open profit", format="$%.0f"),
            "max_open_loss": money_col("Max open loss", format="$%.0f"),
            "exit_reason": st.column_config.TextColumn("Exit reason"),
        },
    )
    st.download_button(
        "Download trades (CSV)", trades.to_csv(index=False).encode(), file_name=f"{run['ticker']}_trades.csv",
        mime="text/csv",
    )

with tab_payoff:
    tr = run["preview"]
    if tr is None:
        st.info("No expiration matched the legs on the last day, so there is no payoff preview.")
    else:
        last = data.iloc[-1]
        st.caption(
            f"The trade the strategy would open on {data.index[-1]:%b %d, %Y} at {money(tr.entry_spot, 2)} "
            f"(IV {tr.entry_iv:.1%}). P/L includes entry fees."
        )
        grid, at_exp, now = payoff_curve(tr, float(last["rate"]), result.config.div_yield, tr.entry_iv,
                                         result.config.skew)
        show(charts.payoff_chart(grid, at_exp, now, tr.entry_spot, th))
        m = st.columns(4)
        m[0].metric("Net premium (credit + / debit −)", money(tr.premium, 2))
        m[1].metric("Buying power", money(tr.bpr))
        m[2].metric("Max profit at expiry (shown range)", money(float(at_exp.max())))
        m[3].metric("Max loss at expiry (shown range)", money(float(at_exp.min())))
        st.code(tr.describe().replace(" | ", "\n"), language=None)

with st.expander("How this backtest works"):
    st.markdown(
        """
- **Prices**: daily closes from Yahoo Finance. Free historical option chains do not exist, so every option is
  priced with **Black-Scholes** using the chosen volatility proxy (VXN for QQQ by default) × the IV multiplier,
  the optional put skew, the 13-week T-bill rate and the dividend yield. Treat results as a model, not fills.
- **Strikes** snap to the strike increment. *Delta* picks the strike whose model delta is closest to the target;
  *Premium* picks the strike whose price is closest to the target.
- **Expirations** come from the chosen listing cycle; each leg takes the expiration nearest its DTE (or exactly
  its DTE with *exact DTE match*).
- **Entries and exits** happen at the daily close. At expiration, legs settle at intrinsic value with no fee
  (no assignment modelled). Trades still open on the last day are closed at the model price ("End of data").
- **Stop loss / take profit** are a % of the net option premium (credit or debit) and are checked at the close.
- **Buying power** is an approximation of broker margin (see *Capital* in the sidebar). *Capital used* is the
  peak total buying power tied up at once, unless you set an account size. Returns, drawdown %, CAGR and
  MAR (= CAGR / max drawdown) use that capital base.
- **Buy & hold** invests the same capital in the underlying on day one (adjusted close, dividends reinvested).
"""
    )
