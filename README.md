# opeval: Options Backtester

A Streamlit app for backtesting multi-leg option strategies (with optional stock legs) on any Yahoo Finance
ticker. The default is QQQ.

## Run

```bash
uv sync
uv run streamlit run app.py
```

If you're behind a corporate TLS proxy (Zscaler, for example), add `--native-tls` to the `uv` commands, or set
`UV_NATIVE_TLS=1`. Then run the tests:

```bash
uv run pytest
```

## What you can configure

**Market**: the ticker and interval (last 1 to 20 years, or a custom date range).

**Legs**: add as many as you need.

| Field | Options |
|---|---|
| Asset | Option or Stock |
| Direction | Buy / Sell |
| Type | Call / Put |
| Quantity | Contracts (shares for stock legs) |
| Strike selection | Delta (e.g. 35), $ from price (e.g. -5), % from price, ATM, Premium (target price), $ from another leg |
| Expiration | DTE. The leg uses the listed expiration closest to it. |

Presets are included: put/call credit spread, short put, iron condor, strangle, covered call, long call,
straddle, calendar and poor man's covered call.

**Entry conditions** (all optional): every trading day or chosen weekdays, exact DTE match only, a cap on
active trades, and a VIX range.

**Exit conditions** (all optional; by default the trade is held to expiration): exit at a given DTE, exit after
N days in the trade, stop loss as a % of premium, take profit as a % of premium, and a VIX-above exit. The VIX
exit also pauses new entries.

**Costs**: IBKR Pro Fixed, IBKR Pro Tiered, a custom per-contract fee, or no fees, plus optional slippage.

## Results

- Strategy compared with buy & hold: total P/L, max drawdown, return on capital used, CAGR and MAR ratio
- Charts: cumulative P/L, drawdown, monthly P/L heatmap, buying power in use, open trades, P/L per trade,
  P/L distribution, exit reasons, volatility, and a payoff diagram of today's trade
- Details: trade counts, win and loss rates, largest win and loss, average return, days, BPR, premium,
  P/L, win and loss per trade, premium capture rate, profit factor, totals, CAGR, MAR, Sharpe, total premium
  and total fees
- A full trade log you can download as CSV

## Modelling assumptions

Free historical option chains don't exist, so this is a **model-based** backtest:

- Underlying, VIX, VXN and the 13-week T-bill come from Yahoo Finance as daily closes.
- Options are priced with Black-Scholes. The volatility input is an IV proxy: VXN (the default for QQQ), VIX,
  realized volatility or a constant. You can scale it with a multiplier and add an optional linear put skew.
- Trades open and close at the daily close. Stops and targets are checked once a day.
- At expiration, options settle at intrinsic value with no assignment and no commission.
- Buying power is an approximation of Reg-T margin (or cash-secured max loss). *Capital used* is the peak
  buying power tied up at once, unless you set an account size.
- Buy & hold invests the same capital on day one, using the adjusted close (dividends reinvested).

## Layout

```
app.py              Streamlit UI
opeval/data.py      Yahoo Finance download and IV/rate inputs
opeval/pricing.py   Black-Scholes, skew, strike selection
opeval/engine.py    Daily backtest loop, expirations, buying power, payoff
opeval/fees.py      IBKR commission models
opeval/metrics.py   Equity and trade statistics
opeval/charts.py    Plotly figures (light and dark)
opeval/presets.py   Strategy presets
tests/              pytest suite (synthetic data, no network)
```
