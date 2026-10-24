"""Plotly figures for the results page (light and dark aware)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass(frozen=True)
class Theme:
    strategy: str
    benchmark: str
    positive: str
    negative: str
    midpoint: str
    text: str
    text_secondary: str
    muted: str
    grid: str
    baseline: str
    surface: str
    border: str


LIGHT = Theme(
    strategy="#2a78d6",
    benchmark="#eb6834",
    positive="#2a78d6",
    negative="#e34948",
    midpoint="#f0efec",
    text="#0b0b0b",
    text_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    surface="#fcfcfb",
    border="rgba(11,11,11,0.10)",
)
DARK = Theme(
    strategy="#3987e5",
    benchmark="#d95926",
    positive="#3987e5",
    negative="#e66767",
    midpoint="#383835",
    text="#f0efec",
    text_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    baseline="#383835",
    surface="#1a1a19",
    border="rgba(255,255,255,0.10)",
)


def theme_for(mode: str | None) -> Theme:
    return DARK if mode == "dark" else LIGHT


def _style(fig: go.Figure, th: Theme, *, height: int = 320, y_format: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=36 if legend else 12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=th.text_secondary),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=th.text_secondary)),
        hoverlabel=dict(bgcolor=th.surface, bordercolor=th.border, font=dict(family=FONT, color=th.text)),
        bargap=0.15,
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, showline=True, linecolor=th.baseline, tickfont=dict(color=th.muted),
        showspikes=True, spikecolor=th.baseline, spikethickness=1, spikedash="solid", spikemode="across",
    )
    fig.update_yaxes(
        gridcolor=th.grid, gridwidth=1, zeroline=True, zerolinecolor=th.baseline, zerolinewidth=1,
        tickfont=dict(color=th.muted), tickformat=y_format,
    )
    return fig


def _end_label(fig: go.Figure, x, y, text: str, th: Theme) -> None:
    fig.add_annotation(
        x=x, y=y, text=text, showarrow=False, xanchor="left", xshift=6, font=dict(color=th.text_secondary, size=12)
    )


def equity_chart(strategy_pnl: pd.Series, benchmark_pnl: pd.Series, th: Theme) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strategy_pnl.index, y=strategy_pnl, name="Strategy", mode="lines",
        line=dict(color=th.strategy, width=2), hovertemplate="%{y:$,.0f}<extra>Strategy</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=benchmark_pnl.index, y=benchmark_pnl, name="Buy & hold", mode="lines",
        line=dict(color=th.benchmark, width=2), hovertemplate="%{y:$,.0f}<extra>Buy & hold</extra>",
    ))
    _style(fig, th, height=380, y_format="$,.0f")
    fig.update_layout(hovermode="x unified", margin=dict(r=110))
    _end_label(fig, strategy_pnl.index[-1], strategy_pnl.iloc[-1], f"Strategy {strategy_pnl.iloc[-1]:,.0f}", th)
    _end_label(fig, benchmark_pnl.index[-1], benchmark_pnl.iloc[-1], f"B&H {benchmark_pnl.iloc[-1]:,.0f}", th)
    return fig


def drawdown_chart(strategy_dd: pd.Series, benchmark_dd: pd.Series, th: Theme) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strategy_dd.index, y=strategy_dd, name="Strategy", mode="lines",
        line=dict(color=th.strategy, width=2), hovertemplate="%{y:.1%}<extra>Strategy</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=benchmark_dd.index, y=benchmark_dd, name="Buy & hold", mode="lines",
        line=dict(color=th.benchmark, width=2), hovertemplate="%{y:.1%}<extra>Buy & hold</extra>",
    ))
    _style(fig, th, height=280, y_format=".0%")
    fig.update_layout(hovermode="x unified")
    return fig


def trade_pnl_chart(trades: pd.DataFrame, th: Theme) -> go.Figure:
    colors = np.where(trades["pnl"] >= 0, th.positive, th.negative)
    custom = pd.DataFrame({
        "id": trades["trade_id"],
        "entry": trades["entry_date"].dt.strftime("%Y-%m-%d"),
        "reason": trades["exit_reason"],
        "ret": trades["return_pct"] * 100,
    }).to_numpy(dtype=object)  # object keeps numbers numeric for the hover format
    fig = go.Figure(go.Bar(
        x=trades["exit_date"], y=trades["pnl"], marker=dict(color=colors, line=dict(width=0)),
        customdata=custom,
        hovertemplate=(
            "Trade #%{customdata[0]}<br>Entry %{customdata[1]} · Exit %{x|%Y-%m-%d}<br>"
            "P&L %{y:$,.0f} (%{customdata[3]:.1f}% of BPR)<br>%{customdata[2]}<extra></extra>"
        ),
    ))
    _style(fig, th, height=300, y_format="$,.0f", legend=False)
    fig.update_xaxes(showspikes=False)
    return fig


def monthly_heatmap(table: pd.DataFrame, th: Theme) -> go.Figure:
    limit = float(np.nanmax(np.abs(table.to_numpy()))) if table.notna().any().any() else 1.0
    fig = go.Figure(go.Heatmap(
        z=table.to_numpy(), x=MONTHS, y=[str(y) for y in table.index],
        colorscale=[[0.0, th.negative], [0.5, th.midpoint], [1.0, th.positive]],
        zmid=0, zmin=-limit, zmax=limit, xgap=2, ygap=2,
        colorbar=dict(tickformat="$,.0f", outlinewidth=0, tickfont=dict(color=th.muted), thickness=10),
        hovertemplate="%{x} %{y}<br>%{z:$,.0f}<extra></extra>",
    ))
    _style(fig, th, height=max(220, 34 * len(table) + 60), legend=False)
    fig.update_xaxes(showline=False, showspikes=False, side="top")
    fig.update_yaxes(autorange="reversed", showgrid=False, zeroline=False, type="category")
    return fig


def pnl_histogram(trades: pd.DataFrame, th: Theme) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=trades["pnl"], nbinsx=min(60, max(10, len(trades) // 3)),
        marker=dict(color=th.strategy, line=dict(width=0)),
        hovertemplate="P&L %{x}<br>%{y} trades<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=th.baseline, width=1))
    _style(fig, th, height=280, legend=False)
    fig.update_xaxes(tickformat="$,.0f", showspikes=False)
    fig.update_yaxes(title=dict(text="Trades", font=dict(color=th.muted)))
    return fig


def single_series_chart(series: pd.Series, th: Theme, *, name: str, y_format: str, shape: str = "linear") -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=series.index, y=series, name=name, mode="lines", line=dict(color=th.strategy, width=2, shape=shape),
        hovertemplate=f"%{{y:{y_format}}}<extra>{name}</extra>",
    ))
    _style(fig, th, height=240, y_format=y_format, legend=False)
    fig.update_layout(hovermode="x unified")
    return fig


def vol_chart(daily: pd.DataFrame, th: Theme) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["iv"], name="IV used for pricing", mode="lines",
        line=dict(color=th.strategy, width=2), hovertemplate="%{y:.1%}<extra>IV</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["vix"] / 100.0, name="VIX", mode="lines",
        line=dict(color=th.benchmark, width=2), hovertemplate="%{y:.1%}<extra>VIX</extra>",
    ))
    _style(fig, th, height=260, y_format=".0%")
    fig.update_layout(hovermode="x unified")
    return fig


def exit_reason_chart(trades: pd.DataFrame, th: Theme) -> go.Figure:
    counts = trades["exit_reason"].value_counts().sort_values()
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h", marker=dict(color=th.strategy, line=dict(width=0)),
        text=counts.values, textposition="outside", textfont=dict(color=th.text_secondary),
        hovertemplate="%{y}: %{x} trades<extra></extra>", cliponaxis=False,
    ))
    _style(fig, th, height=max(160, 44 * len(counts) + 40), legend=False)
    fig.update_layout(barcornerradius=4, margin=dict(r=40))
    fig.update_xaxes(showgrid=False, showline=False, showticklabels=False, showspikes=False)
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(color=th.text_secondary))
    return fig


def payoff_chart(grid: np.ndarray, at_expiry: np.ndarray, today: np.ndarray, spot: float, th: Theme) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grid, y=at_expiry, name="At first expiration", mode="lines",
        line=dict(color=th.strategy, width=2), hovertemplate="%{y:$,.0f}<extra>At expiration</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=grid, y=today, name="Today (model)", mode="lines",
        line=dict(color=th.benchmark, width=2), hovertemplate="%{y:$,.0f}<extra>Today</extra>",
    ))
    fig.add_vline(x=spot, line=dict(color=th.baseline, width=1))
    fig.add_annotation(x=spot, y=1, yref="paper", text=f"Spot {spot:,.2f}", showarrow=False, yanchor="bottom",
                       font=dict(color=th.muted, size=12))
    _style(fig, th, height=300, y_format="$,.0f")
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(tickformat="$,.0f")
    return fig
