"""Ready-made leg setups that can be loaded into the strategy builder."""

from __future__ import annotations

from .models import Leg


def _opt(direction, option_type, method, value, dte, qty=1, ref=1) -> Leg:
    return Leg("Option", direction, option_type, qty, method, value, ref, dte)


PRESETS: dict[str, list[Leg]] = {
    "Put credit spread (35Δ / -5, 60 DTE)": [
        _opt("Sell", "Put", "Delta", 35, 60),
        _opt("Buy", "Put", "$ from leg", -5, 60),
    ],
    "Short put (30Δ, 45 DTE)": [_opt("Sell", "Put", "Delta", 30, 45)],
    "Call credit spread (30Δ / +5, 45 DTE)": [
        _opt("Sell", "Call", "Delta", 30, 45),
        _opt("Buy", "Call", "$ from leg", 5, 45),
    ],
    "Iron condor (16Δ, $10 wings, 45 DTE)": [
        _opt("Sell", "Put", "Delta", 16, 45),
        _opt("Buy", "Put", "$ from leg", -10, 45, ref=1),
        _opt("Sell", "Call", "Delta", 16, 45),
        _opt("Buy", "Call", "$ from leg", 10, 45, ref=3),
    ],
    "Short strangle (16Δ, 45 DTE)": [
        _opt("Sell", "Put", "Delta", 16, 45),
        _opt("Sell", "Call", "Delta", 16, 45),
    ],
    "Covered call (100 sh + 30Δ call, 30 DTE)": [
        Leg("Stock", "Buy", "Call", 100, "ATM", 0, 1, 0),
        _opt("Sell", "Call", "Delta", 30, 30),
    ],
    "Long call (50Δ, 90 DTE)": [_opt("Buy", "Call", "Delta", 50, 90)],
    "Long straddle (ATM, 45 DTE)": [
        _opt("Buy", "Call", "ATM", 0, 45),
        _opt("Buy", "Put", "ATM", 0, 45),
    ],
    "Put calendar (ATM, 30/60 DTE)": [
        _opt("Sell", "Put", "ATM", 0, 30),
        _opt("Buy", "Put", "$ from leg", 0, 60),
    ],
    "Poor man's covered call (80Δ 180 DTE / 30Δ 30 DTE)": [
        _opt("Buy", "Call", "Delta", 80, 180),
        _opt("Sell", "Call", "Delta", 30, 30),
    ],
}

DEFAULT_PRESET = next(iter(PRESETS))
