"""Commission models (Interactive Brokers Pro, US options and stocks).

IBKR Pro Fixed  – options: $0.65/contract ($0.50 if premium $0.05-$0.10,
                  $0.25 if premium < $0.05), $1.00 minimum per order.
                  stocks: $0.005/share, $1.00 minimum, max 1% of trade value.
IBKR Pro Tiered – options: same commission schedule (<=10k contracts/month)
                  plus clearing ($0.02) and an estimate of exchange/regulatory
                  fees ($0.05) per contract, $1.00 minimum per order.
                  stocks: $0.0035/share + ~$0.0032/share clearing/exchange,
                  $0.35 minimum, max 1% of trade value.
Options that expire are settled at intrinsic value with no commission.
"""

from __future__ import annotations

TIERED_OPTION_EXTRA = 0.02 + 0.05
TIERED_STOCK_EXTRA = 0.0002 + 0.0030


def _ibkr_option_rate(premium: float) -> float:
    if premium >= 0.10:
        return 0.65
    if premium >= 0.05:
        return 0.50
    return 0.25


def option_order_fee(contracts: list[tuple[int, float]], model: str, custom_per_contract: float = 0.65) -> float:
    """Fee for one multi-leg option order. ``contracts`` = [(quantity, price per share), ...]."""
    if not contracts or model == "None":
        return 0.0
    if model == "Custom per contract":
        return sum(q for q, _ in contracts) * custom_per_contract
    fee = sum(q * _ibkr_option_rate(p) for q, p in contracts)
    if model == "IBKR Pro Tiered":
        fee += sum(q for q, _ in contracts) * TIERED_OPTION_EXTRA
    return max(fee, 1.00)


def stock_order_fee(shares: int, price: float, model: str) -> float:
    if shares <= 0 or model in ("None", "Custom per contract"):
        return 0.0
    cap = 0.01 * shares * price
    if model == "IBKR Pro Tiered":
        fee = max(shares * 0.0035, 0.35) + shares * TIERED_STOCK_EXTRA
    else:
        fee = max(shares * 0.005, 1.00)
    return min(fee, cap) if cap > 0 else fee
