"""Configuration objects for a backtest."""

from __future__ import annotations

from dataclasses import dataclass, field

STRIKE_METHODS = ["Delta", "$ from price", "% from price", "ATM", "Premium", "$ from leg"]
STRIKE_METHOD_HELP = {
    "Delta": "Target |delta| x100, e.g. 35 = 0.35 delta (puts use -0.35 automatically)",
    "$ from price": "Dollars from the underlying price, e.g. -5 = $5 below spot",
    "% from price": "Percent from the underlying price, e.g. -10 = 10% below spot",
    "ATM": "Strike nearest the underlying price",
    "Premium": "Option price closest to this value ($ per share), e.g. 2.50",
    "$ from leg": "Dollars from another option leg's strike, e.g. -5 = $5 below that leg",
}
EXPIRATION_CYCLES = ["Daily", "Mon/Wed/Fri", "Weekly (Fri)", "Monthly (3rd Fri)"]
FEE_MODELS = ["IBKR Pro Fixed", "IBKR Pro Tiered", "Custom per contract", "None"]
MARGIN_MODELS = ["Reg-T (approx)", "Max loss / cash-secured"]
IV_SOURCES = ["VXN (Nasdaq-100 vol index)", "VIX", "Historical volatility", "Constant"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


@dataclass
class Leg:
    asset: str = "Option"  # "Option" | "Stock"
    direction: str = "Sell"  # "Buy" | "Sell"
    option_type: str = "Put"  # "Call" | "Put"
    quantity: int = 1  # contracts for options, shares for stock
    strike_method: str = "Delta"
    strike_value: float = 30.0
    ref_leg: int = 1  # 1-based leg number used by "$ from leg"
    dte: int = 45

    @property
    def is_stock(self) -> bool:
        return self.asset == "Stock"

    @property
    def sign(self) -> int:
        return 1 if self.direction == "Buy" else -1

    def describe(self) -> str:
        if self.is_stock:
            return f"{self.direction} {self.quantity} sh"
        if self.strike_method == "ATM":
            strike = "ATM"
        elif self.strike_method == "$ from leg":
            strike = f"L{self.ref_leg} {self.strike_value:+g}"
        else:
            strike = f"{self.strike_value:g} {self.strike_method}"
        return f"{self.direction} {self.quantity} {self.option_type} @ {strike}, {self.dte} DTE"


@dataclass
class EntryConfig:
    weekdays: list[int] | None = None  # None = every trading day; 0=Mon .. 4=Fri
    exact_dte: bool = False
    max_open_trades: int | None = None
    vix_min: float | None = None
    vix_max: float | None = None


@dataclass
class ExitConfig:
    exit_dte: int | None = None
    max_days_in_trade: int | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    vix_exit_above: float | None = None


@dataclass
class BacktestConfig:
    legs: list[Leg]
    entry: EntryConfig = field(default_factory=EntryConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    expiration_cycle: str = "Daily"
    strike_increment: float = 1.0
    iv_multiplier: float = 1.0
    skew: float = 0.0
    div_yield: float = 0.006
    fee_model: str = "IBKR Pro Fixed"
    custom_fee_per_contract: float = 0.65
    slippage_per_share: float = 0.0
    margin_model: str = "Reg-T (approx)"
    account_size: float | None = None  # None = use peak capital used

    def validate(self) -> list[str]:
        errors = []
        if not self.legs:
            errors.append("Add at least one leg.")
        for i, leg in enumerate(self.legs, start=1):
            if leg.quantity <= 0:
                errors.append(f"Leg {i}: quantity must be positive.")
            if leg.is_stock:
                continue
            if leg.dte < 1:
                errors.append(f"Leg {i}: DTE must be at least 1.")
            if leg.strike_method not in STRIKE_METHODS:
                errors.append(f"Leg {i}: unknown strike method {leg.strike_method!r}.")
            if leg.strike_method == "Delta" and not 0 < abs(leg.strike_value) < 100:
                errors.append(f"Leg {i}: delta must be between 0 and 100.")
            if leg.strike_method == "Premium" and leg.strike_value <= 0:
                errors.append(f"Leg {i}: target premium must be positive.")
            if leg.strike_method == "$ from leg":
                ref = leg.ref_leg
                if not 1 <= ref < i:
                    errors.append(f"Leg {i}: '$ from leg' must reference an earlier leg (1..{i - 1}).")
                elif self.legs[ref - 1].is_stock:
                    errors.append(f"Leg {i}: referenced leg {ref} is a stock leg.")
        if self.strike_increment <= 0:
            errors.append("Strike increment must be positive.")
        return errors
