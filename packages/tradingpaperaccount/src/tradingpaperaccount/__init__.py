"""TradingPaperAccount.

Rebalance Alpaca paper accounts to a target set of portfolio weights.

The pure rebalance math (:mod:`tradingpaperaccount.rebalance`),
configuration (:mod:`tradingpaperaccount.config`) and data structures
(:mod:`tradingpaperaccount.models`) do not import the Alpaca SDK. The
SDK-backed client lives in :mod:`tradingpaperaccount.client` and is imported
on demand.
"""

from tradingpaperaccount.accounts import get_positions
from tradingpaperaccount.config import (
    ConfigError,
    PaperAccountConfig,
    PaperAccountSummary,
    configured_accounts,
    list_paper_accounts,
    load_weights_file,
    resolve_account,
)
from tradingpaperaccount.models import (
    AccountState,
    ExecutionReport,
    OrderIntent,
    OrderResult,
    Position,
    RebalancePlan,
)
from tradingpaperaccount.rebalance import (
    DEFAULT_CASH_BUFFER,
    DEFAULT_MIN_ORDER_VALUE,
    RebalanceError,
    compute_rebalance_plan,
    validate_weights,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AccountState",
    "ConfigError",
    "DEFAULT_CASH_BUFFER",
    "DEFAULT_MIN_ORDER_VALUE",
    "ExecutionReport",
    "OrderIntent",
    "OrderResult",
    "PaperAccountConfig",
    "PaperAccountSummary",
    "Position",
    "RebalanceError",
    "RebalancePlan",
    "compute_rebalance_plan",
    "configured_accounts",
    "get_positions",
    "list_paper_accounts",
    "load_weights_file",
    "resolve_account",
    "validate_weights",
]
