"""
Realistic Portfolio Management Engine for the Ho Chi Minh Stock Exchange (HOSE).

Modules:
    - data_tri: Ingestion, OHLC auditing, and Forward-Compounded Total Return Index (TRI) calculation.
    - rmt_erc_allocator: Marchenko-Pastur RMT correlation denoising, Ward clustering, and ERC optimization.
    - hose_simulator: Event-driven execution simulator enforcing HOSE lot size, T+2.5 settlement, and Rolling HWM.
"""

from .data_tri import (
    HOSE_MARKET_RULES,
    FALLBACK_VN100,
    fetch_vn100_yfinance,
    audit_market_data,
    compute_realistic_tri,
)
from .rmt_erc_allocator import (
    DEFAULT_RISK_CONFIG,
    prepare_universe_matrices,
    denoise_correlation_rmt,
    select_cluster_representatives,
    optimize_erc_weights,
    compute_benchmark_weights,
    generate_walk_forward_weights,
)
from .hose_simulator import (
    DEFAULT_EXEC_CONFIG,
    compute_hwm_scalar,
    run_hose_simulation,
)

__all__ = [
    "HOSE_MARKET_RULES",
    "FALLBACK_VN100",
    "fetch_vn100_yfinance",
    "audit_market_data",
    "compute_realistic_tri",
    "DEFAULT_RISK_CONFIG",
    "prepare_universe_matrices",
    "denoise_correlation_rmt",
    "select_cluster_representatives",
    "optimize_erc_weights",
    "compute_benchmark_weights",
    "generate_walk_forward_weights",
    "DEFAULT_EXEC_CONFIG",
    "compute_hwm_scalar",
    "run_hose_simulation",
]