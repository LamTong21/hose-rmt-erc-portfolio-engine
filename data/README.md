# Data Artifacts & Schema Specification

Large binary `.parquet` files generated during notebook execution are excluded from version control. Running the three pipeline stages sequentially produces the following datasets:

## 1. `market_tri.parquet` (Output of Stage 1)
Point-in-time daily market and Total Return Index store (`zstd` compressed).

| Column | Type | Description |
| :--- | :--- | :--- |
| `ticker` | `String` | HOSE stock symbol (VN100 constituent) |
| `trade_date` | `Date` | Trading session date |
| `open`, `high`, `low`, `close` | `Float64` | Unadjusted market prices (VND) |
| `volume_match` | `Int64` | Matched order volume (shares) |
| `value_match` | `Float64` | Matched trading value (`close * volume_match`, VND) |
| `adj_shares_factor` | `Float64` | Cumulative share multiplier reflecting splits, stock dividends & reinvested net cash dividends |
| `price_return` | `Float64` | Daily raw close-to-close return |
| `total_return` | `Float64` | Daily total return inclusive of corporate actions and dividend receivables |
| `tri` | `Float64` | Forward-compounded Total Return Index (base = `100.0`) |

## 2. `target_weights_{base,inv_vol,equal}.parquet` (Output of Stage 2)
Wide-format target allocation matrices generated via walk-forward optimization (strictly $t-1$ cutoff).

| Column | Type | Description |
| :--- | :--- | :--- |
| `trade_date` | `Date` | Execution date $t$ |
| `is_rebalance_day` | `Boolean` | `True` every 10 trading sessions (`rebalance_freq_days`) |
| `<TICKER>` | `Float64` | Target portfolio weight for each VN100 symbol (capped at `0.15`) |
| `CASH` | `Float64` | Unallocated cash weight ($1 - \sum w_i$) |

## 3. `backtest_ledger.parquet` & `trade_history.parquet` (Output of Stage 3)
Daily net asset value ledger and executed order log from the HOSE event-driven simulator.

* **`backtest_ledger.parquet`:** `trade_date`, `nav_vnd`, `cash_vnd`, `stock_value_vnd`, `actual_stock_weight`, `target_stock_weight`, `hwm_alpha`, `drawdown`, `net_daily_return`, `turnover_vnd`, `fees_tax_vnd`.
* **`trade_history.parquet`:** `trade_date`, `ticker`, `side` (`BUY`/`SELL`), `qty` (multiples of `100`), `price`, `gross_vnd`, `fee_tax_vnd`.