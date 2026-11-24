# Realistic Portfolio Management on HOSE (VN100)

A quantitative portfolio construction and event-driven execution research archive tailored to the microstructure and regulatory rules of the Ho Chi Minh Stock Exchange (HOSE).

This repository documents a three-stage research pipeline spanning data integrity auditing, Random Matrix Theory (RMT) risk-budgeted allocation, and friction-aware backtesting over the **VN100** universe (2018–2026), alongside a companion theoretical study on multi-asset Markowitz optimization in Vietnam.

---

## Repository Architecture

```text
vietnam-quant-portfolio-management/
├── README.md                                  # Project overview, methodology & architecture
├── requirements.txt                           # Python dependencies
├── notebooks/                                 # Original research notebooks (Colab workflow)
│   ├── 01_part1_immutable_core_tri.ipynb      # Data ingestion, OHLC audit & Forward-Compounded TRI
│   ├── 02_proc_universe_and_risk.ipynb        # RMT denoising, Ward clustering & ERC optimization
│   └── 03_proc_hose_execution_sim.ipynb       # Event-driven HOSE simulation (T+2.5, lot 100, HWM)
├── src/                                       # Clean, modular Python source extracted from notebooks
│   ├── __init__.py
│   ├── data_tri.py                            # Total Return Index (TRI) & DataAuditor engine
│   ├── rmt_erc_allocator.py                   # Marchenko-Pastur RMT, Ward clustering & SLSQP ERC
│   └── hose_simulator.py                      # Event-driven HOSE execution simulator
├── paper/                                     # Theoretical research & empirical thesis
│   ├── markowitz_multi_asset_thesis.md        # Multi-asset Markowitz vs. Bank Savings study (2019–2024)
│   └── figures/                               # Charts & diagnostic plots
└── data/                                      # Data schema documentation (parquet artifacts excluded)
    └── README.md
```

## Research Pipeline Overview

### Stage 1: Immutable Core & Realistic Total Return Index (`01` / `src/data_tri.py`)

Standard adjusted-close series introduce backward-looking price distortions and ignore cash dividend settlement lags. This stage builds an immutable Point-in-Time (PIT) market store:

- **Data Auditing (`audit_market_data`):** Validates OHLC logic (`high >= low`, `close > 0`) and cross-checks raw daily returns exceeding HOSE's $\pm 7\%$ price band against corporate action logs.
- **Forward-Compounded TRI (`compute_realistic_tri`):** Starts at `100.0` and tracks actual shareholder cash flows—accounting for the **5% personal income tax** on cash dividends, settlement lag (`default_cash_lag_days = 30`) before reinvestment, stock splits/bonus shares, and preferential rights issue dilution.

### Stage 2: RMT Denoising, Clustering & ERC Allocation (`02` / `src/rmt_erc_allocator.py`)

Constructs a concentrated, turnover-controlled equity portfolio strictly using information available up to $t-1$:

- **Liquidity & History Filtering:** Filters VN100 constituents requiring at least 120 sessions of history and a 20-day Average Daily Trading Value ($\text{ADTV}_{20}$) $\ge 10\text{ billion VND}$.
- **Marchenko-Pastur RMT Denoising (`denoise_correlation_rmt`):** Eigenvalues of the sample correlation matrix falling below the theoretical upper noise bound $\lambda_+ = (1 + \sqrt{N/T})^2$ are replaced by their mean while preserving trace and eigenvector structure.
- **Ward Hierarchical Clustering with Turnover Hysteresis (`select_cluster_representatives`):** Groups stocks into $K = 15$ risk clusters using correlation distance $d_{ij} = \sqrt{2(1 - \rho_{ij})}$. Within each cluster, selects the representative maximizing $\ln(1 + \text{ADTV}) / \sigma^2$, applying a `1.25x`retention multiplier to existing holdings to suppress unnecessary churn.
- **Equal Risk Contribution (`optimize_erc_weights`):** Solves for portfolio weights via `SLSQP` such that each asset contributes equally to total portfolio variance ($\text{RC}_i = w_i (\mathbf{\Sigma}\mathbf{w})_i = \frac{1}{N}\mathbf{w}^T\mathbf{\Sigma}\mathbf{w}$), capped at `15%`per stock, alongside `Inverse-Volatility` and `Equal-Weight` benchmarks.

### Stage 3: Event-Driven HOSE Execution Simulator (`03` / `src/hose_simulator.py`)

Evaluates target weights under realistic Vietnamese market frictions:

- **Microstructure & Settlement:** Enforces **100-share board lots**, **T+2.5 settlement queues** (newly purchased shares remain locked until settlement completes), **0.15%** brokerage fees per side, and **0.10%** securities transfer tax on sales.
- **Liquidity Locks & Drift Buffer:** Blocks sells at floor prices ($\le -6.8\%$), blocks buys at ceiling prices ($\ge +6.8\%$), and ignores weight deviations within a **1.5% no-trade buffer**.
- **Rolling High-Water Mark (HWM) Soft-Landing:** Scales equity exposure linearly from `100%` down to a `20%` floor as rolling 120-session drawdown widens from `5%` (`dd_soft_start`) to `12%` (`dd_hard_stop`).

## Companion Thesis (`paper/`)

Included in `paper/markowitz_multi_asset_thesis.md` is a companion empirical study (*Portfolio Optimization Based on Markowitz Theory: Empirical Analysis of the "First Investment Plan Portfolio" vs. Bank Savings Deposits in Vietnam*, 2019–2024) examining strategic multi-asset allocation across Growth Equities (45%),Bond Fund Certificates (35%), Defensive Gold (10%), and Cash (10%).