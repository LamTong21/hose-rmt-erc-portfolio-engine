"""
Data Ingestion, Integrity Auditing, and Forward-Compounded Total Return Index (TRI) Engine[cite: 2].
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf

HOSE_MARKET_RULES = {
    "div_cash_tax_rate": 0.05,       # 5% personal income tax on cash dividends[cite: 2]
    "default_cash_lag_days": 30,     # Default lag from Ex-date to cash settlement[cite: 2]
    "price_band_limit": 0.07,        # +/- 7% daily price limit on HOSE[cite: 2]
    "lot_size": 100,                 # Standard board lot size[cite: 2]
    "start_date": "2018-01-01",[cite: 2]
    "end_date": "2026-11-23",[cite: 2]
}

FALLBACK_VN100 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
    "ANV", "BCG", "BMP", "BSI", "BWE", "CII", "CMG", "CTR", "DBC", "DCM",
    "DGC", "DGW", "DIG", "DPM", "DXG", "DXS", "EIB", "EVF", "FRT", "FTS",
    "GEX", "GMD", "HAG", "HCM", "HDC", "HDG", "HHV", "HSG", "HT1", "IMP",
    "KBC", "KDC", "KDH", "KOS", "LPB", "MSB", "NAB", "NKG", "NLG", "NT2",
    "NVL", "OCB", "ORS", "PAN", "PC1", "PDR", "PHR", "PNJ", "PPC", "PTB",
    "PVD", "PVT", "REE", "SBT", "SCS", "SJS", "SZC", "TCH", "TLG", "VCI",
    "VGC", "VHC", "VIX", "VND", "VOS", "VPI", "VTP", "ASM", "CTD", "HAH",
][cite: 2]

ACTIONS_SCHEMA = {
    "ticker": pl.String,
    "ex_date": pl.Date,
    "payment_date": pl.Date,
    "action_type": pl.String,
    "cash_per_share": pl.Float64,
    "share_ratio": pl.Float64,
    "issue_price": pl.Float64,
}[cite: 2]


def fetch_vn100_yfinance(
    start_date: str = HOSE_MARKET_RULES["start_date"],
    end_date: str = HOSE_MARKET_RULES["end_date"],
    tickers: Optional[list[str]] = None,
    cache_dir: Optional[Path] = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Fetch historical unadjusted OHLCV and corporate actions (dividends & splits)
    for the VN100 universe via Yahoo Finance[cite: 2].
    """
    symbols = sorted(set(tickers or FALLBACK_VN100))[cite: 2]
    vn_tickers = [f"{s}.VN" for s in symbols][cite: 2]

    raw_pd = yf.download(
        tickers=vn_tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,  # Keep raw Close separate from Adj Close[cite: 2]
        actions=True,
        threads=True,
        progress=False,
    )[cite: 2]

    market_dfs: list[pl.DataFrame] = [][cite: 2]
    action_dfs: list[pl.DataFrame] = [][cite: 2]

    for sym, yf_sym in zip(symbols, vn_tickers):[cite: 2]
        try:
            if isinstance(raw_pd.columns, pd.MultiIndex):[cite: 2]
                if yf_sym not in raw_pd.columns.get_level_values(0):[cite: 2]
                    continue
                sub = raw_pd[yf_sym].copy()[cite: 2]
            else:
                sub = raw_pd.copy()[cite: 2]

            sub = sub.reset_index()[cite: 2]
            date_col = "Date" if "Date" in sub.columns else sub.columns[0][cite: 2]
            sub["trade_date"] = pd.to_datetime(sub[date_col]).dt.date[cite: 2]
            sub = sub.dropna(subset=["Close"])[cite: 2]

            if sub.empty:[cite: 2]
                continue

            df_ohlc = (
                pl.DataFrame({
                    "ticker": [sym] * len(sub),
                    "trade_date": sub["trade_date"].tolist(),
                    "open": sub["Open"].astype(float).tolist(),
                    "high": sub["High"].astype(float).tolist(),
                    "low": sub["Low"].astype(float).tolist(),
                    "close": sub["Close"].astype(float).tolist(),
                    "volume_match": sub["Volume"].fillna(0).astype("int64").tolist(),
                })
                .with_columns(
                    (pl.col("close") * pl.col("volume_match")).alias("value_match")
                )
            )[cite: 2]
            market_dfs.append(df_ohlc)[cite: 2]

            if "Dividends" in sub.columns:[cite: 2]
                divs = sub[sub["Dividends"].fillna(0) > 0][cite: 2]
                if not divs.empty:[cite: 2]
                    action_dfs.append(
                        pl.DataFrame({
                            "ticker": [sym] * len(divs),
                            "ex_date": divs["trade_date"].tolist(),
                            "payment_date": [None] * len(divs),
                            "action_type": ["CASH_DIVIDEND"] * len(divs),
                            "cash_per_share": divs["Dividends"].astype(float).tolist(),
                            "share_ratio": [0.0] * len(divs),
                            "issue_price": [0.0] * len(divs),
                        }, schema=ACTIONS_SCHEMA)
                    )[cite: 2]

            if "Stock Splits" in sub.columns:[cite: 2]
                splits = sub[sub["Stock Splits"].fillna(0) > 0][cite: 2]
                if not splits.empty:[cite: 2]
                    action_dfs.append(
                        pl.DataFrame({
                            "ticker": [sym] * len(splits),
                            "ex_date": splits["trade_date"].tolist(),
                            "payment_date": [None] * len(splits),
                            "action_type": ["STOCK_DIVIDEND_OR_SPLIT"] * len(splits),
                            "cash_per_share": [0.0] * len(splits),
                            "share_ratio": (splits["Stock Splits"].astype(float) - 1.0).tolist(),
                            "issue_price": [0.0] * len(splits),
                        }, schema=ACTIONS_SCHEMA)
                    )[cite: 2]
        except Exception:[cite: 2]
            continue

    df_market_all = (
        pl.concat(market_dfs).unique(subset=["ticker", "trade_date"]).sort(["ticker", "trade_date"])
        if market_dfs else pl.DataFrame()
    )[cite: 2]
    df_actions_all = (
        pl.concat(action_dfs).unique().sort(["ticker", "ex_date"])
        if action_dfs else pl.DataFrame(schema=ACTIONS_SCHEMA)
    )[cite: 2]

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)[cite: 2]
        if not df_market_all.is_empty():[cite: 2]
            df_market_all.write_parquet(cache_dir / "vn100_market_raw_cache.parquet", compression="zstd")[cite: 2]
        if not df_actions_all.is_empty():[cite: 2]
            df_actions_all.write_parquet(cache_dir / "vn100_actions_raw_cache.parquet", compression="zstd")[cite: 2]

    return df_market_all, df_actions_all[cite: 2]


def audit_market_data(df_market: pl.DataFrame, df_actions: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """
    Audit raw OHLC integrity and detect unexplained price jumps exceeding HOSE +/- 7.2% bounds[cite: 2].
    """
    df_check = (
        df_market
        .sort(["ticker", "trade_date"])
        .with_columns([
            pl.col("close").shift(1).over("ticker").alias("prev_close"),
            (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("raw_pct_change"),
        ])
        .join(
            df_actions.group_by(["ticker", "ex_date"]).agg(pl.len().alias("has_corp_action")),
            left_on=["ticker", "trade_date"],
            right_on=["ticker", "ex_date"],
            how="left",
        )
        .with_columns(pl.col("has_corp_action").fill_null(0) > 0)
    )[cite: 2]

    invalid_ohlc = df_check.filter(
        (pl.col("close") <= 0) | (pl.col("high") < pl.col("low")) | (pl.col("volume_match") < 0)
    )[cite: 2]

    unexplained_jumps = df_check.filter(
        (pl.col("raw_pct_change").abs() > 0.072) & (~pl.col("has_corp_action"))
    )[cite: 2]

    return {
        "audited_data": df_check,
        "invalid_ohlc": invalid_ohlc,
        "unexplained_jumps": unexplained_jumps,
    }


def compute_realistic_tri(
    df_market: pl.DataFrame,
    df_actions: pl.DataFrame,
    tax_rate: float = HOSE_MARKET_RULES["div_cash_tax_rate"],
    default_lag_days: int = HOSE_MARKET_RULES["default_cash_lag_days"],
) -> pl.DataFrame:
    """
    Compute Forward-Compounded Total Return Index (TRI, base=100.0) modeling:
    - Net cash dividend receivables settled at payment_date (T+lag)[cite: 2]
    - Stock dividends, splits, and preferential rights issue dilution[cite: 2]
    """
    actions = df_actions.with_columns(
        pl.when(pl.col("payment_date").is_null())
        .then(pl.col("ex_date") + pl.duration(days=default_lag_days))
        .otherwise(pl.col("payment_date"))
        .alias("payment_date")
    )[cite: 2]

    results = [][cite: 2]
    for ticker, df_t in df_market.sort("trade_date").group_by("ticker", maintain_order=True):[cite: 2]
        t_name = ticker[0][cite: 2]
        act_t = actions.filter(pl.col("ticker") == t_name)[cite: 2]

        dates = df_t["trade_date"].to_list()[cite: 2]
        closes = df_t["close"].to_numpy()[cite: 2]
        n = len(dates)[cite: 2]

        shares = np.ones(n, dtype=np.float64)[cite: 2]
        pending_cash: dict = {}[cite: 2]
        tri_values = np.zeros(n, dtype=np.float64)[cite: 2]
        daily_total_returns = np.zeros(n, dtype=np.float64)[cite: 2]

        ex_map: dict = {}[cite: 2]
        for row in act_t.iter_rows(named=True):[cite: 2]
            ex_map.setdefault(row["ex_date"], []).append(row)[cite: 2]

        current_shares = 1.0[cite: 2]
        prev_nav = closes[0] * current_shares[cite: 2]
        tri_values[0] = 100.0[cite: 2]

        for i in range(n):[cite: 2]
            d = dates[i][cite: 2]
            p_close = closes[i][cite: 2]

            # 1. Process corporate actions on Ex-date[cite: 2]
            if d in ex_map:[cite: 2]
                for ev in ex_map[d]:[cite: 2]
                    if ev["cash_per_share"] > 0:[cite: 2]
                        net_cash = current_shares * ev["cash_per_share"] * (1.0 - tax_rate)[cite: 2]
                        pay_d = ev["payment_date"][cite: 2]
                        pending_cash[pay_d] = pending_cash.get(pay_d, 0.0) + net_cash[cite: 2]

                    if ev["share_ratio"] > 0 and ev["issue_price"] == 0:[cite: 2]
                        current_shares *= (1.0 + ev["share_ratio"])[cite: 2]

                    if ev["share_ratio"] > 0 and ev["issue_price"] > 0 and p_close > ev["issue_price"]:[cite: 2]
                        dilution_factor = (
                            p_close + ev["share_ratio"] * ev["issue_price"]
                        ) / (1.0 + ev["share_ratio"])[cite: 2]
                        current_shares *= (p_close / dilution_factor)[cite: 2]

            # 2. Settle matured cash dividends and reinvest at closing price[cite: 2]
            cash_received_today = 0.0[cite: 2]
            matured_dates = [pd for pd in pending_cash.keys() if pd <= d][cite: 2]
            for pd in matured_dates:[cite: 2]
                cash_received_today += pending_cash.pop(pd)[cite: 2]

            if cash_received_today > 0 and p_close > 0:[cite: 2]
                current_shares += cash_received_today / p_close[cite: 2]

            shares[i] = current_shares[cite: 2]

            # 3. Compute total NAV (equity + dividend receivables)[cite: 2]
            receivable_cash = sum(pending_cash.values())[cite: 2]
            current_nav = current_shares * p_close + receivable_cash[cite: 2]

            if i > 0 and prev_nav > 0:[cite: 2]
                r_tot = (current_nav / prev_nav) - 1.0[cite: 2]
                daily_total_returns[i] = r_tot[cite: 2]
                tri_values[i] = tri_values[i - 1] * (1.0 + r_tot)[cite: 2]

            prev_nav = current_nav[cite: 2]

        df_res = df_t.with_columns([
            pl.Series("adj_shares_factor", shares),
            pl.Series("total_return", daily_total_returns),
            pl.Series("tri", tri_values),
            (pl.col("close") / pl.col("close").shift(1) - 1.0).fill_null(0.0).alias("price_return"),
        ])[cite: 2]
        results.append(df_res)[cite: 2]

    return pl.concat(results)[cite: 2]