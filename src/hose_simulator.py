"""
Event-Driven Execution Simulator for the Ho Chi Minh Stock Exchange (HOSE)[cite: 1].
Models:
- 100-share board lot rounding[cite: 1]
- T+2.5 settlement queue (shares locked until settlement_days elapse)[cite: 1]
- Brokerage fees (0.15% buy/sell) and securities transfer tax (0.10% sell)[cite: 1]
- Ceiling/floor price liquidity locks (+/- 6.8% threshold)[cite: 1]
- No-trade buffer zone (1.5%)[cite: 1]
- Rolling High-Water Mark (HWM) drawdown soft-landing overlay[cite: 1]
"""

import numpy as np
import polars as pl

DEFAULT_EXEC_CONFIG = {
    "initial_capital_vnd": 1_000_000_000,  # 1 billion VND starting capital[cite: 1]
    "buy_fee_rate": 0.0015,                # 0.15% brokerage buy fee[cite: 1]
    "sell_fee_rate": 0.0015,               # 0.15% brokerage sell fee[cite: 1]
    "sell_tax_rate": 0.0010,               # 0.10% personal income tax on sales[cite: 1]
    "lot_size": 100,                       # HOSE board lot size: 100 shares[cite: 1]
    "settlement_days": 2,                  # T+2.5 cycle (bought on t, sellable at t+2)[cite: 1]
    "no_trade_buffer": 0.015,              # 1.5% weight drift tolerance band[cite: 1]
    "limit_band": 0.068,                   # ~6.8% threshold to detect ceiling/floor locks[cite: 1]
    "hwm_window": 120,                     # 120-session rolling peak NAV window[cite: 1]
    "dd_soft_start": 0.05,                 # Start de-risking above 5% drawdown[cite: 1]
    "dd_hard_stop": 0.12,                  # Floor equity exposure at >= 12% drawdown[cite: 1]
    "min_exposure_floor": 0.20,            # Minimum 20% equity exposure under max defense[cite: 1]
}


def compute_hwm_scalar(nav_history: list[float], cfg: dict = DEFAULT_EXEC_CONFIG) -> tuple[float, float]:
    """
    Compute portfolio exposure scaling factor based on Rolling High-Water Mark drawdown[cite: 1].
    Returns (alpha_exposure, current_drawdown)[cite: 1].
    """
    if not nav_history:[cite: 1]
        return 1.0, 0.0[cite: 1]

    window = nav_history[-cfg["hwm_window"] :][cite: 1]
    hwm = max(window)[cite: 1]
    curr_nav = nav_history[-1][cite: 1]
    dd = max(0.0, 1.0 - (curr_nav / hwm))[cite: 1]

    dd_start = cfg["dd_soft_start"][cite: 1]
    dd_stop = cfg["dd_hard_stop"][cite: 1]
    floor_exp = cfg["min_exposure_floor"][cite: 1]

    if dd <= dd_start:[cite: 1]
        return 1.0, dd[cite: 1]
    if dd >= dd_stop:[cite: 1]
        return floor_exp, dd[cite: 1]

    slope = (1.0 - floor_exp) / (dd_stop - dd_start)[cite: 1]
    alpha = 1.0 - slope * (dd - dd_start)[cite: 1]
    return float(alpha), dd[cite: 1]


def _align_market_matrices(
    df_tri: pl.DataFrame, df_weights: pl.DataFrame, limit_band: float
) -> dict:
    """
    Align raw prices, corporate action share multipliers, and ceiling/floor flags
    to the exact simulation date spine[cite: 1].
    """
    meta_cols = {"trade_date", "CASH", "is_rebalance_day"}[cite: 1]
    tickers = [col for col in df_weights.columns if col not in meta_cols][cite: 1]

    df_weights_sorted = df_weights.sort("trade_date")[cite: 1]
    sim_dates = df_weights_sorted["trade_date"].to_list()[cite: 1]
    start_sim_date = sim_dates[0][cite: 1]

    if "is_rebalance_day" in df_weights_sorted.columns:[cite: 1]
        rebal_flags = df_weights_sorted["is_rebalance_day"].fill_null(False).to_numpy().astype(bool)[cite: 1]
    else:
        rebal_flags = np.ones(len(sim_dates), dtype=bool)[cite: 1]

    df_market_aligned = (
        df_tri
        .filter(pl.col("ticker").is_in(tickers))
        .sort(["ticker", "trade_date"])
        .with_columns([
            pl.col("close").shift(1).over("ticker").alias("prev_close"),
            (pl.col("adj_shares_factor") / pl.col("adj_shares_factor").shift(1).over("ticker"))
            .fill_null(1.0)
            .alias("shares_multiplier"),
            ((pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0) >= limit_band)
            .fill_null(False)
            .alias("is_ceiling"),
            ((pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0) <= -limit_band)
            .fill_null(False)
            .alias("is_floor"),
        ])
        .filter(pl.col("trade_date") >= start_sim_date)
    )[cite: 1]

    df_dates_spine = pl.DataFrame({"trade_date": sim_dates})[cite: 1]

    def _pivot_matrix(val_col: str, fill_val=0.0, ffill: bool = False) -> np.ndarray:
        piv = df_market_aligned.pivot(index="trade_date", on="ticker", values=val_col).sort("trade_date")[cite: 1]
        piv = df_dates_spine.join(piv, on="trade_date", how="left").sort("trade_date")[cite: 1]
        for tk in tickers:[cite: 1]
            if tk not in piv.columns:[cite: 1]
                piv = piv.with_columns(pl.lit(None).alias(tk))[cite: 1]
        piv = piv.select(tickers)[cite: 1]
        if ffill:[cite: 1]
            piv = piv.with_columns([pl.col(tk).forward_fill().backward_fill() for tk in tickers])[cite: 1]
        return piv.fill_null(fill_val).to_numpy()[cite: 1]

    return {
        "tickers": tickers,
        "sim_dates": sim_dates,
        "rebal_flags": rebal_flags,
        "close": _pivot_matrix("close", fill_val=0.0, ffill=True),
        "mult": _pivot_matrix("shares_multiplier", fill_val=1.0, ffill=False),
        "is_ceiling": _pivot_matrix("is_ceiling", fill_val=False, ffill=False).astype(bool),
        "is_floor": _pivot_matrix("is_floor", fill_val=False, ffill=False).astype(bool),
        "target_weights": df_weights_sorted.select(tickers).to_numpy(),
    }[cite: 1]


def run_hose_simulation(
    df_tri: pl.DataFrame,
    df_weights: pl.DataFrame,
    cfg: dict = DEFAULT_EXEC_CONFIG,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Execute the event-driven HOSE portfolio backtest[cite: 1].
    Returns (df_ledger, df_trades)[cite: 1].
    """
    aligned = _align_market_matrices(df_tri, df_weights, cfg["limit_band"])[cite: 1]
    tickers = aligned["tickers"][cite: 1]
    sim_dates = aligned["sim_dates"][cite: 1]
    rebal_flags = aligned["rebal_flags"][cite: 1]
    close_mat = aligned["close"][cite: 1]
    mult_mat = aligned["mult"][cite: 1]
    ceil_mat = aligned["is_ceiling"][cite: 1]
    floor_mat = aligned["is_floor"][cite: 1]
    target_w_mat = aligned["target_weights"][cite: 1]

    n_days, n_assets = close_mat.shape[cite: 1]
    cash = float(cfg["initial_capital_vnd"])[cite: 1]

    avail_shares = np.zeros(n_assets, dtype=np.int64)[cite: 1]
    pending_settlement: list[tuple[int, int, int]] = []  # (unlock_day_idx, ticker_idx, qty)[cite: 1]

    nav_history: list[float] = [][cite: 1]
    ledger_records: list[dict] = [][cite: 1]
    trade_records: list[dict] = [][cite: 1]

    lot = cfg["lot_size"][cite: 1]
    buf = cfg["no_trade_buffer"][cite: 1]
    buy_fee = cfg["buy_fee_rate"][cite: 1]
    sell_cost = cfg["sell_fee_rate"] + cfg["sell_tax_rate"][cite: 1]

    for t in range(n_days):[cite: 1]
        d = sim_dates[t][cite: 1]
        prices = close_mat[t][cite: 1]

        # 1. Unlock shares completing T+2.5 settlement[cite: 1]
        still_pending = [][cite: 1]
        for unlock_t, tk_idx, qty in pending_settlement:[cite: 1]
            if t >= unlock_t:[cite: 1]
                avail_shares[tk_idx] += qty[cite: 1]
            else:
                still_pending.append((unlock_t, tk_idx, qty))[cite: 1]
        pending_settlement = still_pending[cite: 1]

        # 2. Apply corporate actions (stock splits/dividends; fractional shares credited as cash)[cite: 1]
        if t > 0:[cite: 1]
            for i in range(n_assets):[cite: 1]
                m = mult_mat[t, i][cite: 1]
                if abs(m - 1.0) > 1e-6 and prices[i] > 0:[cite: 1]
                    new_avail_exact = avail_shares[i] * m[cite: 1]
                    new_avail_int = int(np.floor(new_avail_exact))[cite: 1]
                    cash += (new_avail_exact - new_avail_int) * prices[i][cite: 1]
                    avail_shares[i] = new_avail_int[cite: 1]

            updated_pending = [][cite: 1]
            for unlock_t, tk_idx, qty in pending_settlement:[cite: 1]
                m = mult_mat[t, tk_idx][cite: 1]
                if abs(m - 1.0) > 1e-6 and prices[tk_idx] > 0:[cite: 1]
                    new_q_exact = qty * m[cite: 1]
                    new_q_int = int(np.floor(new_q_exact))[cite: 1]
                    cash += (new_q_exact - new_q_int) * prices[tk_idx][cite: 1]
                    updated_pending.append((unlock_t, tk_idx, new_q_int))[cite: 1]
                else:
                    updated_pending.append((unlock_t, tk_idx, qty))[cite: 1]
            pending_settlement = updated_pending[cite: 1]

        locked_shares = np.zeros(n_assets, dtype=np.int64)[cite: 1]
        for _, tk_idx, qty in pending_settlement:[cite: 1]
            locked_shares[tk_idx] += qty[cite: 1]
        total_shares = avail_shares + locked_shares[cite: 1]

        # 3. Pre-trade NAV valuation & Rolling HWM exposure scalar[cite: 1]
        stock_value = float(np.sum(total_shares * prices))[cite: 1]
        pre_trade_nav = cash + stock_value[cite: 1]

        hwm_alpha, curr_dd = compute_hwm_scalar(nav_history, cfg)[cite: 1]
        adj_target_weights = target_w_mat[t] * hwm_alpha[cite: 1]

        current_weights = np.where(pre_trade_nav > 0, (total_shares * prices) / pre_trade_nav, 0.0)[cite: 1]
        daily_turnover_vnd = 0.0[cite: 1]
        daily_fees_vnd = 0.0[cite: 1]

        prev_alpha = ledger_records[-1]["hwm_alpha"] if ledger_records else 1.0[cite: 1]
        should_trade = bool(rebal_flags[t] or (hwm_alpha < prev_alpha - 1e-4))[cite: 1]

        if should_trade:[cite: 1]
            # 4. Execute SELL orders first (restricted to settled avail_shares & non-floor prices)[cite: 1]
            for i in range(n_assets):[cite: 1]
                if prices[i] <= 0 or floor_mat[t, i]:[cite: 1]
                    continue[cite: 1]

                w_diff = adj_target_weights[i] - current_weights[i][cite: 1]
                if w_diff < -buf or (adj_target_weights[i] == 0.0 and avail_shares[i] > 0):[cite: 1]
                    target_val = adj_target_weights[i] * pre_trade_nav[cite: 1]
                    curr_val = total_shares[i] * prices[i][cite: 1]
                    sell_val_needed = curr_val - target_val[cite: 1]

                    raw_sell_qty = int(np.floor(sell_val_needed / prices[i] / lot)) * lot[cite: 1]
                    sell_qty = min(raw_sell_qty, int(avail_shares[i]))[cite: 1]

                    if sell_qty >= lot:[cite: 1]
                        gross_proceeds = sell_qty * prices[i][cite: 1]
                        fee_and_tax = gross_proceeds * sell_cost[cite: 1]
                        net_proceeds = gross_proceeds - fee_and_tax[cite: 1]

                        avail_shares[i] -= sell_qty[cite: 1]
                        cash += net_proceeds[cite: 1]
                        daily_turnover_vnd += gross_proceeds[cite: 1]
                        daily_fees_vnd += fee_and_tax[cite: 1]

                        trade_records.append({
                            "trade_date": d,
                            "ticker": tickers[i],
                            "side": "SELL",
                            "qty": sell_qty,
                            "price": prices[i],
                            "gross_vnd": gross_proceeds,
                            "fee_tax_vnd": fee_and_tax,
                        })[cite: 1]

            # 5. Execute BUY orders prioritized by largest positive weight deficit[cite: 1]
            buy_candidates = [][cite: 1]
            for i in range(n_assets):[cite: 1]
                if prices[i] <= 0 or ceil_mat[t, i]:[cite: 1]
                    continue[cite: 1]
                w_diff = adj_target_weights[i] - current_weights[i][cite: 1]
                if w_diff > buf:[cite: 1]
                    buy_candidates.append((w_diff, i))[cite: 1]

            buy_candidates.sort(reverse=True)[cite: 1]

            for _, i in buy_candidates:[cite: 1]
                target_val = adj_target_weights[i] * pre_trade_nav[cite: 1]
                curr_locked = sum(q for _, idx, q in pending_settlement if idx == i)[cite: 1]
                curr_val = (avail_shares[i] + curr_locked) * prices[i][cite: 1]
                buy_val_needed = target_val - curr_val[cite: 1]

                if buy_val_needed <= 0:[cite: 1]
                    continue[cite: 1]

                max_affordable_val = cash / (1.0 + buy_fee)[cite: 1]
                actual_buy_val = min(buy_val_needed, max_affordable_val)[cite: 1]
                buy_qty = int(np.floor(actual_buy_val / prices[i] / lot)) * lot[cite: 1]

                if buy_qty >= lot:[cite: 1]
                    gross_cost = buy_qty * prices[i][cite: 1]
                    fee = gross_cost * buy_fee[cite: 1]
                    total_outflow = gross_cost + fee[cite: 1]

                    if total_outflow <= cash:[cite: 1]
                        cash -= total_outflow[cite: 1]
                        pending_settlement.append((t + cfg["settlement_days"], i, buy_qty))[cite: 1]
                        daily_turnover_vnd += gross_cost[cite: 1]
                        daily_fees_vnd += fee[cite: 1]

                        trade_records.append({
                            "trade_date": d,
                            "ticker": tickers[i],
                            "side": "BUY",
                            "qty": buy_qty,
                            "price": prices[i],
                            "gross_vnd": gross_cost,
                            "fee_tax_vnd": fee,
                        })[cite: 1]

        # 6. End-of-day NAV accounting[cite: 1]
        locked_after = np.zeros(n_assets, dtype=np.int64)[cite: 1]
        for _, tk_idx, qty in pending_settlement:[cite: 1]
            locked_after[tk_idx] += qty[cite: 1]
        end_stock_val = float(np.sum((avail_shares + locked_after) * prices))[cite: 1]
        end_nav = cash + end_stock_val[cite: 1]

        nav_history.append(end_nav)[cite: 1]
        prev_eod_nav = nav_history[-2] if len(nav_history) >= 2 else float(cfg["initial_capital_vnd"])[cite: 1]
        net_daily_ret = (end_nav / prev_eod_nav) - 1.0[cite: 1]

        ledger_records.append({
            "trade_date": d,
            "nav_vnd": end_nav,
            "cash_vnd": cash,
            "stock_value_vnd": end_stock_val,
            "actual_stock_weight": end_stock_val / end_nav if end_nav > 0 else 0.0,
            "target_stock_weight": float(np.sum(adj_target_weights)),
            "hwm_alpha": hwm_alpha,
            "drawdown": curr_dd,
            "net_daily_return": net_daily_ret,
            "turnover_vnd": daily_turnover_vnd,
            "fees_tax_vnd": daily_fees_vnd,
        })[cite: 1]

    df_ledger = pl.DataFrame(ledger_records)[cite: 1]
    df_trades = pl.DataFrame(trade_records) if trade_records else pl.DataFrame()[cite: 1]
    return df_ledger, df_trades[cite: 1]