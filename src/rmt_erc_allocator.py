"""
Random Matrix Theory (Marchenko-Pastur) Correlation Denoising, Hierarchical Ward Clustering,
and Equal Risk Contribution (ERC) Portfolio Optimization[cite: 3].
"""

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

DEFAULT_RISK_CONFIG = {
    "lookback_window": 120,          # 120-session rolling window (~6 months)[cite: 3]
    "adtv_window": 20,               # 20-session average daily trading value window[cite: 3]
    "min_adtv_vnd": 10_000_000_000,  # Minimum liquidity: 10 billion VND/session[cite: 3]
    "rebalance_freq_days": 10,       # Bi-weekly rebalance (every 10 trading days)[cite: 3]
    "n_clusters": 15,                # Target number of risk clusters[cite: 3]
    "max_single_weight": 0.15,       # Maximum weight cap per stock (15%)[cite: 3]
    "min_assets_required": 5,        # Minimum eligible stocks to stay invested[cite: 3]
    "hysteresis_bonus": 1.25,        # Retention multiplier for existing portfolio constituents[cite: 3]
}


def prepare_universe_matrices(df_tri: pl.DataFrame, cfg: dict = DEFAULT_RISK_CONFIG) -> dict:
    """
    Compute rolling liquidity/volatility metrics and pivot into aligned wide matrices[cite: 3].
    """
    df_metrics = (
        df_tri
        .sort(["ticker", "trade_date"])
        .with_columns([
            pl.col("value_match").rolling_mean(cfg["adtv_window"]).over("ticker").alias("adtv_20"),
            pl.col("total_return").rolling_std(cfg["lookback_window"]).over("ticker").alias("roll_vol"),
            pl.cum_count("trade_date").over("ticker").alias("history_len"),
        ])
        .with_columns(
            (
                (pl.col("adtv_20") >= cfg["min_adtv_vnd"])
                & (pl.col("history_len") >= cfg["lookback_window"])
                & (pl.col("volume_match") > 0)
                & (pl.col("roll_vol") > 1e-6)
            ).alias("is_eligible")
        )
    )[cite: 3]

    def _pivot_wide(col: str) -> pl.DataFrame:
        return df_metrics.pivot(index="trade_date", on="ticker", values=col).sort("trade_date")[cite: 3]

    returns_wide = _pivot_wide("total_return")[cite: 3]
    eligible_wide = _pivot_wide("is_eligible").fill_null(False)[cite: 3]
    adtv_wide = _pivot_wide("adtv_20").fill_null(0.0)[cite: 3]
    vol_wide = _pivot_wide("roll_vol").fill_null(np.inf)[cite: 3]

    tickers = [c for c in returns_wide.columns if c != "trade_date"][cite: 3]
    dates = returns_wide["trade_date"].to_list()[cite: 3]

    return {
        "tickers": tickers,
        "dates": dates,
        "returns": returns_wide.select(tickers).to_numpy(),
        "eligible": eligible_wide.select(tickers).to_numpy(),
        "adtv": adtv_wide.select(tickers).to_numpy(),
        "volatility": vol_wide.select(tickers).to_numpy(),
    }[cite: 3]


def denoise_correlation_rmt(returns_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Denoise sample correlation matrix via Marchenko-Pastur eigenvalue clipping[cite: 3].
    Returns (corr_clean, stds).
    """
    T, N = returns_matrix.shape[cite: 3]
    stds = np.std(returns_matrix, axis=0, ddof=1)[cite: 3]
    stds = np.where(stds < 1e-8, 1e-8, stds)[cite: 3]
    if N == 1:[cite: 3]
        return np.ones((1, 1)), stds[cite: 3]

    corr_matrix = np.corrcoef(returns_matrix, rowvar=False)[cite: 3]
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)[cite: 3]
    np.fill_diagonal(corr_matrix, 1.0)[cite: 3]

    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)[cite: 3]
    idx = eigenvalues.argsort()[::-1][cite: 3]
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx][cite: 3]

    q = N / float(T)[cite: 3]
    lambda_plus = (1.0 + np.sqrt(q)) ** 2[cite: 3]

    denoised_evals = eigenvalues.copy()[cite: 3]
    noise_mask = denoised_evals <= lambda_plus[cite: 3]
    if np.any(noise_mask) and not np.all(noise_mask):[cite: 3]
        denoised_evals[noise_mask] = np.mean(denoised_evals[noise_mask])[cite: 3]

    corr_clean = eigenvectors @ np.diag(denoised_evals) @ eigenvectors.T[cite: 3]
    diag_sqrt = np.sqrt(np.diag(corr_clean))[cite: 3]
    corr_clean = corr_clean / np.outer(diag_sqrt, diag_sqrt)[cite: 3]
    corr_clean = np.clip(corr_clean, -1.0, 1.0)[cite: 3]
    np.fill_diagonal(corr_clean, 1.0)[cite: 3]
    return corr_clean, stds[cite: 3]


def select_cluster_representatives(
    corr_clean: np.ndarray,
    stds: np.ndarray,
    adtvs: np.ndarray,
    n_clusters: int,
    valid_global_idx: np.ndarray,
    prev_selected_global: set[int],
    hysteresis_bonus: float = 1.25,
) -> np.ndarray:
    """
    Perform Ward hierarchical clustering on RMT correlation distance and pick one
    representative per cluster using liquidity-to-variance score with turnover hysteresis[cite: 3].
    """
    N = corr_clean.shape[0][cite: 3]
    if N <= n_clusters:[cite: 3]
        return np.arange(N)[cite: 3]

    dist_mat = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - corr_clean)))[cite: 3]
    np.fill_diagonal(dist_mat, 0.0)[cite: 3]
    condensed_dist = squareform(dist_mat, checks=False)[cite: 3]

    Z = linkage(condensed_dist, method="ward")[cite: 3]
    cluster_ids = fcluster(Z, t=n_clusters, criterion="maxclust")[cite: 3]

    base_metric = np.log1p(adtvs) / (stds ** 2 + 1e-8)[cite: 3]
    adjusted_metric = base_metric.copy()[cite: 3]
    for local_i, global_i in enumerate(valid_global_idx):[cite: 3]
        if global_i in prev_selected_global:[cite: 3]
            adjusted_metric[local_i] *= hysteresis_bonus[cite: 3]

    selected_sub_idx = [][cite: 3]
    for c_id in np.unique(cluster_ids):[cite: 3]
        members = np.where(cluster_ids == c_id)[0][cite: 3]
        best_member = members[np.argmax(adjusted_metric[members])][cite: 3]
        selected_sub_idx.append(best_member)[cite: 3]

    return np.sort(np.array(selected_sub_idx, dtype=int))[cite: 3]


def optimize_erc_weights(cov_matrix: np.ndarray, max_weight: float = 0.15) -> np.ndarray:
    """
    Solve for Equal Risk Contribution (ERC) weights via SLSQP subject to box constraints[cite: 3].
    """
    N = cov_matrix.shape[0][cite: 3]
    if N == 0:[cite: 3]
        return np.array([])[cite: 3]

    target_invested = min(1.0, N * max_weight)[cite: 3]
    init_weights = np.repeat(target_invested / N, N)[cite: 3]
    bounds = tuple((0.0, max_weight) for _ in range(N))[cite: 3]

    def risk_budget_objective(w: np.ndarray) -> float:
        port_var = w.T @ cov_matrix @ w[cite: 3]
        if port_var <= 1e-12:[cite: 3]
            return 0.0[cite: 3]
        risk_contrib = w * (cov_matrix @ w)[cite: 3]
        target_rc = port_var / N[cite: 3]
        return float(np.sum((risk_contrib - target_rc) ** 2) * 1e6)[cite: 3]

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - target_invested},)[cite: 3]
    res = minimize(
        risk_budget_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )[cite: 3]
    weights = res.x if res.success else init_weights[cite: 3]
    weights = np.where(weights < 0.001, 0.0, weights)[cite: 3]
    if np.sum(weights) > 0:[cite: 3]
        weights = weights * (target_invested / np.sum(weights))[cite: 3]
    return weights[cite: 3]


def cap_and_normalize_weights(
    raw_weights: np.ndarray, target_invested: float, max_weight: float
) -> np.ndarray:
    """
    Iteratively clip weights at max_weight and redistribute excess proportionally[cite: 3].
    """
    w = np.maximum(0.0, raw_weights.astype(float))[cite: 3]
    if np.sum(w) <= 0:[cite: 3]
        return np.zeros_like(w)[cite: 3]

    w = w * (target_invested / np.sum(w))[cite: 3]
    for _ in range(len(w)):[cite: 3]
        over_mask = w > max_weight + 1e-12[cite: 3]
        if not np.any(over_mask):[cite: 3]
            break[cite: 3]
        under_mask = w < max_weight - 1e-12[cite: 3]
        if not np.any(under_mask):[cite: 3]
            w = np.minimum(w, max_weight)[cite: 3]
            break[cite: 3]
        excess = np.sum(w[over_mask] - max_weight)[cite: 3]
        w[over_mask] = max_weight[cite: 3]
        w[under_mask] += excess * (w[under_mask] / np.sum(w[under_mask]))[cite: 3]
    return w[cite: 3]


def compute_benchmark_weights(
    stds: np.ndarray, max_weight: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute benchmark allocations on the same cluster-selected universe:
    returns (inverse_volatility_weights, equal_weights)[cite: 3].
    """
    N = len(stds)[cite: 3]
    target_invested = min(1.0, N * max_weight)[cite: 3]

    w_eq = np.repeat(target_invested / N, N)[cite: 3]
    inv_v = 1.0 / np.maximum(stds, 1e-6)[cite: 3]
    w_iv = cap_and_normalize_weights(inv_v, target_invested, max_weight)[cite: 3]

    return w_iv, w_eq[cite: 3]


def generate_walk_forward_weights(
    df_tri: pl.DataFrame, cfg: dict = DEFAULT_RISK_CONFIG
) -> dict[str, pl.DataFrame]:
    """
    Run point-in-time walk-forward allocation (strictly t-1 information cutoff)
    and return target weight DataFrames for 'erc', 'inv_vol', and 'equal'[cite: 3].
    """
    mats = prepare_universe_matrices(df_tri, cfg)[cite: 3]
    tickers_all = mats["tickers"][cite: 3]
    dates_all = mats["dates"][cite: 3]
    ret_mat = mats["returns"][cite: 3]
    elig_mat = mats["eligible"][cite: 3]
    adtv_mat = mats["adtv"][cite: 3]

    W = cfg["lookback_window"][cite: 3]
    rebal_step = cfg["rebalance_freq_days"][cite: 3]
    min_assets = cfg["min_assets_required"][cite: 3]
    max_w = cfg["max_single_weight"][cite: 3]
    n_clust = cfg["n_clusters"][cite: 3]
    hyst_bonus = cfg.get("hysteresis_bonus", 1.25)[cite: 3]

    rec_erc, rec_ivol, rec_eq = [], [], [][cite: 3]
    curr_erc = np.zeros(len(tickers_all))[cite: 3]
    curr_ivol = np.zeros(len(tickers_all))[cite: 3]
    curr_eq = np.zeros(len(tickers_all))[cite: 3]
    prev_selected_global: set[int] = set()[cite: 3]

    for t in range(W, len(dates_all)):[cite: 3]
        d = dates_all[t][cite: 3]
        is_rebal = (t - W) % rebal_step == 0[cite: 3]

        if is_rebal:[cite: 3]
            valid_idx = np.where(elig_mat[t - 1])[0][cite: 3]
            if len(valid_idx) >= min_assets:[cite: 3]
                win_rets = np.nan_to_num(ret_mat[t - W : t, :][:, valid_idx], nan=0.0)[cite: 3]
                win_adtv = adtv_mat[t - 1, valid_idx][cite: 3]

                corr_full, stds_full = denoise_correlation_rmt(win_rets)[cite: 3]
                rep_sub_idx = select_cluster_representatives(
                    corr_full,
                    stds_full,
                    win_adtv,
                    n_clust,
                    valid_global_idx=valid_idx,
                    prev_selected_global=prev_selected_global,
                    hysteresis_bonus=hyst_bonus,
                )[cite: 3]
                chosen_global_idx = valid_idx[rep_sub_idx][cite: 3]
                prev_selected_global = set(chosen_global_idx.tolist())[cite: 3]

                corr_sub = corr_full[np.ix_(rep_sub_idx, rep_sub_idx)][cite: 3]
                stds_sub = stds_full[rep_sub_idx][cite: 3]
                cov_sub = np.outer(stds_sub, stds_sub) * corr_sub[cite: 3]

                w_erc_sub = optimize_erc_weights(cov_sub, max_weight=max_w)[cite: 3]
                w_iv_sub, w_eq_sub = compute_benchmark_weights(stds_sub, max_weight=max_w)[cite: 3]

                curr_erc = np.zeros(len(tickers_all))[cite: 3]
                curr_erc[chosen_global_idx] = w_erc_sub[cite: 3]

                curr_ivol = np.zeros(len(tickers_all))[cite: 3]
                curr_ivol[chosen_global_idx] = w_iv_sub[cite: 3]

                curr_eq = np.zeros(len(tickers_all))[cite: 3]
                curr_eq[chosen_global_idx] = w_eq_sub[cite: 3]
            else:
                curr_erc = np.zeros(len(tickers_all))[cite: 3]
                curr_ivol = np.zeros(len(tickers_all))[cite: 3]
                curr_eq = np.zeros(len(tickers_all))[cite: 3]
                prev_selected_global = set()[cite: 3]

        for rec_list, w_vec in [(rec_erc, curr_erc), (rec_ivol, curr_ivol), (rec_eq, curr_eq)]:[cite: 3]
            row = {"trade_date": d, "is_rebalance_day": is_rebal}[cite: 3]
            for idx, tk in enumerate(tickers_all):[cite: 3]
                row[tk] = w_vec[idx][cite: 3]
            row["CASH"] = max(0.0, 1.0 - float(np.sum(w_vec)))[cite: 3]
            rec_list.append(row)[cite: 3]

    return {
        "erc": pl.DataFrame(rec_erc),
        "inv_vol": pl.DataFrame(rec_ivol),
        "equal": pl.DataFrame(rec_eq),
    }[cite: 3]