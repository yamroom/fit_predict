# -*- coding: utf-8 -*-
"""
user_portfolio_simulator_numba.py
---------------------------------
（說明略，同上一單元格）
"""

import os, sys, math, time, argparse
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit, prange

os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
plt.rcParams["font.family"] = "Noto Serif CJK JP"

class ERR:
    OK=0; FILE_READ_FAIL=1001; DATA_FORMAT_ERROR=1002; PARAM_ESTIMATION_FAIL=1101
    SIM_FAIL=1301; PLOT_FAIL=1701

def status(stage: str, code: int, msg: str = ""):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    tag = "OK" if code == 0 else f"ERR{code}"
    print(f"[{ts}] [{stage}] [{tag}] {msg}")
    sys.stdout.flush()

@dataclass
class MarketParams:
    mu_m: np.ndarray
    L_m: np.ndarray
    noncash_cols: List[str]
    all_cols: List[str]

def detect_datetime_col(df: pd.DataFrame):
    for c in df.columns:
        p = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
        if p.notna().mean() > 0.9:
            return c, pd.DatetimeIndex(p)
    c = df.columns[0]
    p = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
    if p.notna().mean() > 0.9:
        return c, pd.DatetimeIndex(p)
    return None, None

def pick_col(df: pd.DataFrame, name_candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for want in name_candidates:
        for key, col in low.items():
            if want.lower() == key or want.lower() in key:
                return col
    return None

def load_and_estimate(csv_path: str, cash_return_monthly: float = 0.0) -> MarketParams:
    df = pd.read_csv(csv_path)
    date_col, date_parsed = detect_datetime_col(df)
    if date_parsed is not None:
        prices = df.set_index(date_parsed).drop(columns=[date_col], errors="ignore")
    else:
        prices = df.copy()
        prices.index = pd.RangeIndex(len(prices))

    for c in prices.columns:
        if not pd.api.types.is_numeric_dtype(prices[c]):
            prices[c] = pd.to_numeric(prices[c], errors="coerce")
    prices = prices.sort_index().dropna(how="all")

    cSPY  = pick_col(prices, ["SPY"])
    cQQQ  = pick_col(prices, ["QQQ"])
    cGLD  = pick_col(prices, ["GLD"])
    c0050 = pick_col(prices, ["0050.TW", "0050"])

    use = [c for c in [cSPY, cQQQ, cGLD, c0050] if c is not None]
    if len(use) == 0:
        raise ValueError("No usable asset columns")

    px = prices[use].copy()
    rename = {}
    if cSPY:  rename[cSPY]  = "SPY"
    if cQQQ:  rename[cQQQ]  = "QQQ"
    if cGLD:  rename[cGLD]  = "GLD"
    if c0050: rename[c0050] = "0050"
    px = px.rename(columns=rename).dropna(how="any")

    px_m  = px.resample("M").last()
    ret_m_noncash = px_m.pct_change().dropna()
    ret_m = ret_m_noncash.copy()
    ret_m["Cash"] = cash_return_monthly

    noncash_cols = [c for c in ret_m.columns if c != "Cash"]
    all_cols     = noncash_cols + ["Cash"]

    mu_m = ret_m[noncash_cols].mean(axis=0).values.astype(np.float64)
    Sigma_m = np.cov(ret_m[noncash_cols].T, ddof=0).astype(np.float64)
    eps = 1e-10
    try:
        L_m = np.linalg.cholesky(Sigma_m)
    except np.linalg.LinAlgError:
        L_m = np.linalg.cholesky(Sigma_m + eps*np.eye(Sigma_m.shape[0]))

    return MarketParams(mu_m=mu_m, L_m=L_m, noncash_cols=noncash_cols, all_cols=all_cols)

@njit(parallel=True, fastmath=True, cache=True)
def twr_numba(Wt_full, R_nc, idx_nc, idx_cash, abs_band, rel_band):
    n_sims, M, d_nc = R_nc.shape
    YEARS = M / 12.0
    d_total = Wt_full.shape[0]

    cagr = np.empty(n_sims, dtype=np.float64)
    mdd  = np.empty(n_sims, dtype=np.float64)

    big_mask = np.zeros(d_total, dtype=np.uint8)
    for j in range(d_total):
        if Wt_full[j] >= 0.20:
            big_mask[j] = 1

    for s in prange(n_sims):
        w_full = np.zeros(d_total, dtype=np.float64)
        for j in range(d_total):
            w_full[j] = Wt_full[j]

        nav = 1.0
        peak = 1.0
        min_dd = 0.0

        for t in range(M):
            r_p = 0.0
            for jnc in range(d_nc):
                j_full = idx_nc[jnc]
                r_p += w_full[j_full] * R_nc[s, t, jnc]

            nav *= (1.0 + r_p)
            if nav > peak:
                peak = nav
            dd = nav / peak - 1.0
            if dd < min_dd:
                min_dd = dd

            denom = (1.0 + r_p)
            for jnc in range(d_nc):
                j_full = idx_nc[jnc]
                w_full[j_full] = w_full[j_full] * (1.0 + R_nc[s, t, jnc]) / denom
            w_full[idx_cash] = w_full[idx_cash] / denom

            if ((t+1) % 12) == 0:
                breach = False
                for j in range(d_total):
                    if big_mask[j] == 1:
                        if np.abs(w_full[j] - Wt_full[j]) >= abs_band:
                            breach = True
                            break
                if not breach:
                    for j in range(d_total):
                        if big_mask[j] == 0:
                            lo = Wt_full[j] * (1.0 - rel_band)
                            hi = Wt_full[j] * (1.0 + rel_band)
                            val = w_full[j]
                            if (val <= lo) or (val >= hi):
                                breach = True
                                break
                if breach:
                    for j in range(d_total):
                        w_full[j] = Wt_full[j]

        cagr[s] = nav**(1.0/YEARS) - 1.0
        mdd[s]  = min_dd

    return cagr, mdd

def simulate_monthly_returns(n_sims: int, months: int, mu: np.ndarray, L: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_sims, months, len(mu)))
    return Z @ L.T + mu

def batch_worker(args):
    (batch_idx, W_full, months, sims_this, seed, mu, L, idx_nc, idx_cash, abs_band, rel_band) = args
    R_nc = simulate_monthly_returns(sims_this, months, mu, L, seed).astype(np.float64)
    cagr, mdd = twr_numba(W_full.astype(np.float64), R_nc, idx_nc, idx_cash, abs_band, rel_band)
    return batch_idx, cagr, mdd

def simulate_config_distribution(W_full: np.ndarray,
                                 mu: np.ndarray, L: np.ndarray, all_cols: List[str], noncash_cols: List[str],
                                 months: int, n_sims_total: int, batch_size: int, seed_base: int,
                                 abs_band: float, rel_band: float, n_jobs: int):
    idx_nc  = np.array([all_cols.index(c) for c in noncash_cols], dtype=np.int64)
    idx_cash= np.int64(all_cols.index("Cash"))

    n_batches = math.ceil(n_sims_total / batch_size)
    args = []
    for b in range(n_batches):
        sims_this = batch_size if (b < n_batches-1) else (n_sims_total - batch_size*(n_batches-1))
        args.append((b, W_full, months, sims_this, seed_base + b, mu, L, idx_nc, idx_cash, abs_band, rel_band))

    all_cagr = []
    all_mdd  = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(batch_worker, a): a[0] for a in args}
        total = len(futs)
        for j, fut in enumerate(as_completed(futs), 1):
            b_idx = futs[fut]
            _, cagr, mdd = fut.result()
            all_cagr.append(cagr); all_mdd.append(mdd)
            left = total - j
            status("SIM", ERR.OK, f"批次 {j}/{total} 完成；剩 {left}")
    cagr = np.concatenate(all_cagr); mdd = np.concatenate(all_mdd)
    return cagr, mdd

def plot_histogram(arr_dict: Dict[str, np.ndarray], title: str, xlabel: str, out_png_prefix: str,
                   overlay: bool, save_individual: bool, bins: int = 60):
    if overlay:
        plt.figure(figsize=(9.0, 5.4))
        for name, arr in arr_dict.items():
            plt.hist(arr, bins=bins, density=True, alpha=0.5, label=name)
        plt.title(title); plt.xlabel(xlabel); plt.ylabel("密度"); plt.grid(True, alpha=0.3)
        plt.legend(); plt.tight_layout(); plt.savefig(f"{out_png_prefix}_overlay.png", dpi=160); plt.close()

    if save_individual:
        for name, arr in arr_dict.items():
            plt.figure(figsize=(8.5, 5.0))
            plt.hist(arr, bins=bins, density=True, alpha=0.8)
            plt.title(f"{title} - {name}"); plt.xlabel(xlabel); plt.ylabel("密度")
            plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(f"{out_png_prefix}_{name}.png", dpi=160); plt.close()

def default_user_configs(all_cols: List[str]):
    cfgs = [
        {"name":"Cfg1", "SPY":0.45, "QQQ":0.22, "GLD":0.00,  "0050":0.13, "Cash":0.20},
        {"name":"Cfg2", "SPY":0.35, "QQQ":0.30, "GLD":0.075, "0050":0.15, "Cash":0.125},
    ]
    out = []
    for cfg in cfgs:
        name = cfg.get("name", f"Cfg{len(out)+1}")
        w = np.zeros(len(all_cols), dtype=np.float64)
        for j, c in enumerate(all_cols):
            if c in cfg:
                w[j] = float(cfg[c])
        s = w.sum()
        if s <= 0:
            w[:] = 1.0/len(all_cols)
        else:
            w /= s
        out.append({"name": name, "weights": w})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="normalized_closing_prices.csv")
    ap.add_argument("--years", type=int, default=20)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--abs_band", type=float, default=0.05)
    ap.add_argument("--rel_band", type=float, default=0.25)
    ap.add_argument("--cash_annual", type=float, default=0.00)
    ap.add_argument("--overlay", type=int, default=1)
    ap.add_argument("--save_individual", type=int, default=1)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()

    MONTHS = args.years * 12
    n_jobs = args.jobs if args.jobs>0 else max(1, (os.cpu_count() or 1))
    cash_monthly_ret = (1.0 + args.cash_annual)**(1.0/12.0) - 1.0

    mp = load_and_estimate(args.csv, cash_return_monthly=cash_monthly_ret)
    user_cfgs = default_user_configs(mp.all_cols)

    results = []
    cagr_map = {}
    mdd_map  = {}

    for i, cfg in enumerate(user_cfgs):
        name = cfg["name"]
        W = cfg["weights"]
        S = W.sum()
        if abs(S - 1.0) > 1e-9:
            W = W / max(S, 1e-12)
        status("SIM", ERR.OK, f"[{name}] 開始模擬：{args.sims} 次；workers={n_jobs}")

        cagr, mdd = simulate_config_distribution(
            W, mp.mu_m, mp.L_m, mp.all_cols, mp.noncash_cols,
            months=MONTHS, n_sims_total=args.sims, batch_size=args.batch, seed_base=args.seed,
            abs_band=args.abs_band, rel_band=args.rel_band, n_jobs=n_jobs
        )
        cagr_map[name] = cagr * 100.0
        mdd_map[name]  = -mdd * 100.0

        def P(a, q): return float(np.percentile(a, q))
        results.append({
            "name": name,
            "CAGR_mean": float(np.mean(cagr)),  "CAGR_med": float(np.median(cagr)),
            "CAGR_p05": P(cagr,5), "CAGR_p95": P(cagr,95),
            "MDD_mean":  float(np.mean(mdd)),   "MDD_med":  float(np.median(mdd)),
            "MDD_p05":  P(mdd,5),  "MDD_p95":  P(mdd,95),
        })

    plot_histogram(cagr_map, "年化報酬分佈（CAGR）", "年化報酬 (%)",
                   "cagr", overlay=bool(args.overlay), save_individual=bool(args.save_individual), bins=args.bins)
    plot_histogram(mdd_map,  "最大回撤分佈（|MDD|）", "最大回撤幅度 (%)",
                   "mdd",  overlay=bool(args.overlay), save_individual=bool(args.save_individual), bins=args.bins)

    df = pd.DataFrame(results)
    df.to_csv("summary_results.csv", index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
