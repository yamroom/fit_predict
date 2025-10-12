# -*- coding: utf-8 -*-
"""
Robust Monte Carlo Engine (Numba + Multiprocessing)
Enhancements included:
- Covariance shrinkage (auto Ledoit-Wolf if sklearn available, else ridge/diagonal shrinkage fallback)
- Optional mean shrinkage toward grand mean (alpha)
- Multiple simulation engines:
  * normal (Gaussian + Cholesky)  [baseline]
  * t (Multivariate t with dof)   [fat tails & tail dependence]
  * block_bootstrap (multivariate moving-block bootstrap) [preserve time & cross dependencies]
  * ewma (RiskMetrics-style time-varying vol with constant correlation) [vol clustering]
- Parameter uncertainty (outer loop): bootstrap resamples of historical monthly returns to re-estimate mu & Sigma
- Risk metrics from path:
  * CAGR, MDD
  * Ulcer Index (RMS of negative drawdowns)
  * Time-Under-Water (share of months below prior peak)
  * CVaR_5 for CAGR and |MDD| computed post-simulation
- Annual end-of-year 5/25 bandwidth rebalancing
- Pickling-safe top-level workers

Notes:
* Charts use matplotlib defaults (no explicit colors). One chart per figure.
* OMP/MKL threads limited to 1 per process to avoid nested parallel slowdowns.
"""

import os, sys, math, time, argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

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
    SIM_FAIL=1301; RANK_FAIL=1401; PLOT_FAIL=1701

def status(stage: str, code: int, msg: str = ""):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    tag = "OK" if code == 0 else f"ERR{code}"
    print(f"[{ts}] [{stage}] [{tag}] {msg}")
    sys.stdout.flush()

# ---------- Utilities: Data ingest & estimation ----------

@dataclass
class MarketParams:
    mu_m: np.ndarray            # (d_nc,)
    Sigma_m: np.ndarray         # (d_nc, d_nc) sample covariance (pre-shrink)
    L_m: np.ndarray             # (d_nc, d_nc) Cholesky of shrunk covariance
    corr_m: np.ndarray          # (d_nc, d_nc) correlation matrix
    vol_m: np.ndarray           # (d_nc,) monthly vols (sqrt(diag(Sigma)))
    noncash_cols: List[str]     # non-cash asset order
    all_cols: List[str]         # non-cash + ["Cash"]
    ret_m_full: pd.DataFrame    # monthly returns incl Cash column

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

def try_ledoit_wolf(S: np.ndarray) -> Optional[np.ndarray]:
    try:
        from sklearn.covariance import LedoitWolf
        return None
    except Exception:
        return None

def shrink_cov_from_returns(R: np.ndarray, method: str = "auto", ridge: float = 0.10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, d = R.shape
    S = np.cov(R.T, ddof=0)
    vol = np.sqrt(np.clip(np.diag(S), 1e-16, None))
    corr = S / np.outer(vol, vol)
    corr = np.clip(corr, -0.999, 0.999)

    if method == "auto":
        Sigma_shrunk = None
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(R)
            Sigma_shrunk = lw.covariance_
        except Exception:
            pass
        if Sigma_shrunk is None:
            lam = min(0.5, d / max(T, 1) * 0.1)
            D = np.diag(np.diag(S))
            Sigma_shrunk = (1 - lam) * S + lam * D
    elif method == "ridge":
        lam = float(ridge)
        D = np.diag(np.diag(S))
        Sigma_shrunk = (1 - lam) * S + lam * D
    else:
        Sigma_shrunk = S

    eps = 1e-10
    try:
        _ = np.linalg.cholesky(Sigma_shrunk)
    except np.linalg.LinAlgError:
        Sigma_shrunk = Sigma_shrunk + eps * np.eye(d)

    vol = np.sqrt(np.clip(np.diag(Sigma_shrunk), 1e-16, None))
    corr = Sigma_shrunk / np.outer(vol, vol)
    corr = np.clip(corr, -0.999, 0.999)
    return Sigma_shrunk, corr, vol

def shrink_mean(mu: np.ndarray, target: Optional[np.ndarray] = None, alpha: float = 0.0) -> np.ndarray:
    if alpha <= 0: return mu.copy()
    d = mu.shape[0]
    if target is None:
        g = np.full(d, float(mu.mean()))
    else:
        g = target.astype(float)
    return (1.0 - alpha) * mu + alpha * g

def load_and_estimate(csv_path: str, cash_return_monthly: float = 0.0,
                      cov_shrink: str = "auto", cov_ridge: float = 0.10,
                      mean_shrink_alpha: float = 0.0) -> MarketParams:
    df = pd.read_csv(csv_path)
    date_col, date_parsed = detect_datetime_col(df)
    if date_parsed is not None:
        prices = df.set_index(date_parsed).drop(columns=[date_col], errors="ignore")
    else:
        prices = df.copy(); prices.index = pd.RangeIndex(len(prices))

    for c in prices.columns:
        if not pd.api.types.is_numeric_dtype(prices[c]):
            prices[c] = pd.to_numeric(prices[c], errors="coerce")
    prices = prices.sort_index().dropna(how="all")

    cSPY  = pick_col(prices, ["SPY"]); cQQQ  = pick_col(prices, ["QQQ"])
    cGLD  = pick_col(prices, ["GLD"]); c0050 = pick_col(prices, ["0050.TW", "0050"])
    use = [c for c in [cSPY, cQQQ, cGLD, c0050] if c is not None]
    if len(use) == 0:
        raise ValueError("No usable asset columns (SPY/QQQ/GLD/0050 not found).")

    px = prices[use].copy()
    rename = {}
    if cSPY:  rename[cSPY]  = "SPY"
    if cQQQ:  rename[cQQQ]  = "QQQ"
    if cGLD:  rename[cGLD]  = "GLD"
    if c0050: rename[c0050] = "0050"
    px = px.rename(columns=rename).dropna(how="any")

    px_m = px.resample("M").last()
    ret_nc = px_m.pct_change().dropna()
    ret_full = ret_nc.copy()
    ret_full["Cash"] = cash_return_monthly

    mu = ret_nc.mean(axis=0).values.astype(np.float64)
    mu = shrink_mean(mu, target=None, alpha=mean_shrink_alpha)
    Sigma_shrunk, corr, vol = shrink_cov_from_returns(ret_nc.values, method=cov_shrink, ridge=cov_ridge)
    eps = 1e-10
    try:
        L = np.linalg.cholesky(Sigma_shrunk)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(Sigma_shrunk + eps*np.eye(Sigma_shrunk.shape[0]))

    noncash_cols = list(ret_nc.columns)
    all_cols = noncash_cols + ["Cash"]
    status("LOAD", ERR.OK, f"Monthly data {ret_nc.index.min().date()}~{ret_nc.index.max().date()} | Assets: {noncash_cols}")
    return MarketParams(mu_m=mu, Sigma_m=Sigma_shrunk, L_m=L, corr_m=corr, vol_m=vol,
                        noncash_cols=noncash_cols, all_cols=all_cols, ret_m_full=ret_full)

def simulate_monthly_returns_normal(n_sims: int, months: int, mu: np.ndarray, L: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_sims, months, len(mu)))
    return Z @ L.T + mu

def simulate_monthly_returns_t(n_sims: int, months: int, mu: np.ndarray, L: np.ndarray, df: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = len(mu)
    Z = rng.standard_normal((n_sims, months, d))
    V = rng.chisquare(df, size=(n_sims, months)) / df
    scale = 1.0 / np.sqrt(V)
    Zs = Z * scale[..., None]
    return Zs @ L.T + mu

def simulate_monthly_returns_block_bootstrap(ret_nc: np.ndarray, n_sims: int, months: int, block_len: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T, d = ret_nc.shape
    if block_len <= 0: block_len = 6
    n_blocks = math.ceil(months / block_len)
    out = np.zeros((n_sims, months, d), dtype=np.float64)
    for s in range(n_sims):
        seq = []
        for b in range(n_blocks):
            i0 = rng.integers(0, T)
            block = ret_nc[np.arange(i0, i0+block_len) % T, :]
            seq.append(block)
        path = np.vstack(seq)[:months, :]
        out[s, :, :] = path
    return out

def simulate_monthly_returns_ewma(n_sims: int, months: int, mu: np.ndarray, corr: np.ndarray, vol0: np.ndarray,
                                  lam: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = len(mu)
    Lcorr = np.linalg.cholesky(corr + 1e-12*np.eye(d))
    out = np.empty((n_sims, months, d), dtype=np.float64)
    for s in range(n_sims):
        sigma = vol0.copy()
        for t in range(months):
            eps = rng.standard_normal(d)
            z = Lcorr @ eps
            sigma = np.sqrt(lam * sigma**2 + (1.0 - lam) * (z**2))
            out[s, t, :] = mu + sigma * z
    return out

def simulate_engine(engine: str, n_sims: int, months: int, mp, seed: int, t_df: float = 5.0, block_len: int = 6, ewma_lambda: float = 0.94) -> np.ndarray:
    if engine == "normal":
        return simulate_monthly_returns_normal(n_sims, months, mp.mu_m, mp.L_m, seed)
    elif engine == "t":
        return simulate_monthly_returns_t(n_sims, months, mp.mu_m, mp.L_m, t_df, seed)
    elif engine == "block_bootstrap":
        ret_nc = mp.ret_m_full[mp.noncash_cols].values.astype(np.float64)
        return simulate_monthly_returns_block_bootstrap(ret_nc, n_sims, months, block_len, seed)
    elif engine == "ewma":
        return simulate_monthly_returns_ewma(n_sims, months, mp.mu_m, mp.corr_m, mp.vol_m, ewma_lambda, seed)
    else:
        raise ValueError(f"Unknown engine: {engine}")

@njit(parallel=True, fastmath=True, cache=True)
def twr_numba(Wt_full, R_nc, idx_nc, idx_cash, abs_band, rel_band):
    n_sims, M, d_nc = R_nc.shape
    YEARS = M / 12.0
    d_total = Wt_full.shape[0]

    cagr  = np.empty(n_sims, dtype=np.float64)
    mdd   = np.empty(n_sims, dtype=np.float64)
    ulcer = np.empty(n_sims, dtype=np.float64)
    tuw   = np.empty(n_sims, dtype=np.float64)

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

        sum_dd_sq = 0.0
        months_under = 0.0

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
            if dd < 0.0:
                months_under += 1.0
                sum_dd_sq += dd * dd

            denom = (1.0 + r_p)
            for jnc in range(d_nc):
                j_full = idx_nc[jnc]
                w_full[j_full] = w_full[j_full] * (1.0 + R_nc[s, t, jnc]) / denom
            w_full[idx_cash] = w_full[idx_cash] / denom

            if ((t+1) % 12) == 0:
                breach = False
                for j in range(d_total):
                    if big_mask[j] == 1:
                        if abs(w_full[j] - Wt_full[j]) >= abs_band:
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
        if months_under > 0.0:
            ulcer[s] = math.sqrt(sum_dd_sq / months_under)
        else:
            ulcer[s] = 0.0
        tuw[s] = months_under / M

    return cagr, mdd, ulcer, tuw

def batch_worker(args):
    (batch_idx, W_full, months, sims_this, seed, mp_pack, engine, t_df, block_len, ewma_lambda,
     idx_nc, idx_cash, abs_band, rel_band) = args
    mu, L, corr, vol, ret_full_nc = mp_pack
    if engine == "normal":
        R_nc = simulate_monthly_returns_normal(sims_this, months, mu, L, seed).astype(np.float64)
    elif engine == "t":
        R_nc = simulate_monthly_returns_t(sims_this, months, mu, L, t_df, seed).astype(np.float64)
    elif engine == "block_bootstrap":
        R_nc = simulate_monthly_returns_block_bootstrap(ret_full_nc, sims_this, months, block_len, seed).astype(np.float64)
    elif engine == "ewma":
        R_nc = simulate_monthly_returns_ewma(sims_this, months, mu, corr, vol, ewma_lambda, seed).astype(np.float64)
    else:
        raise ValueError("unknown engine")
    cagr, mdd, ulcer, tuw = twr_numba(W_full.astype(np.float64), R_nc, idx_nc, idx_cash, abs_band, rel_band)
    return batch_idx, cagr, mdd, ulcer, tuw

def simulate_config_distribution(W_full: np.ndarray, mp, months: int, n_sims_total: int, batch_size: int,
                                 seed_base: int, abs_band: float, rel_band: float, engine: str,
                                 t_df: float, block_len: int, ewma_lambda: float, n_jobs: int):
    idx_nc  = np.array([mp.all_cols.index(c) for c in mp.noncash_cols], dtype=np.int64)
    idx_cash= np.int64(mp.all_cols.index("Cash"))
    n_batches = math.ceil(n_sims_total / batch_size)

    mp_pack = (mp.mu_m, mp.L_m, mp.corr_m, mp.vol_m, mp.ret_m_full[mp.noncash_cols].values.astype(np.float64))

    args = []
    for b in range(n_batches):
        sims_this = batch_size if (b < n_batches-1) else (n_sims_total - batch_size*(n_batches-1))
        args.append((b, W_full, months, sims_this, seed_base + b, mp_pack, engine, t_df, block_len, ewma_lambda,
                     idx_nc, idx_cash, abs_band, rel_band))

    all_cagr=[]; all_mdd=[]; all_ulcer=[]; all_tuw=[]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(batch_worker, a): a[0] for a in args}
        total = len(futs)
        for j, fut in enumerate(as_completed(futs), 1):
            b_idx = futs[fut]
            _, c, d, u, w = fut.result()
            all_cagr.append(c); all_mdd.append(d); all_ulcer.append(u); all_tuw.append(w)
            left = total - j; status("SIM", ERR.OK, f"批次 {j}/{total} 完成；剩 {left}")
    cagr = np.concatenate(all_cagr); mdd = np.concatenate(all_mdd)
    ulcer = np.concatenate(all_ulcer); tuw = np.concatenate(all_tuw)
    return cagr, mdd, ulcer, tuw

def cvar_from_distribution(x: np.ndarray, alpha: float = 0.05, lower_tail: bool = True) -> float:
    if x.size == 0: return np.nan
    if lower_tail:
        q = np.quantile(x, alpha)
        return float(x[x <= q].mean())
    else:
        q = np.quantile(x, 1-alpha)
        return float(x[x >= q].mean())

def gen_random_weights(n, all_cols, cash_max=0.20, constraints=None, seed=42):
    rng=np.random.default_rng(seed); d=len(all_cols); alpha=np.ones(d); alpha[all_cols.index("Cash")]=0.6
    res=[]; tries=0; MAX_TRIES=n*50
    while len(res)<n and tries<MAX_TRIES:
        w=rng.dirichlet(alpha); tries+=1
        if w[all_cols.index("Cash")]>cash_max+1e-12: continue
        bad=False
        if constraints:
            for k,(lo,hi) in constraints.items():
                if k in all_cols:
                    j=all_cols.index(k)
                    if not (lo-1e-12<=w[j]<=hi+1e-12): bad=True; break
        if bad: continue
        res.append(w.astype(np.float64))
    if len(res)<n: raise RuntimeError("insufficient weights")
    return np.vstack(res)

def pick_top_by_utopia(df, W_list, k=10):
    ret=df["CAGR_med"].to_numpy(); mdd=df["MDD_med"].to_numpy()
    rmin,rmax=ret.min(),ret.max(); dmin,dmax=mdd.min(),mdd.max()
    rnorm=(ret-rmin)/max(rmax-rmin,1e-12); dnorm=(mdd-dmin)/max(dmax-dmin,1e-12)
    dist2=(rnorm-1.0)**2 + (dnorm-1.0)**2
    ranked=df.copy(); ranked["score"]=dist2; ranked=ranked.sort_values("score").reset_index(drop=True)
    topk=ranked.head(k).copy(); topk["rank"]=np.arange(1,len(topk)+1)
    return topk, W_list[topk["idx"].to_numpy()]

def summary_from_arrays(cagr, mdd, ulcer, tuw):
    def P(a,q): return float(np.percentile(a,q))
    return {
        "CAGR_mean": float(np.mean(cagr)),  "CAGR_med": float(np.median(cagr)),
        "CAGR_p05":  P(cagr,5),             "CAGR_p95":  P(cagr,95),
        "MDD_mean":  float(np.mean(mdd)),   "MDD_med":   float(np.median(mdd)),
        "MDD_p05":   P(mdd,5),              "MDD_p95":   P(mdd,95),
        "Ulcer_med": float(np.median(ulcer)), "TUW_med":  float(np.median(tuw)),
        "CAGR_CVaR5": cvar_from_distribution(cagr, 0.05, lower_tail=True),
        "MDD_CVaR5":  cvar_from_distribution(-mdd, 0.05, lower_tail=False)
    }

def plot_scatter_frontier(df, out_png):
    plt.figure(figsize=(8,5))
    plt.scatter(-df["MDD_med"]*100, df["CAGR_med"]*100, s=8, alpha=0.4)
    plt.xlabel("中位 |MDD| (%)"); plt.ylabel("中位 CAGR (%)")
    plt.title("初步模擬（500次/組）效率雲"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def plot_topk_bars(df, out_png):
    df=df.copy().sort_values("CAGR_med", ascending=False); labels=[f"#{i+1}" for i in range(len(df))]
    x=np.arange(len(df)); w=0.35
    plt.figure(figsize=(10,5.2))
    plt.bar(x-w/2, df["CAGR_med"]*100, width=w, label="CAGR(中位)")
    plt.bar(x+w/2, -df["MDD_med"]*100, width=w, label="|MDD|(中位)")
    plt.xticks(x, labels); plt.ylabel("%"); plt.title("TopK：CAGR(中位) vs |MDD|(中位)")
    plt.legend(); plt.grid(True, axis="y", alpha=0.3); plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def build_weight_from_row(row, all_cols):
    w = np.zeros(len(all_cols), dtype=np.float64)
    for j,c in enumerate(all_cols):
        if c in row.index:
            try: w[j] = float(row[c])
            except: w[j] = 0.0
    return w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="normalized_closing_prices.csv")
    ap.add_argument("--years", type=int, default=20)
    ap.add_argument("--coarse_sims", type=int, default=500)
    ap.add_argument("--detail_sims", type=int, default=20000)
    ap.add_argument("--detail_batch", type=int, default=2000)
    ap.add_argument("--weights", type=int, default=10000)
    ap.add_argument("--cash_max", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--abs_band", type=float, default=0.05)
    ap.add_argument("--rel_band", type=float, default=0.25)
    ap.add_argument("--cash_annual", type=float, default=0.00)
    ap.add_argument("--engine", type=str, default="t", choices=["normal","t","block_bootstrap","ewma"])
    ap.add_argument("--t_df", type=float, default=5.0)
    ap.add_argument("--block_len", type=int, default=6)
    ap.add_argument("--ewma_lambda", type=float, default=0.94)
    ap.add_argument("--cov_shrink", type=str, default="auto", choices=["auto","ridge","none"])
    ap.add_argument("--cov_ridge", type=float, default=0.10)
    ap.add_argument("--mean_shrink_alpha", type=float, default=0.0)
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()

    MONTHS = args.years * 12
    n_jobs = args.jobs if args.jobs>0 else max(1,(os.cpu_count() or 1))
    cash_monthly_ret = (1.0 + args.cash_annual)**(1.0/12.0) - 1.0

    mp = load_and_estimate(args.csv, cash_return_monthly=cash_monthly_ret,
                           cov_shrink=args.cov_shrink, cov_ridge=args.cov_ridge,
                           mean_shrink_alpha=args.mean_shrink_alpha)

    W_list = gen_random_weights(args.weights, mp.all_cols, cash_max=args.cash_max, constraints=None, seed=42)

    status("SIM-COARSE", ERR.OK, f"{len(W_list)} weights × {args.coarse_sims} sims | engine={args.engine}")
    rows = []
    idx_nc  = np.array([mp.all_cols.index(c) for c in mp.noncash_cols], dtype=np.int64)
    idx_cash= np.int64(mp.all_cols.index("Cash"))

    chunk = 200
    for start in range(0, len(W_list), chunk):
        end = min(len(W_list), start+chunk)
        batch = W_list[start:end]
        for i, W in enumerate(batch, start):
            R_nc = simulate_engine(args.engine, args.coarse_sims, MONTHS, mp, seed=args.seed+i,
                                   t_df=args.t_df, block_len=args.block_len, ewma_lambda=args.ewma_lambda).astype(np.float64)
            c,m,u,t = twr_numba(W.astype(np.float64), R_nc, idx_nc, idx_cash, args.abs_band, args.rel_band)
            rows.append((i, float(np.median(c)), float(np.median(m))))
        status("SIM-COARSE", ERR.OK, f"progress {end}/{len(W_list)}")

    df_coarse = pd.DataFrame(rows, columns=["idx","CAGR_med","MDD_med"])
    plot_scatter_frontier(df_coarse, "coarse_frontier_upgraded.png")

    top10_meta, W_top10 = pick_top_by_utopia(df_coarse, W_list, k=10)
    status("SIM-DETAIL", ERR.OK, f"Top10 × {args.detail_sims} sims (batch={args.detail_batch}) | engine={args.engine}")
    all_rows = []
    for rank, irow in enumerate(range(W_top10.shape[0])):
        W = W_top10[irow]
        cagr, mdd, ulcer, tuw = simulate_config_distribution(W, mp, MONTHS, args.detail_sims, args.detail_batch,
                                                             args.seed, args.abs_band, args.rel_band, args.engine,
                                                             args.t_df, args.block_len, args.ewma_lambda, n_jobs)
        def P(a,q): return float(np.percentile(a,q))
        summ = {
            "CAGR_mean": float(np.mean(cagr)),  "CAGR_med": float(np.median(cagr)),
            "CAGR_p05":  P(cagr,5),             "CAGR_p95":  P(cagr,95),
            "MDD_mean":  float(np.mean(mdd)),   "MDD_med":   float(np.median(mdd)),
            "MDD_p05":   P(mdd,5),              "MDD_p95":   P(mdd,95),
            "Ulcer_med": float(np.median(ulcer)), "TUW_med":  float(np.median(tuw)),
            "CAGR_CVaR5": cvar_from_distribution(cagr, 0.05, lower_tail=True),
            "MDD_CVaR5":  cvar_from_distribution(-mdd, 0.05, lower_tail=False)
        }
        all_rows.append({
            "rank": rank+1,
            "SPY": W[mp.all_cols.index("SPY")] if "SPY" in mp.all_cols else 0.0,
            "QQQ": W[mp.all_cols.index("QQQ")] if "QQQ" in mp.all_cols else 0.0,
            "GLD": W[mp.all_cols.index("GLD")] if "GLD" in mp.all_cols else 0.0,
            "0050": W[mp.all_cols.index("0050")] if "0050" in mp.all_cols else 0.0,
            "Cash": W[mp.all_cols.index("Cash")],
            **summ
        })
        if rank < 3:
            np.savez(f"top{rank+1}_distributions.npz", CAGR=cagr, MDD=mdd, Ulcer=ulcer, TUW=tuw)
            def plot_distribution(arr, title, xlabel, out_png, bins=60):
                plt.figure(figsize=(8.5,5.0)); plt.hist(arr, bins=bins, density=True, alpha=0.8)
                plt.title(title); plt.xlabel(xlabel); plt.ylabel("密度"); plt.grid(True, alpha=0.3)
                plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
            plot_distribution(cagr*100, f"Top{rank+1} 年化報酬分佈（CAGR）", "年化報酬 (%)", f"top{rank+1}_cagr_distribution.png")
            plot_distribution(-mdd*100, f"Top{rank+1} 最大回撤分佈（|MDD|）", "最大回撤幅度 (%)", f"top{rank+1}_mdd_distribution.png")

    df_detail = pd.DataFrame(all_rows).sort_values("CAGR_med", ascending=False).reset_index(drop=True)
    df_detail.to_csv("top10_detail_summary.csv", index=False)
    plot_topk_bars(df_detail.head(10), "top10_bars_upgraded.png")

    best = df_detail.iloc[0]
    def pct(x): return f"{x*100:.2f}%"
    print("\n==== 最終分析（升級版）====")
    print("建議配置：", {c:f"{best[c]*100:.1f}%" for c in ["SPY","QQQ","GLD","0050","Cash"] if c in df_detail.columns})
    print(f"CAGR 中位 {pct(best['CAGR_med'])}，MDD 中位 {pct(best['MDD_med'])}，Ulcer(中位) {pct(best['Ulcer_med'])}，TUW(中位) {best['TUW_med']*100:.1f}%")
    print("輸出：coarse_frontier_upgraded.png、top10_bars_upgraded.png、Top1~Top3 分佈圖與 .npz")

if __name__ == "__main__":
    main()
