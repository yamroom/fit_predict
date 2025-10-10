# -*- coding: utf-8 -*-
"""
Monte Carlo 最佳化配置（Numba + 多程序）
- 用 Numba 加速 TWR + 帶寬再平衡的「逐月迴圈」
- 外層以多程序（ProcessPoolExecutor）對「批次」平行
依賴：numpy pandas matplotlib numba
"""
import os, sys, math, time, traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit, prange

os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
plt.rcParams["font.family"] = "Noto Serif CJK JP"

class ERR:
    OK=0; FILE_READ_FAIL=1001; DATA_FORMAT_ERROR=1002; PARAM_ESTIMATION_FAIL=1101; GEN_WEIGHTS_FAIL=1201
    SIM_INIT_FAIL=1301; SIM_COARSE_FAIL=1302; RANKING_FAIL=1401; SIM_DETAILED_FAIL=1501; ANALYSIS_FAIL=1601; PLOTTING_FAIL=1701

def status(stage, code, msg=""):
    ts=time.strftime("%Y-%m-%d %H:%M:%S"); tag="OK" if code==0 else f"ERR{code}"; print(f"[{ts}] [{stage}] [{tag}] {msg}"); sys.stdout.flush()

def detect_datetime_col(df):
    for c in df.columns:
        p=pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
        if p.notna().mean()>0.9: return c, pd.DatetimeIndex(p)
    c=df.columns[0]; p=pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
    if p.notna().mean()>0.9: return c, pd.DatetimeIndex(p)
    return None, None

def pick_col(df, names):
    low={c.lower():c for c in df.columns}
    for want in names:
        for key,col in low.items():
            if want.lower()==key or want.lower() in key: return col
    return None

@dataclass
class MarketParams:
    mu_m: np.ndarray
    L_m: np.ndarray
    noncash_cols: List[str]
    all_cols: List[str]

def load_and_estimate(csv_path, cash_return_monthly=0.0):
    df=pd.read_csv(csv_path)
    date_col, date_parsed=detect_datetime_col(df)
    if date_parsed is not None: prices=df.set_index(date_parsed).drop(columns=[date_col],errors="ignore")
    else: prices=df.copy(); prices.index=pd.RangeIndex(len(prices))
    for c in prices.columns:
        if not pd.api.types.is_numeric_dtype(prices[c]):
            prices[c]=pd.to_numeric(prices[c], errors="coerce")
    prices=prices.sort_index().dropna(how="all")

    cSPY=pick_col(prices,["SPY"]); cQQQ=pick_col(prices,["QQQ"]); cGLD=pick_col(prices,["GLD"]); c0050=pick_col(prices,["0050.TW","0050"])
    use=[c for c in [cSPY,cQQQ,cGLD,c0050] if c is not None]
    if not use: raise ValueError("No assets found")

    px=prices[use].copy(); rename={}
    if cSPY: rename[cSPY]="SPY"
    if cQQQ: rename[cQQQ]="QQQ"
    if cGLD: rename[cGLD]="GLD"
    if c0050: rename[c0050]="0050"
    px=px.rename(columns=rename).dropna(how="any")

    px_m=px.resample("M").last()
    ret_m=px_m.pct_change().dropna(); ret_m["Cash"]=cash_return_monthly

    mu_m=ret_m[[c for c in ret_m.columns if c!="Cash"]].mean(axis=0).values.astype(np.float64)
    Sigma=np.cov(ret_m[[c for c in ret_m.columns if c!="Cash"]].T, ddof=0).astype(np.float64)
    eps=1e-10
    try: L=np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError: L=np.linalg.cholesky(Sigma+eps*np.eye(Sigma.shape[0]))
    noncash_cols=[c for c in ret_m.columns if c!="Cash"]; all_cols=noncash_cols+["Cash"]
    return MarketParams(mu_m=mu_m, L_m=L, noncash_cols=noncash_cols, all_cols=all_cols)

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

def simulate_monthly_returns(n_sims, months, mu, L, seed):
    rng=np.random.default_rng(seed); Z=rng.standard_normal((n_sims, months, len(mu))); return Z @ L.T + mu

@njit(parallel=True, fastmath=True)
def twr_numba(Wt_full, R_nc, idx_nc, idx_cash, abs_band, rel_band):
    """
    Numba 加速的 TWR + 年終 5/25 帶寬檢查
    Wt_full: (d_total,)
    R_nc: (n_sims, M, d_noncash)
    idx_nc: 非現金資產在 full 向量中的索引（長度 d_noncash）
    idx_cash: Cash 的索引
    回傳：cagr(n_sims,), mdd(n_sims,)
    """
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
        # 初始化
        w_full = np.zeros(d_total, dtype=np.float64)
        # 設為目標
        for j in range(d_total):
            w_full[j] = Wt_full[j]

        nav = 1.0
        peak = 1.0
        min_dd = 0.0

        for t in range(M):
            # r_p：僅非現金部分有報酬
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
            # 更新非現金權重（Cash 月報酬=0）
            for jnc in range(d_nc):
                j_full = idx_nc[jnc]
                w_full[j_full] = w_full[j_full] * (1.0 + R_nc[s, t, jnc]) / denom
            # Cash 權重
            w_full[idx_cash] = w_full[idx_cash] / denom

            # 年終（每12個月）
            if ((t+1) % 12) == 0:
                breach = False
                # 大權重：絕對 ±5pp
                for j in range(d_total):
                    if big_mask[j] == 1:
                        if np.abs(w_full[j] - Wt_full[j]) >= abs_band:
                            breach = True
                            break
                # 小權重：相對 ±25%
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

def coarse_worker(Wt, mu, L, idx_nc, idx_cash, abs_band, rel_band, n_sims, months, seed):
    R_nc = simulate_monthly_returns(n_sims, months, mu, L, seed)
    cagr, mdd = twr_numba(Wt.astype(np.float64), R_nc.astype(np.float64), idx_nc, idx_cash, abs_band, rel_band)
    return float(np.median(cagr)), float(np.median(mdd))


def coarse_screening_numba(W_list, mp, months=240, n_sims=500, seed=1234, abs_band=0.05, rel_band=0.25, n_jobs=None):
    n_jobs = n_jobs or max(1, (os.cpu_count() or 1))
    idx_nc  = np.array([mp.all_cols.index(c) for c in mp.noncash_cols], dtype=np.int64)
    idx_cash= np.int64(mp.all_cols.index("Cash"))

    # 先小編譯一次（warm-up）
    _ = coarse_worker(W_list[0], mp.mu_m, mp.L_m, idx_nc, idx_cash, abs_band, rel_band, n_sims, months, seed)

    rows=[]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(coarse_worker, W_list[i], mp.mu_m, mp.L_m, idx_nc, idx_cash, abs_band, rel_band, n_sims, months, seed): i for i in range(W_list.shape[0])}
        total=len(futs)
        for j, fut in enumerate(as_completed(futs), 1):
            i=futs[fut]
            c_med, d_med = fut.result()
            rows.append((i, c_med, d_med))
            left = total - j
            if j % 200 == 0 or left in (0,1):
                status("SIM-COARSE", 0, f"進度 {j}/{total}，剩 {left}")
    rows.sort(key=lambda x:x[0])
    return pd.DataFrame({"idx":[r[0] for r in rows], "CAGR_med":[r[1] for r in rows], "MDD_med":[r[2] for r in rows]})

def run_batch(b, W_top10, mp, months, n_sims_total, batch_size, seed_base, abs_band, rel_band):
    sims = batch_size if (b<math.ceil(n_sims_total/batch_size)-1) else (n_sims_total - batch_size*(math.ceil(n_sims_total/batch_size)-1))
    R_nc = simulate_monthly_returns(sims, months, mp.mu_m, mp.L_m, seed_base+b).astype(np.float64)
    idx_nc  = np.array([mp.all_cols.index(c) for c in mp.noncash_cols], dtype=np.int64)
    idx_cash= np.int64(mp.all_cols.index("Cash"))
    out=[]
    for i in range(W_top10.shape[0]):
        cagr, mdd = twr_numba(W_top10[i].astype(np.float64), R_nc, idx_nc, idx_cash, abs_band, rel_band)
        out.append((i, cagr, mdd))
    return b, out


def detailed_simulation_numba(W_top10, mp, months=240, n_sims_total=20000, batch_size=2000, seed_base=9876, abs_band=0.05, rel_band=0.25, n_jobs=None):
    n_jobs = n_jobs or max(1, (os.cpu_count() or 1))
    n_batches=math.ceil(n_sims_total/batch_size)

    # warm-up（編譯）
    _ = run_batch(0, W_top10, mp, months, n_sims_total, batch_size, seed_base, abs_band, rel_band)

    all_cagr=[[] for _ in range(W_top10.shape[0])]
    all_mdd =[[] for _ in range(W_top10.shape[0])]

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs={ex.submit(run_batch, b, W_top10, mp, months, n_sims_total, batch_size, seed_base, abs_band, rel_band): b for b in range(n_batches)}
        total=len(futs)
        for j, fut in enumerate(as_completed(futs), 1):
            b=futs[fut]
            _, out=fut.result()
            for (i,cagr,mdd) in out:
                all_cagr[i].append(cagr); all_mdd[i].append(mdd)
            left=total-j; status("SIM-DETAIL", 0, f"批次 {j}/{total} 完成；剩 {left}")

    rows=[]
    for i in range(W_top10.shape[0]):
        cagr=np.concatenate(all_cagr[i]); mdd=np.concatenate(all_mdd[i])
        rows.append({"rank":i+1,
                     "SPY":W_top10[i, mp.all_cols.index("SPY")] if "SPY" in mp.all_cols else 0.0,
                     "QQQ":W_top10[i, mp.all_cols.index("QQQ")] if "QQQ" in mp.all_cols else 0.0,
                     "GLD":W_top10[i, mp.all_cols.index("GLD")] if "GLD" in mp.all_cols else 0.0,
                     "0050":W_top10[i, mp.all_cols.index("0050")] if "0050" in mp.all_cols else 0.0,
                     "Cash":W_top10[i, mp.all_cols.index("Cash")],
                     "CAGR_mean":float(np.mean(cagr)), "CAGR_med":float(np.median(cagr)),
                     "CAGR_p05":float(np.percentile(cagr,5)), "CAGR_p95":float(np.percentile(cagr,95)),
                     "MDD_mean":float(np.mean(mdd)), "MDD_med":float(np.median(mdd)),
                     "MDD_p05":float(np.percentile(mdd,5)), "MDD_p95":float(np.percentile(mdd,95))})
    import pandas as pd
    return pd.DataFrame(rows).sort_values("CAGR_med", ascending=False).reset_index(drop=True)

def pick_top10_by_utopia(df, W_list):
    ret=df["CAGR_med"].to_numpy(); mdd=df["MDD_med"].to_numpy()
    rmin,rmax=ret.min(),ret.max(); dmin,dmax=mdd.min(),mdd.max()
    rnorm=(ret-rmin)/max(rmax-rmin,1e-12); dnorm=(mdd-dmin)/max(dmax-dmin,1e-12)
    dist2=(rnorm-1.0)**2 + (dnorm-1.0)**2
    ranked=df.copy(); ranked["score"]=dist2; ranked=ranked.sort_values("score").reset_index(drop=True)
    top10=ranked.head(10).copy(); top10["rank"]=np.arange(1,len(top10)+1)
    return top10, W_list[top10["idx"].to_numpy()]

def plot_coarse_frontier(df, out_png):
    plt.figure(figsize=(8.0,5.0))
    plt.scatter(-df["MDD_med"]*100, df["CAGR_med"]*100, s=8, alpha=0.4)
    plt.xlabel("Median |MDD| (%)")
    plt.ylabel("Median CAGR (%)")
    plt.title("Coarse Simulation (500 runs/group) Efficiency Cloud")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def plot_top10_bars(df, out_png):
    df=df.copy().sort_values("CAGR_med", ascending=False)
    labels=[f"#{i+1}" for i in range(len(df))]; x=np.arange(len(df)); w=0.35
    plt.figure(figsize=(10,5.2))
    plt.bar(x-w/2, df["CAGR_med"]*100, width=w, label="Median CAGR")
    plt.bar(x+w/2, -df["MDD_med"]*100, width=w, label="Median |MDD|")
    plt.xticks(x, labels); plt.ylabel("%"); plt.title("Top10: Median CAGR vs Median |MDD|")
    plt.legend(); plt.grid(True, axis="y", alpha=0.3); plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def main():
    CSV_PATH="/content/normalized_closing_prices.csv"
    CASH_ANNUAL_RATE=0.0; CASH_MONTHLY_RET=(1.0+CASH_ANNUAL_RATE)**(1/12)-1.0
    N_WEIGHTS=100_000; CASH_MAX=0.20; CONSTRAINTS=None
    YEARS=20; MONTHS=YEARS*12
    COARSE_NSIMS=5000; COARSE_SEED=1234
    DETAIL_NSIMS=200_000; DETAIL_BATCH=2000; DETAIL_SEED_BASE=9876
    ABS_BAND=0.05; REL_BAND=0.25
    N_JOBS=max(1,(os.cpu_count() or 1))

    mp = load_and_estimate(CSV_PATH, cash_return_monthly=CASH_MONTHLY_RET)
    W_list = gen_random_weights(N_WEIGHTS, mp.all_cols, cash_max=CASH_MAX, constraints=CONSTRAINTS, seed=42)

    # 粗篩（Numba 內核 + 多程序）
    df_coarse = coarse_screening_numba(W_list, mp, months=MONTHS, n_sims=COARSE_NSIMS, seed=COARSE_SEED, abs_band=ABS_BAND, rel_band=REL_BAND, n_jobs=N_JOBS)
    plot_coarse_frontier(df_coarse, "coarse_frontier_numba.png"); status("PLOT", 0, "coarse_frontier_numba.png")

    top10_meta, W_top10 = pick_top10_by_utopia(df_coarse, W_list)
    print("\n初步 Top10（Numba+MP）：")
    print(top10_meta[["idx","CAGR_med","MDD_med"]].to_string(index=False))

    # 強化（批次平行）
    df_detail = detailed_simulation_numba(W_top10, mp, months=MONTHS, n_sims_total=DETAIL_NSIMS, batch_size=DETAIL_BATCH, seed_base=DETAIL_SEED_BASE, abs_band=ABS_BAND, rel_band=REL_BAND, n_jobs=N_JOBS)
    plot_top10_bars(df_detail, "top10_bars_numba.png"); status("PLOT", 0, "top10_bars_numba.png")

    best=df_detail.iloc[0]
    def pct(x): return f"{x*100:.2f}%"
    print("\n==== 最終分析（Numba+MP）====")
    print("建議配置：", {c:f"{best[c]*100:.1f}%" for c in ["SPY","QQQ","GLD","0050","Cash"] if c in df_detail.columns})
    print(f"CAGR 中位 {pct(best['CAGR_med'])}，MDD 中位 {pct(best['MDD_med'])}")

if __name__=="__main__":
    main()