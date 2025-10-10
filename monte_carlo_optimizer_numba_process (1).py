# -*- coding: utf-8 -*-
"""
Monte Carlo 最佳化配置（Numba + 多程序，含 Top3 分佈繪圖；已修正 pickling）

功能摘要
1) 讀取歷史價格 CSV，月化後估計 μ、Σ（對非現金資產）
2) 生成 10,000 組權重（含 Cash，Cash≤20%）
3) 初步模擬：每組 500 次，年終 5/25 帶寬再平衡（Numba 內核加速）
4) 以「烏托邦距離」挑前 10 名
5) 強化模擬：前 10 名各 20,000 次（批次平行）→ 產生評比分佈統計（CAGR / MDD 的均值、中位、分位等）
6) **新增** Top 3 分佈：依強化結果排序選前 3 名，再各自跑 20,000 次，輸出
   - 年化報酬（CAGR）分佈直方圖
   - 最大回撤（MDD）分佈直方圖（以百分比的絕對幅度表示）
7) 產出圖檔：
   - coarse_frontier_numba.png
    - top10_bars_numba.png
    - top{1,2,3}_cagr_distribution.png
    - top{1,2,3}_mdd_distribution.png

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

# 避免 BLAS 在多程序下過度平行
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

# ---- Numba 加速內核（頂層定義，可被多程序 pickling/導入） ----
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

# ---- 頂層 worker：粗篩（每個權重） ----
def coarse_worker_numba(args):
    (Wt, months, n_sims, seed, mu, L, idx_nc, idx_cash, abs_band, rel_band) = args
    R_nc = simulate_monthly_returns(n_sims, months, mu, L, seed).astype(np.float64)
    cagr, mdd = twr_numba(Wt.astype(np.float64), R_nc, idx_nc, idx_cash, abs_band, rel_band)
    return float(np.median(cagr)), float(np.median(mdd))

# ---- 頂層 worker：強化（每個批次一次，內含 Top10 全部權重） ----
def detail_batch_worker_numba(args):
    (batch_idx, sims_this, months, seed, mu, L, W_top10, idx_nc, idx_cash, abs_band, rel_band) = args
    R_nc = simulate_monthly_returns(sims_this, months, mu, L, seed).astype(np.float64)
    out = []
    for i in range(W_top10.shape[0]):
        cagr, mdd = twr_numba(W_top10[i].astype(np.float64), R_nc, idx_nc, idx_cash, abs_band, rel_band)
        out.append((i, cagr, mdd))
    return batch_idx, out

def coarse_screening_numba(W_list, mu, L, all_cols, noncash_cols, months=240, n_sims=500, seed=1234, abs_band=0.05, rel_band=0.25, n_jobs=None):
    n_jobs = n_jobs or max(1, (os.cpu_count() or 1))
    idx_nc  = np.array([all_cols.index(c) for c in noncash_cols], dtype=np.int64)
    idx_cash= np.int64(all_cols.index("Cash"))
    status("SIM-COARSE", 0, f"Numba+MP 初步：{len(W_list)} 組 × {n_sims} 次；workers={n_jobs}")
    args = [(W_list[i], months, n_sims, seed, mu, L, idx_nc, idx_cash, abs_band, rel_band) for i in range(W_list.shape[0])]
    rows=[]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(coarse_worker_numba, a): i for i,a in enumerate(args)}
        total=len(futs)
        for j, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            c_med, d_med = fut.result()
            rows.append((i, c_med, d_med))
            left = total - j
            if j % 200 == 0 or left in (0,1):
                status("SIM-COARSE", 0, f"進度 {j}/{total}，剩 {left}")
    rows.sort(key=lambda x:x[0])
    return pd.DataFrame({"idx":[r[0] for r in rows], "CAGR_med":[r[1] for r in rows], "MDD_med":[r[2] for r in rows]})

def detailed_simulation_numba(W_top10, mu, L, all_cols, noncash_cols, months=240, n_sims_total=20000, batch_size=2000, seed_base=9876, abs_band=0.05, rel_band=0.25, n_jobs=None):
    n_jobs = n_jobs or max(1, (os.cpu_count() or 1))
    n_batches=math.ceil(n_sims_total/batch_size)
    idx_nc  = np.array([all_cols.index(c) for c in noncash_cols], dtype=np.int64)
    idx_cash= np.int64(all_cols.index("Cash"))

    status("SIM-DETAIL", 0, f"Numba+MP 強化：Top10×{n_sims_total}；批次 {n_batches}；workers={n_jobs}")
    args = []
    for b in range(n_batches):
        sims = batch_size if (b<n_batches-1) else (n_sims_total - batch_size*(n_batches-1))
        args.append((b, sims, months, seed_base+b, mu, L, W_top10, idx_nc, idx_cash, abs_band, rel_band))

    all_cagr=[[] for _ in range(W_top10.shape[0])]
    all_mdd =[[] for _ in range(W_top10.shape[0])]

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs={ex.submit(detail_batch_worker_numba, a): a[0] for a in args}
        total=len(futs)
        for j, fut in enumerate(as_completed(futs), 1):
            b_idx=futs[fut]
            _, out=fut.result()
            for (i,cagr,mdd) in out:
                all_cagr[i].append(cagr); all_mdd[i].append(mdd)
            left=total-j; status("SIM-DETAIL", 0, f"批次 {j}/{total} 完成；剩 {left}")

    rows=[]
    for i in range(W_top10.shape[0]):
        cagr=np.concatenate(all_cagr[i]); mdd=np.concatenate(all_mdd[i])
        rows.append({"rank":i+1,
                     "SPY":W_top10[i, all_cols.index("SPY")] if "SPY" in all_cols else 0.0,
                     "QQQ":W_top10[i, all_cols.index("QQQ")] if "QQQ" in all_cols else 0.0,
                     "GLD":W_top10[i, all_cols.index("GLD")] if "GLD" in all_cols else 0.0,
                     "0050":W_top10[i, all_cols.index("0050")] if "0050" in all_cols else 0.0,
                     "Cash":W_top10[i, all_cols.index("Cash")],
                     "CAGR_mean":float(np.mean(cagr)), "CAGR_med":float(np.median(cagr)),
                     "CAGR_p05":float(np.percentile(cagr,5)), "CAGR_p95":float(np.percentile(cagr,95)),
                     "MDD_mean":float(np.mean(mdd)),  "MDD_med":float(np.median(mdd)),
                     "MDD_p05":float(np.percentile(mdd,5)),  "MDD_p95":float(np.percentile(mdd,95))})
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
    plt.xlabel("中位 |MDD| (%)"); plt.ylabel("中位 CAGR (%)")
    plt.title("初步模擬（500次/組）效率雲")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def plot_top10_bars(df, out_png):
    df=df.copy().sort_values("CAGR_med", ascending=False)
    labels=[f"#{i+1}" for i in range(len(df))]; x=np.arange(len(df)); w=0.35
    plt.figure(figsize=(10,5.2))
    plt.bar(x-w/2, df["CAGR_med"]*100, width=w, label="CAGR(中位)")
    plt.bar(x+w/2, -df["MDD_med"]*100, width=w, label="|MDD|(中位)")
    plt.xticks(x, labels); plt.ylabel("%"); plt.title("Top10：CAGR(中位) vs |MDD|(中位)")
    plt.legend(); plt.grid(True, axis="y", alpha=0.3); plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

# ====== 新增：針對前 3 名，收集完整分佈並繪圖 ======
def build_weight_from_row(row, all_cols):
    """從 df_detail 的一列重建和 all_cols 對齊的權重向量。缺失資產補 0。"""
    w = []
    for c in all_cols:
        if c in row.index:
            w.append(float(row[c]))
        else:
            w.append(0.0)
    return np.array(w, dtype=np.float64)

def detail_batch_worker_numba_for_selected(args):
    """與 detail_batch_worker_numba 類似，但 W_top10 改為一般 W_sel（K <= 3）"""
    (batch_idx, sims_this, months, seed, mu, L, W_sel, idx_nc, idx_cash, abs_band, rel_band) = args
    R_nc = simulate_monthly_returns(sims_this, months, mu, L, seed).astype(np.float64)
    out = []
    for i in range(W_sel.shape[0]):
        cagr, mdd = twr_numba(W_sel[i].astype(np.float64), R_nc, idx_nc, idx_cash, abs_band, rel_band)
        out.append((i, cagr, mdd))
    return batch_idx, out

def simulate_distributions_for_weights(W_sel, mu, L, all_cols, noncash_cols,
                                       months=240, n_sims_total=20000, batch_size=2000, seed_base=5555,
                                       abs_band=0.05, rel_band=0.25, n_jobs=None):
    """對選出的多個權重（通常是前 3 名）收集完整的 CAGR / MDD 分佈。"""
    n_jobs = n_jobs or max(1, (os.cpu_count() or 1))
    idx_nc  = np.array([all_cols.index(c) for c in noncash_cols], dtype=np.int64)
    idx_cash= np.int64(all_cols.index("Cash"))

    n_batches = math.ceil(n_sims_total / batch_size)
    k = len(W_sel)
    all_cagr = [ [] for _ in range(k) ]
    all_mdd  = [ [] for _ in range(k) ]

    args = []
    for b in range(n_batches):
        sims = batch_size if (b<n_batches-1) else (n_sims_total - batch_size*(n_batches-1))
        args.append((b, sims, months, seed_base+b, mu, L, np.vstack(W_sel), idx_nc, idx_cash, abs_band, rel_band))

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(detail_batch_worker_numba_for_selected, a): a[0] for a in args}
        total = len(futs)
        for j, fut in enumerate(as_completed(futs), 1):
            b_idx = futs[fut]
            _, out = fut.result()
            for (i, cagr, mdd) in out:
                if i < k:
                    all_cagr[i].append(cagr)
                    all_mdd[i].append(mdd)
            left = total - j
            status("TOP3-SIMS", 0, f"批次 {j}/{total} 完成；剩 {left}")

    cagr_arrays = [ np.concatenate(all_cagr[i]) if all_cagr[i] else np.array([]) for i in range(k) ]
    mdd_arrays  = [ np.concatenate(all_mdd[i])  if all_mdd[i]  else np.array([]) for i in range(k) ]
    return cagr_arrays, mdd_arrays

def plot_distribution(arr, title, xlabel, out_png, bins=60):
    plt.figure(figsize=(8.5, 5.0))
    plt.hist(arr, bins=bins, density=True, alpha=0.8)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("密度")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def main():
    CSV_PATH="normalized_closing_prices.csv"
    CASH_ANNUAL_RATE=0.0; CASH_MONTHLY_RET=(1.0+CASH_ANNUAL_RATE)**(1/12)-1.0
    N_WEIGHTS=10_000; CASH_MAX=0.20; CONSTRAINTS=None
    YEARS=20; MONTHS=YEARS*12
    COARSE_NSIMS=500; COARSE_SEED=1234
    DETAIL_NSIMS=20_000; DETAIL_BATCH=2000; DETAIL_SEED_BASE=9876
    ABS_BAND=0.05; REL_BAND=0.25
    N_JOBS=max(1,(os.cpu_count() or 1))

    mp = load_and_estimate(CSV_PATH, cash_return_monthly=CASH_MONTHLY_RET)
    W_list = gen_random_weights(N_WEIGHTS, mp.all_cols, cash_max=CASH_MAX, constraints=CONSTRAINTS, seed=42)

    # 粗篩（Numba 內核 + 多程序；worker 已是頂層函式）
    df_coarse = coarse_screening_numba(W_list, mp.mu_m, mp.L_m, mp.all_cols, mp.noncash_cols, months=MONTHS, n_sims=COARSE_NSIMS, seed=COARSE_SEED, abs_band=ABS_BAND, rel_band=REL_BAND, n_jobs=N_JOBS)
    plot_coarse_frontier(df_coarse, "coarse_frontier_numba.png"); status("PLOT", 0, "coarse_frontier_numba.png")

    top10_meta, W_top10 = pick_top10_by_utopia(df_coarse, W_list)
    print("\n初步 Top10（Numba+MP）：")
    print(top10_meta[["idx","CAGR_med","MDD_med"]].to_string(index=False))

    # 強化（批次平行；worker 已是頂層函式）
    df_detail = detailed_simulation_numba(W_top10, mp.mu_m, mp.L_m, mp.all_cols, mp.noncash_cols, months=MONTHS, n_sims_total=DETAIL_NSIMS, batch_size=DETAIL_BATCH, seed_base=DETAIL_SEED_BASE, abs_band=ABS_BAND, rel_band=REL_BAND, n_jobs=N_JOBS)
    plot_top10_bars(df_detail, "top10_bars_numba.png"); status("PLOT", 0, "top10_bars_numba.png")

    best=df_detail.iloc[0]
    def pct(x): return f"{x*100:.2f}%"
    print("\n==== 最終分析（Numba+MP，已修正 pickling）====")
    print("建議配置：", {c:f"{best[c]*100:.1f}%" for c in ["SPY","QQQ","GLD","0050","Cash"] if c in df_detail.columns})
    print(f"CAGR 中位 {pct(best['CAGR_med'])}，MDD 中位 {pct(best['MDD_med'])}")

    # === 新增：前 3 名分佈 ===
    df_sorted = df_detail.sort_values("CAGR_med", ascending=False).reset_index(drop=True)
    top3 = df_sorted.head(3)
    W_sel = [build_weight_from_row(top3.iloc[i], mp.all_cols) for i in range(len(top3))]

    cagr_arrays, mdd_arrays = simulate_distributions_for_weights(
        W_sel, mp.mu_m, mp.L_m, mp.all_cols, mp.noncash_cols,
        months=MONTHS, n_sims_total=DETAIL_NSIMS, batch_size=DETAIL_BATCH,
        seed_base=DETAIL_SEED_BASE, abs_band=ABS_BAND, rel_band=REL_BAND, n_jobs=N_JOBS
    )

    for i in range(len(W_sel)):
        if cagr_arrays[i].size > 0:
            plot_distribution(cagr_arrays[i]*100.0,
                              title=f"Top{i+1} 年化報酬分佈（CAGR）",
                              xlabel="年化報酬 (%)",
                              out_png=f"top{i+1}_cagr_distribution.png",
                              bins=60)
        if mdd_arrays[i].size > 0:
            plot_distribution(mdd_arrays[i]*-100.0,
                              title=f"Top{i+1} 最大回撤分佈（|MDD|）",
                              xlabel="最大回撤幅度 (%)",
                              out_png=f"top{i+1}_mdd_distribution.png",
                              bins=60)

    print("\nTop 3 配置（依 CAGR_med 排序）：")
    for i in range(len(W_sel)):
        print(f"Top{i+1} 權重：", {c: f"{top3.iloc[i][c]*100:.1f}%" for c in ['SPY','QQQ','GLD','0050','Cash'] if c in top3.columns})
    print("已輸出：top1_cagr_distribution.png / top1_mdd_distribution.png 等圖檔。\n")

if __name__=="__main__":
    main()
