"""Step 7 - repeat the baseline ladder with 5 seeds and test the key comparisons.

Each seed trains every model on a different random 90% of the training split
(the test split never changes), so all models - including deterministic ones -
show run-to-run variance. Reported per protocol x model: mean, std, min, max of
PR-AUC, ROC-AUC and recall at 1% FPR.

Key comparisons (paired bootstrap on the SAME test rows, seed-averaged scores):
  context effect   : logreg_tab+ctx vs logreg_tab, lgbm_tab+ctx vs lgbm_tab
  unsupervised     : iforest_ctx vs best supervised model of that protocol
  text             : logreg_text vs logreg_tab
p = share of bootstrap resamples where the difference has the opposite sign
(one-sided); 95% CI of the PR-AUC difference is reported.

Usage: python 07_repeat_seeds.py --data 100m [--seeds 0,1,2,3,4] [--only asa_temporal,vpc_wave]
Outputs: results/seeds/<protocols>/runs.csv, summary.csv, comparisons.csv
"""
import argparse
import importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from common import load_signals, make_target
from config import DATA_DIR, RESULTS_DIR
from features import context_features, text_features

spec = importlib.util.spec_from_file_location("baselines", "04_baselines.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

PAIRS = [("logreg_tab+ctx", "logreg_tab", "context effect (logreg)"),
         ("lgbm_tab+ctx", "lgbm_tab", "context effect (lgbm)"),
         ("logreg_text", "logreg_tab", "raw text vs tabular (logreg)")]


def paired_bootstrap(y, a, b, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    obs = average_precision_score(y, a) - average_precision_score(y, b)
    diffs = []
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(n):  # stratified resampling keeps the base rate fixed
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs.append(average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i]))
    diffs = np.array(diffs)
    p = float((diffs <= 0).mean()) if obs > 0 else float((diffs >= 0).mean())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return obs, float(lo), float(hi), p


def _cluster_index(codes):
    """rows grouped by cluster: order, start offsets and lengths"""
    order = np.argsort(codes, kind="stable")
    uniq, start, cnt = np.unique(codes[order], return_index=True, return_counts=True)
    return order, start, cnt


def cluster_bootstrap(y, a, b, codes, n=500, seed=0):
    """Paired bootstrap of the PR-AUC difference a - b that resamples whole clusters
    (incidents or time blocks) with replacement, so correlated rows move together."""
    rng = np.random.RandomState(seed)
    order, start, cnt = _cluster_index(codes)
    k = len(start)
    obs = average_precision_score(y, a) - average_precision_score(y, b)
    diffs = []
    for _ in range(n):
        c = rng.randint(0, k, k)
        L = cnt[c]
        off = np.repeat(start[c] - np.concatenate([[0], np.cumsum(L)[:-1]]), L)
        i = order[np.arange(L.sum()) + off]
        if y[i].min() == y[i].max():
            continue
        diffs.append(average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i]))
    diffs = np.array(diffs)
    p = float((diffs <= 0).mean()) if obs > 0 else float((diffs >= 0).mean())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return obs, float(lo), float(hi), p


def test_clusters(df_te):
    """incident clusters (rows of one incident together; others alone) and 1-hour blocks"""
    from common import first_incident
    inc = df_te["incident_ids"].map(first_incident) if "incident_ids" in df_te else pd.Series([None] * len(df_te))
    inc = inc.astype(object).where(inc.notna(), pd.Series([f"row{i}" for i in range(len(df_te))], index=df_te.index))
    inc_codes = pd.factorize(inc.astype(str))[0]
    block = (df_te["timestamp"].values // 3600).astype(np.int64)
    return {"incident": inc_codes, "hour_block": pd.factorize(block)[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["small", "100m"], default="100m")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--train-frac", type=float, default=0.9)
    ap.add_argument("--only", default="", help="comma list of protocols")
    ap.add_argument("--vpc-cut", default="2024-07-28")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--models", default="", help="comma list; empty = all")
    ap.add_argument("--tag", default="", help="suffix of the output folder")
    ap.add_argument("--n-cluster-boot", type=int, default=500,
                    help="resamples for incident- and time-block bootstrap (0 = skip)")
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m] or None
    seeds = [int(s) for s in args.seeds.split(",")]

    path = DATA_DIR / "100m" / "rulehits.parquet" if args.data == "100m" else None
    df, y = make_target(load_signals(path), "escalation")
    counts = pd.read_parquet(DATA_DIR / "100m" / "minute_counts.parquet") if args.data == "100m" else None
    print(f"rows={len(df):,} positives={y.sum():,}; computing context + text features...", flush=True)
    CTX = context_features(df, counts).to_numpy(np.float32)
    TXT = text_features(df)
    P = B.protocols(df, y, args.vpc_cut)
    keep = [p for p in args.only.split(",") if p] or [p for p in P if p != "naive_global_temporal"]

    tag = "_".join(keep) + (f"_{args.tag}" if args.tag else "")
    out = RESULTS_DIR / "seeds" / tag      # one folder per run, so batches never overwrite
    out.mkdir(parents=True, exist_ok=True)
    rows, scores = [], {}
    for proto in keep:
        tr, te = P[proto]
        for sd in seeds:
            print(f"\n== {proto} seed={sd}", flush=True)
            rows += B.run_protocol(df, y, CTX, TXT, tr, te, proto, seed=sd,
                                   train_frac=args.train_frac, models=models, scores_out=scores)
        pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)   # save as we go

    runs = pd.DataFrame(rows)
    summ = (runs.groupby(["protocol", "model"])
                .agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                     pr_auc_min=("pr_auc", "min"), pr_auc_max=("pr_auc", "max"),
                     roc_mean=("roc_auc", "mean"), roc_std=("roc_auc", "std"),
                     r1_mean=("recall@fpr1%", "mean"), base_rate=("base_rate", "first"),
                     n_seeds=("seed", "nunique"))
                .reset_index())
    summ["lift_mean"] = summ.pr_auc_mean / summ.base_rate
    summ = summ.sort_values(["protocol", "pr_auc_mean"], ascending=[True, False])
    summ.to_csv(out / "summary.csv", index=False)
    print("\n##### SUMMARY (mean ± std over seeds)")
    for proto, g in summ.groupby("protocol", sort=False):
        print(f"\n{proto}  (base rate {g.base_rate.iloc[0]:.4f})")
        for _, r in g.iterrows():
            print(f"  {r.model:15s} PR-AUC {r.pr_auc_mean:.3f} ± {r.pr_auc_std:.3f}"
                  f"  (lift {r.lift_mean:.1f}x)  ROC {r.roc_mean:.3f} ± {r.roc_std:.3f}"
                  f"  R@1%FPR {r.r1_mean:.3f}")

    # paired comparisons on seed-averaged scores
    comps = []
    for proto in keep:
        _, te = P[proto]
        yt = y[te]
        avg = {}
        for m in B.MODELS:
            ss = [scores[(proto, m, sd)] for sd in seeds if (proto, m, sd) in scores]
            if ss:  # average ranks, so models with different score scales are comparable
                avg[m] = np.mean([pd.Series(s).rank(pct=True).to_numpy() for s in ss], axis=0)
        pairs = list(PAIRS)
        sup = summ[(summ.protocol == proto) & (summ.model != "iforest_ctx")]
        if len(sup):
            pairs.append(("iforest_ctx", sup.iloc[0].model, "unsupervised vs best supervised"))
        if "iforest_ctx_all" in avg and "iforest_ctx" in avg:
            pairs.append(("iforest_ctx_all", "iforest_ctx", "iForest: all events vs negatives only"))
        clus = test_clusters(df.iloc[te]) if args.n_cluster_boot else {}
        for a, b, label in pairs:
            if a in avg and b in avg and yt.sum() > 0:
                d, lo, hi, p = paired_bootstrap(yt, avg[a], avg[b], n=args.n_boot)
                row = {"protocol": proto, "comparison": label, "a": a, "b": b,
                       "pr_auc_diff": d, "ci95_lo": lo, "ci95_hi": hi, "p_one_sided": p}
                for cname, codes in clus.items():
                    _, clo, chi, cp = cluster_bootstrap(yt, avg[a], avg[b], codes, n=args.n_cluster_boot)
                    row.update({f"{cname}_ci95_lo": clo, f"{cname}_ci95_hi": chi, f"{cname}_p": cp})
                comps.append(row)
                print(f"  {proto:16s} {label:38s} diff={d:+.4f} row-CI=[{lo:+.4f},{hi:+.4f}] p={p:.3f}"
                      + "".join(f" | {c} CI=[{row[c+'_ci95_lo']:+.4f},{row[c+'_ci95_hi']:+.4f}] p={row[c+'_p']:.3f}"
                                for c in clus), flush=True)
    comps = pd.DataFrame(comps)
    comps.to_csv(out / "comparisons.csv", index=False)
    print("\n##### PAIRED COMPARISONS (PR-AUC difference a - b, seed-averaged)")
    if len(comps):
        print(comps.round(4).to_string(index=False))
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
