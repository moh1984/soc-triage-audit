"""Step 9 - is CtxGraph significantly better than the best baseline?

For each protocol, re-runs the strongest baselines with the SAME seeds and the same
90% training subsets as 07_repeat_seeds.py (so their numbers match the baseline
table), loads CtxGraph's saved test scores, and runs a paired stratified bootstrap
on the same test rows for:
  - PR-AUC difference   (ctxgraph - best baseline)
  - recall at 1% FPR difference
using seed-averaged rank scores for both sides.

Usage:
  python 09_compare_vs_baselines.py --ctx-tag asa_temporal_vpc_wave --only asa_temporal,vpc_wave
  python 09_compare_vs_baselines.py --ctx-tag asa_to_vpc_vpc_to_asa_uda --only asa_to_vpc,vpc_to_asa
Output: results/compare/<ctx-tag>.csv
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import load_signals, make_target, recall_at_fpr
from config import DATA_DIR, RESULTS_DIR
from features import context_features, text_features


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = load_module("baselines", "04_baselines.py")
BASELINES = ["logreg_tab", "logreg_tab+ctx", "lgbm_tab+ctx", "lgbm_all", "iforest_ctx"]


def rank(s):
    return pd.Series(s).rank(pct=True).to_numpy()


def paired(y, a, b, n=1000, seed=0):
    """Stratified paired bootstrap for PR-AUC and recall@1%FPR differences a - b."""
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    obs = (average_precision_score(y, a) - average_precision_score(y, b),
           recall_at_fpr(y, a, 0.01) - recall_at_fpr(y, b, 0.01))
    d_ap, d_r = [], []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d_ap.append(average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i]))
        d_r.append(recall_at_fpr(y[i], a[i], 0.01) - recall_at_fpr(y[i], b[i], 0.01))
    out = {}
    for name, o, d in [("pr_auc", obs[0], np.array(d_ap)), ("r1fpr", obs[1], np.array(d_r))]:
        lo, hi = np.quantile(d, [0.025, 0.975])
        p = float((d <= 0).mean()) if o > 0 else float((d >= 0).mean())
        out.update({f"{name}_diff": o, f"{name}_ci_lo": lo, f"{name}_ci_hi": hi, f"{name}_p": p})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx-tag", required=True, help="folder name under results/ctxgraph/")
    ap.add_argument("--only", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--vpc-cut", default="2024-07-28")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    protos = args.only.split(",")
    ctx_dir = RESULTS_DIR / "ctxgraph" / args.ctx_tag

    df, y = make_target(load_signals(DATA_DIR / "100m" / "rulehits.parquet"), "escalation")
    df = df.reset_index(drop=True)
    counts = pd.read_parquet(DATA_DIR / "100m" / "minute_counts.parquet")
    print("computing baseline features...", flush=True)
    CTX = context_features(df, counts).to_numpy(np.float32)
    TXT = text_features(df)
    P = B.protocols(df, y, args.vpc_cut)

    rows = []
    for proto in protos:
        tr, te = P[proto]
        yt = y[te]
        # CtxGraph scores
        cg = []
        for sd in seeds:
            f = ctx_dir / f"scores_{proto}_{sd}.npz"
            z = np.load(f)
            assert np.array_equal(z["idx"], te), f"test rows differ for {f}"
            cg.append(z["score"])
        cg_rank = np.mean([rank(s) for s in cg], axis=0)
        cg_ap = [average_precision_score(yt, s) for s in cg]
        # baselines, same seeds and subsets as 07
        scores = {}
        for sd in seeds:
            print(f"== {proto} baselines seed={sd}", flush=True)
            B.run_protocol(df, y, CTX, TXT, tr, te, proto, seed=sd, train_frac=0.9,
                           models=BASELINES, scores_out=scores)
        means = {m: np.mean([average_precision_score(yt, scores[(proto, m, sd)]) for sd in seeds])
                 for m in BASELINES}
        best = max(means, key=means.get)
        for m in BASELINES:
            b_rank = np.mean([rank(scores[(proto, m, sd)]) for sd in seeds], axis=0)
            res = paired(yt, cg_rank, b_rank, n=args.n_boot)
            rows.append({"protocol": proto, "baseline": m, "is_best_baseline": m == best,
                         "baseline_pr_auc_mean": means[m],
                         "ctxgraph_pr_auc_mean": float(np.mean(cg_ap)),
                         "ctxgraph_pr_auc_std": float(np.std(cg_ap, ddof=1)) if len(cg_ap) > 1 else np.nan,
                         **res})
    out = RESULTS_DIR / "compare"
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / f"{args.ctx_tag}.csv", index=False)
    print("\n##### CtxGraph vs baselines (paired bootstrap on seed-averaged ranks)")
    cols = ["protocol", "baseline", "is_best_baseline", "baseline_pr_auc_mean", "ctxgraph_pr_auc_mean",
            "pr_auc_diff", "pr_auc_ci_lo", "pr_auc_ci_hi", "pr_auc_p",
            "r1fpr_diff", "r1fpr_ci_lo", "r1fpr_ci_hi", "r1fpr_p"]
    print(res[cols].round(4).to_string(index=False))
    print("\nsaved ->", out / f"{args.ctx_tag}.csv")


if __name__ == "__main__":
    main()
