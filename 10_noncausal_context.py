"""Step 10 - how much does NON-causal (event-wide) context inflate results?

Motivation: some published alert classifiers enrich each alert with statistics of the
whole security event or cluster it belongs to, which includes alerts that arrive AFTER
the alert being classified, and evaluate with alert-level random splits. This script
computes the same entity-context features in two ways and compares them:

  causal     : counts of earlier rule-matched events for the same entity (paper default)
  two_sided  : counts of earlier AND later events for the same entity (event-wide)

under four protocols:
  random_row       : alert-level random 80/20 split (events of one incident on both sides)
  incident_grouped : random split with incidents kept whole
  asa_temporal     : cisco_asa, past -> future
  vpc_wave         : aws_vpc_flow_log, first wave -> second wave

Models: LR and GBDT on tabular + context. Five seeds, each on a random 90% of training.
Paired bootstrap (two_sided - causal) on the same test rows, seed-averaged ranks.

Usage: python 10_noncausal_context.py --data 100m
Output: results/noncausal/runs.csv, summary.csv, comparisons.csv
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import load_signals, make_target, split_random
from config import DATA_DIR, RESULTS_DIR
from features import context_features


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = load_module("baselines", "04_baselines.py")
S = load_module("seeds", "07_repeat_seeds.py")
MODELS = ["logreg_tab+ctx", "lgbm_tab+ctx"]


def rank(s):
    return pd.Series(s).rank(pct=True).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["small", "100m"], default="100m")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--only", default="random_row,incident_grouped,asa_temporal,vpc_wave")
    ap.add_argument("--vpc-cut", default="2024-07-28")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tag", default="", help="subfolder of results/noncausal")
    ap.add_argument("--n-cluster-boot", type=int, default=500,
                    help="resamples for incident- and hour-block bootstrap (0 = skip)")
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]
    seeds = [int(s) for s in args.seeds.split(",")]
    protos = [p for p in args.only.split(",") if p]

    path = DATA_DIR / "100m" / "rulehits.parquet" if args.data == "100m" else None
    df, y = make_target(load_signals(path), "escalation")
    df = df.reset_index(drop=True)
    counts = pd.read_parquet(DATA_DIR / "100m" / "minute_counts.parquet") if args.data == "100m" else None
    print(f"rows={len(df):,} positives={y.sum():,}; computing causal and two-sided context...", flush=True)
    CTX = {"causal": context_features(df, counts, mode="causal").to_numpy(np.float32),
           "two_sided": context_features(df, counts, mode="two_sided").to_numpy(np.float32)}

    P = B.protocols(df, y, args.vpc_cut)
    P["random_row"] = split_random(df, y)
    out = RESULTS_DIR / "noncausal" / args.tag if args.tag else RESULTS_DIR / "noncausal"
    out.mkdir(parents=True, exist_ok=True)

    rows, scores = [], {}
    for proto in protos:
        tr, te = P[proto]
        for mode in ["causal", "two_sided"]:
            for sd in seeds:
                print(f"\n== {proto} context={mode} seed={sd}", flush=True)
                sc = {}
                r = B.run_protocol(df, y, CTX[mode], None, tr, te, proto, seed=sd,
                                   train_frac=0.9, models=models, scores_out=sc)
                for x in r:
                    x["context"] = mode
                rows += r
                for (p, m, s), v in sc.items():
                    scores[(p, m, mode, s)] = v
            pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)

    runs = pd.DataFrame(rows)
    summ = (runs.groupby(["protocol", "model", "context"])
                .agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                     roc_mean=("roc_auc", "mean"), r1_mean=("recall@fpr1%", "mean"),
                     base_rate=("base_rate", "first")).reset_index())
    summ["lift_mean"] = summ.pr_auc_mean / summ.base_rate
    summ.to_csv(out / "summary.csv", index=False)

    comps = []
    for proto in protos:
        _, te = P[proto]
        yt = y[te]
        for m in models:
            a = np.mean([rank(scores[(proto, m, "two_sided", sd)]) for sd in seeds], axis=0)
            b = np.mean([rank(scores[(proto, m, "causal", sd)]) for sd in seeds], axis=0)
            d, lo, hi, p = S.paired_bootstrap(yt, a, b, n=args.n_boot)
            row = {"protocol": proto, "model": m, "two_sided_minus_causal": d,
                   "ci95_lo": lo, "ci95_hi": hi, "p_one_sided": p}
            if args.n_cluster_boot:
                for cname, codes in S.test_clusters(df.iloc[te]).items():
                    _, clo, chi, cp = S.cluster_bootstrap(yt, a, b, codes, n=args.n_cluster_boot)
                    row.update({f"{cname}_ci95_lo": clo, f"{cname}_ci95_hi": chi, f"{cname}_p": cp})
            comps.append(row)
            print(f"  {proto:17s} {m:15s} diff={d:+.4f} p={p:.3f}"
                  + (f" | incident p={row['incident_p']:.3f} | hour_block p={row['hour_block_p']:.3f}"
                     if args.n_cluster_boot else ""), flush=True)
    comps = pd.DataFrame(comps)
    comps.to_csv(out / "comparisons.csv", index=False)

    print("\n##### SUMMARY: PR-AUC mean ± std over seeds (lift)")
    for (proto, m), g in summ.groupby(["protocol", "model"], sort=False):
        c = g.set_index("context")
        print(f"  {proto:17s} {m:15s} causal {c.loc['causal','pr_auc_mean']:.3f} ± {c.loc['causal','pr_auc_std']:.3f}"
              f" ({c.loc['causal','lift_mean']:.1f}x) | two-sided {c.loc['two_sided','pr_auc_mean']:.3f}"
              f" ± {c.loc['two_sided','pr_auc_std']:.3f} ({c.loc['two_sided','lift_mean']:.1f}x)")
    print("\n##### PAIRED: two-sided minus causal (PR-AUC, seed-averaged ranks)")
    print(comps.round(4).to_string(index=False))
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
