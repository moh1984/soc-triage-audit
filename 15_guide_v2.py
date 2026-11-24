"""Step 15 - GUIDE v2: incident-separated temporal split + clustered bootstrap.

Answers two review points on step 13:
  1. The step-13 temporal protocol cut evidence ROWS at --cut, so one incident could have
     rows on both sides. Here we (a) count such incidents and rows, and (b) assign each
     incident, with all its rows, by its FIRST timestamp (incident_temporal).
  2. Confidence intervals: incident-level metrics with bootstrap over incidents AND over
     organizations (whole organizations resampled), and paired differences between
     feature sets on the same test incidents (seed-averaged ranks).

Protocols: official, incident_temporal, org_holdout (seed s holds out draw s, as in step 13).
Feature sets: org prior (no model), content, clean, all  - same definitions as step 13.

Usage:  python 15_guide_v2.py --dir guide            (~2.5-3 h on free Colab; trains on 1M rows per fit)
Output: results/guide_v2/  span_check.csv, runs.csv, incident_ci.csv, paired.csv,
        incident_scores_<protocol>.parquet (seed-averaged incident ranks per feature set)
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import RESULTS_DIR

_s = importlib.util.spec_from_file_location("g12", Path(__file__).with_name("12_guide_audit.py"))
G = importlib.util.module_from_spec(_s)
_s.loader.exec_module(G)
_s = importlib.util.spec_from_file_location("g13", Path(__file__).with_name("13_guide_robust.py"))
R = importlib.util.module_from_spec(_s)
_s.loader.exec_module(R)

FSETS = ["content", "clean", "all"]


def rank(s):
    return pd.Series(s).rank(pct=True).to_numpy()


def to_incidents(org, inc, y, scores):
    """incident table: org, label, max score per feature set"""
    d = pd.DataFrame({"org": org, "inc": inc, "y": y, **scores})
    agg = {"y": "max", **{k: "max" for k in scores}}
    return d.groupby(["org", "inc"], sort=False).agg(agg).reset_index()


def boot_indices(groups, n, rng):
    """yield index arrays that resample whole groups with replacement"""
    order = np.argsort(groups, kind="stable")
    _, start, cnt = np.unique(groups[order], return_index=True, return_counts=True)
    k = len(start)
    for _ in range(n):
        c = rng.randint(0, k, k)
        L = cnt[c]
        off = np.repeat(start[c] - np.concatenate([[0], np.cumsum(L)[:-1]]), L)
        yield order[np.arange(L.sum()) + off]


def clustered_stats(tab, names, n, seed):
    """CIs of incident ROC-AUC / PR-AUC per feature set and of paired differences vs content,
    under incident resampling and organisation resampling."""
    y = tab["y"].values
    out_ci, out_pair = [], []
    for level, groups in [("incident", np.arange(len(tab))),
                          ("organization", pd.factorize(tab["org"])[0])]:
        rng = np.random.RandomState(seed)
        roc = {k: [] for k in names}
        pr = {k: [] for k in names}
        for i in boot_indices(groups, n, rng):
            yi = y[i]
            if yi.min() == yi.max():
                continue
            for k in names:
                s = tab[k].values[i]
                roc[k].append(roc_auc_score(yi, s))
                pr[k].append(average_precision_score(yi, s))
        for k in names:
            r, p = np.array(roc[k]), np.array(pr[k])
            out_ci.append({"level": level, "features": k,
                           "roc": roc_auc_score(y, tab[k]), "roc_lo": np.quantile(r, .025), "roc_hi": np.quantile(r, .975),
                           "pr": average_precision_score(y, tab[k]), "pr_lo": np.quantile(p, .025), "pr_hi": np.quantile(p, .975)})
        for k in names:
            if k == "content":
                continue
            d = np.array(roc[k]) - np.array(roc["content"])
            obs = roc_auc_score(y, tab[k]) - roc_auc_score(y, tab["content"])
            out_pair.append({"level": level, "a": k, "b": "content", "roc_diff": obs,
                             "lo": np.quantile(d, .025), "hi": np.quantile(d, .975),
                             "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
    return out_ci, out_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="guide")
    ap.add_argument("--cut", default="2024-06-10")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max-train", type=int, default=1_000_000)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--protocols", default="official,incident_temporal,org_holdout")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    out = RESULTS_DIR / "guide_v2"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df, y = G.add_derived(G.load_all(args.dir))
    org = df["OrgId"].values
    inc_key = pd.factorize(pd.MultiIndex.from_arrays([org, df["IncidentId"].values]))[0]
    ts = df["Timestamp"].values
    cut = pd.Timestamp(args.cut, tz="UTC").to_datetime64()
    print(f"labelled rows: {len(df):,}  [{time.time()-t0:.0f}s]")

    # ---------- 1. incidents spanning the cut
    first = pd.Series(ts).groupby(inc_key).transform("min").values
    last = pd.Series(ts).groupby(inc_key).transform("max").values
    span = (first < cut) & (last >= cut)
    n_inc = len(np.unique(inc_key))
    n_span = len(np.unique(inc_key[span]))
    row_test = ts >= cut
    leak_rows = (row_test & span).mean() / row_test.mean()
    sc = {"incidents": n_inc, "incidents_spanning_cut": n_span, "share_incidents_spanning": n_span / n_inc,
          "rows_in_spanning_incidents": int(span.sum()), "share_rows_spanning": float(span.mean()),
          "share_of_row_split_test_rows_whose_incident_also_in_train": float(leak_rows)}
    print("\n== 1. INCIDENTS SPANNING THE TEMPORAL CUT")
    for k, v in sc.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v:,}")
    pd.DataFrame([sc]).to_csv(out / "span_check.csv", index=False)

    # ---------- splits
    off_tr = np.where(df["split"].values == "train")[0]
    off_te = np.where(df["split"].values == "test")[0]
    it_tr, it_te = np.where(first < cut)[0], np.where(first >= cut)[0]
    orgs = np.unique(org)

    def splits(name, seed):
        if name == "official":
            return off_tr, off_te
        if name == "incident_temporal":
            return it_tr, it_te
        held = np.random.default_rng(seed).choice(orgs, int(0.3 * len(orgs)), replace=False)
        m = np.isin(org, held)
        return np.where(~m)[0], np.where(m)[0]

    print(f"\nincident_temporal: train {len(it_tr):,} rows, test {len(it_te):,} rows "
          f"(base rate {y[it_te].mean():.3f})")

    # ---------- 2. models, incident tables, clustered CIs
    runs, ci_rows, pair_rows = [], [], []
    for proto in args.protocols.split(","):
        fixed = proto != "org_holdout"
        acc = {}                              # seed-averaged ranks for fixed test sets
        for sd in seeds:
            tr, te = splits(proto, sd)
            d_te, y_te = df.iloc[te], y[te]
            sc_rows = {"org_prior": R.org_prior(df, y, tr, te)}
            for fs in FSETS:
                sc_rows[fs] = G.fit_score(df.iloc[tr], d_te, y[tr], y_te, R.FEATURE_SETS[fs], sd, args.max_train)
            for k, s in sc_rows.items():
                r = {"protocol": proto, "features": k, "seed": sd, "n_train": len(tr), "n_test": len(te),
                     **G.metrics(y_te, s), **R.incident_metrics(d_te, y_te, s)}
                runs.append(r)
                print(f"  {proto:17s} seed={sd} {k:9s} row ROC={r['roc_auc']:.3f} | incident "
                      f"ROC={r['inc_roc_auc']:.3f} PR={r['inc_pr_auc']:.3f} base={r['inc_base_rate']:.3f}"
                      f"  [{time.time()-t0:.0f}s]", flush=True)
            pd.DataFrame(runs).to_csv(out / "runs.csv", index=False)
            if fixed:
                for k, s in sc_rows.items():
                    acc[k] = acc.get(k, 0) + rank(s) / len(seeds)
            else:                             # different test set per seed: CIs per draw
                tab = to_incidents(org[te], inc_key[te], y_te, {k: rank(s) for k, s in sc_rows.items()})
                ci, pr_ = clustered_stats(tab, list(sc_rows), args.n_boot, sd)
                for x in ci + pr_:
                    x.update({"protocol": proto, "draw": sd})
                ci_rows += ci
                pair_rows += pr_
        if fixed:
            tr, te = splits(proto, seeds[0])
            tab = to_incidents(org[te], inc_key[te], y[te], acc)
            tab.to_parquet(out / f"incident_scores_{proto}.parquet", index=False)
            ci, pr_ = clustered_stats(tab, list(acc), args.n_boot, 0)
            for x in ci + pr_:
                x.update({"protocol": proto, "draw": "fixed"})
            ci_rows += ci
            pair_rows += pr_
        pd.DataFrame(ci_rows).to_csv(out / "incident_ci.csv", index=False)
        pd.DataFrame(pair_rows).to_csv(out / "paired.csv", index=False)
        print(f"\n  CIs for {proto} done  [{time.time()-t0:.0f}s]", flush=True)

    runs = pd.DataFrame(runs)
    summ = (runs.groupby(["protocol", "features"])[["roc_auc", "rec_at_1fpr", "inc_roc_auc", "inc_pr_auc",
                                                   "base_rate", "inc_base_rate"]].agg(["mean", "std"]).round(4))
    summ.to_csv(out / "summary.csv")
    print("\n##### SUMMARY (mean / std over seeds)\n", summ.to_string())
    print("\n##### INCIDENT-LEVEL ROC-AUC, 95% CI (seed-averaged ranks; org_holdout per draw)")
    print(pd.DataFrame(ci_rows)[["protocol", "draw", "level", "features", "roc", "roc_lo", "roc_hi"]]
          .round(3).to_string(index=False))
    print("\n##### PAIRED: feature set minus content, incident ROC-AUC")
    print(pd.DataFrame(pair_rows)[["protocol", "draw", "level", "a", "roc_diff", "lo", "hi", "p_two_sided"]]
          .round(4).to_string(index=False))
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
