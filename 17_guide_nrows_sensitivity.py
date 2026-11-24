"""Step 17 - GUIDE: sensitivity of the results to how `n_rows_alert` is computed.

`add_derived` in 12_guide_audit.py counts the evidence rows of each (OrgId, AlertId) on the union of
the released training and test files. An alert attached to incidents in both files (40% of test
alerts) therefore gets a count that includes rows from the other side of the split. The feature
carries no label, but it is information from across the split, so this step re-fits the models of
step 15 with three versions of the feature:

  pooled       : as in the paper (counted on training + test rows together)
  within_split : counted separately on the training rows and on the test rows of each protocol
  dropped      : feature removed

Protocols: official and incident_temporal (as in step 15). org_holdout is a control: all rows of a
held-out organization are on the test side and the count is per organization, so pooled and
within_split are identical there by construction (add it with --protocols to check).

Usage:  python 17_guide_nrows_sensitivity.py --dir guide                     (~1-2 h on free Colab)
        python 17_guide_nrows_sensitivity.py --dir guide --features content,clean   (~2-4 h)
Output: results/guide_nrows_sensitivity/  runs.csv, summary.csv, count_changes.csv
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS_DIR


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load("g12", "12_guide_audit.py")
R = _load("g13", "13_guide_robust.py")
VARIANTS = ["pooled", "within_split", "dropped"]


def within_counts(d):
    return d.groupby(["OrgId", "AlertId"], observed=True)["Id"].transform("size").astype("int32").values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="guide")
    ap.add_argument("--cut", default="2024-06-10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=1_000_000)
    ap.add_argument("--protocols", default="official,incident_temporal")
    ap.add_argument("--features", default="content")
    args = ap.parse_args()
    out = RESULTS_DIR / "guide_nrows_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df, y = G.add_derived(G.load_all(args.dir))           # n_rows_alert = pooled count (as published)
    org = df["OrgId"].values
    inc_key = pd.factorize(pd.MultiIndex.from_arrays([org, df["IncidentId"].values]))[0]
    first = pd.Series(df["Timestamp"].values).groupby(inc_key).transform("min").values
    cut = pd.Timestamp(args.cut, tz="UTC").to_datetime64()
    orgs = np.unique(org)
    print(f"labelled rows: {len(df):,}  [{time.time()-t0:.0f}s]", flush=True)

    def splits(name):
        if name == "official":
            return np.where(df["split"].values == "train")[0], np.where(df["split"].values == "test")[0]
        if name == "incident_temporal":
            return np.where(first < cut)[0], np.where(first >= cut)[0]
        held = np.random.default_rng(args.seed).choice(orgs, int(0.3 * len(orgs)), replace=False)
        m = np.isin(org, held)
        return np.where(~m)[0], np.where(m)[0]

    runs, changes = [], []
    for proto in args.protocols.split(","):
        tr, te = splits(proto)
        d_tr, d_te = df.iloc[tr].copy(), df.iloc[te].copy()
        pooled_tr, pooled_te = d_tr["n_rows_alert"].values.copy(), d_te["n_rows_alert"].values.copy()
        win_tr, win_te = within_counts(d_tr), within_counts(d_te)
        ch = {"protocol": proto,
              "train_rows_count_changes": float((win_tr != pooled_tr).mean()),
              "test_rows_count_changes": float((win_te != pooled_te).mean()),
              "test_alerts_count_changes": float(
                  pd.Series(win_te != pooled_te).groupby(pd.factorize(
                      pd.MultiIndex.from_arrays([d_te["OrgId"].values, d_te["AlertId"].values]))[0]).any().mean())}
        changes.append(ch)
        print(f"\n== {proto}: share of rows whose count changes  train {ch['train_rows_count_changes']:.3f}"
              f"  test {ch['test_rows_count_changes']:.3f}  (test alerts {ch['test_alerts_count_changes']:.3f})",
              flush=True)
        y_tr, y_te = y[tr], y[te]
        for fs in args.features.split(","):
            base_cols = R.FEATURE_SETS[fs]
            for v in VARIANTS:
                cols = [c for c in base_cols if c != "n_rows_alert"] if v == "dropped" else base_cols
                d_tr["n_rows_alert"] = win_tr if v == "within_split" else pooled_tr
                d_te["n_rows_alert"] = win_te if v == "within_split" else pooled_te
                s = G.fit_score(d_tr, d_te, y_tr, y_te, cols, args.seed, args.max_train)
                r = {"protocol": proto, "features": fs, "variant": v, "seed": args.seed,
                     **G.metrics(y_te, s), **R.incident_metrics(d_te, y_te, s)}
                runs.append(r)
                print(f"  {proto:17s} {fs:8s} {v:13s} row ROC={r['roc_auc']:.3f} | incident "
                      f"ROC={r['inc_roc_auc']:.3f} PR={r['inc_pr_auc']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
                pd.DataFrame(runs).to_csv(out / "runs.csv", index=False)
        pd.DataFrame(changes).to_csv(out / "count_changes.csv", index=False)

    runs = pd.DataFrame(runs)
    summ = runs.pivot_table(index=["protocol", "features"], columns="variant",
                            values=["roc_auc", "inc_roc_auc", "inc_pr_auc"]).round(4)
    for m in ["roc_auc", "inc_roc_auc", "inc_pr_auc"]:
        for v in ["within_split", "dropped"]:
            summ[(f"{m}_diff", f"{v}-pooled")] = (summ[(m, v)] - summ[(m, "pooled")]).round(4)
    summ.to_csv(out / "summary.csv")
    print("\n##### SUMMARY (seed %d)\n" % args.seed, summ.to_string())
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
