"""Step 13 - GUIDE: robustness of the protocol effect found in step 12.

Adds, on top of step 12:
  D. organisation label purity - how many orgs grade (almost) everything one way, and an
     'org prior' scorer (train TP rate of the event's organisation, no model at all)
  E. three feature sets x three protocols x several seeds, evaluated per evidence row
     AND per incident (incident score = max over its rows):
       content : detector, title, category, MITRE, entity type/role, geo, OS, hour, rows per alert
       clean   : content + hashed entity identifiers (accounts, IPs, devices, files ...)
       all     : clean + OrgId, IncidentId, AlertId, Id
     protocols: official (Microsoft split), temporal (test from --cut), org_holdout
     (a different 30% of organisations held out for each seed)

Usage:  python 13_guide_robust.py --dir guide                    (3 seeds, ~2-3 h on free Colab)
        python 13_guide_robust.py --dir guide --seeds 0 --max-train 1000000   (quick)
Output: results/guide_robust/*.csv
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS_DIR

spec = importlib.util.spec_from_file_location("g12", Path(__file__).with_name("12_guide_audit.py"))
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

CONTENT = G.CONTEXT + ["hour", "n_rows_alert"]
FEATURE_SETS = {"content": CONTENT, "clean": G.CLEAN, "all": G.CLEAN + ["OrgId"] + G.ID_COLS}


def incident_metrics(df_te, y_te, s):
    key = pd.MultiIndex.from_arrays([df_te["OrgId"].values, df_te["IncidentId"].values])
    g = pd.DataFrame({"y": y_te, "s": s}, index=key).groupby(level=[0, 1]).max()
    m = G.metrics(g["y"].values, g["s"].values)
    return {f"inc_{k}": v for k, v in m.items()}


def org_prior(df, y, tr, te):
    rate = pd.Series(y[tr]).groupby(df["OrgId"].values[tr]).agg(["sum", "count"])
    prior = y[tr].mean()
    enc = (rate["sum"] + 10 * prior) / (rate["count"] + 10)
    return pd.Series(df["OrgId"].values[te]).map(enc).fillna(prior).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="guide")
    ap.add_argument("--cut", default="2024-06-10")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max-train", type=int, default=2_000_000)
    ap.add_argument("--protocols", default="official,temporal,org_holdout")
    ap.add_argument("--features", default="content,clean,all")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    out = RESULTS_DIR / "guide_robust"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df, y = G.add_derived(G.load_all(args.dir))
    print(f"labelled rows: {len(df):,}; TP share {y.mean():.4f}  [{time.time()-t0:.0f}s]")

    # ---------- D. organisation label purity
    print("\n== D. ORGANISATION LABEL PURITY")
    org = pd.DataFrame({"org": df["OrgId"].values, "y": y}).groupby("org")["y"].agg(["mean", "size"])
    bins = pd.cut(org["mean"], [-0.001, 0.05, 0.2, 0.8, 0.95, 1.0],
                  labels=["TP<5%", "5-20%", "20-80%", "80-95%", "TP>95%"])
    pur = pd.DataFrame({"orgs": org.groupby(bins, observed=False).size(),
                        "rows": org.groupby(bins, observed=False)["size"].sum()})
    pur["orgs_share"] = pur.orgs / pur.orgs.sum()
    pur["rows_share"] = pur.rows / pur.rows.sum()
    print(pur.round(4).to_string())
    pure = float(pur.loc[["TP<5%", "TP>95%"], "rows_share"].sum())
    print(f"rows in organisations that grade >95% one way (TP or not): {pure:.2%}")
    pur.to_csv(out / "D_org_purity.csv")

    # ---------- splits
    off_tr = np.where(df["split"].values == "train")[0]
    off_te = np.where(df["split"].values == "test")[0]
    cut = pd.Timestamp(args.cut, tz="UTC").to_datetime64()
    tmp_tr = np.where(df["Timestamp"].values < cut)[0]
    tmp_te = np.where(df["Timestamp"].values >= cut)[0]
    orgs = np.unique(df["OrgId"].values)

    def splits(name, seed):
        if name == "official":
            return off_tr, off_te
        if name == "temporal":
            return tmp_tr, tmp_te
        held = np.random.default_rng(seed).choice(orgs, int(0.3 * len(orgs)), replace=False)
        m = np.isin(df["OrgId"].values, held)
        return np.where(~m)[0], np.where(m)[0]

    # ---------- E. feature sets x protocols x seeds
    # official/temporal splits are fixed: extra seeds vary the training subsample and LightGBM;
    # org_holdout also draws a different set of held-out organisations per seed
    print("\n== E. PROTOCOLS x FEATURE SETS x SEEDS")
    rows = []
    for proto in args.protocols.split(","):
        for sd in seeds:
            tr, te = splits(proto, sd)
            d_te, y_te = df.iloc[te], y[te]
            s = org_prior(df, y, tr, te)
            rows.append({"protocol": proto, "features": "org_prior_only", "seed": sd,
                         "n_train": len(tr), **G.metrics(y_te, s), **incident_metrics(d_te, y_te, s)})
            print(f"  {proto:12s} seed={sd} org_prior_only PR-AUC={rows[-1]['pr_auc']:.3f} "
                  f"ROC={rows[-1]['roc_auc']:.3f}", flush=True)
            for fs in args.features.split(","):
                s = G.fit_score(df.iloc[tr], d_te, y[tr], y_te, FEATURE_SETS[fs], sd, args.max_train)
                r = {"protocol": proto, "features": fs, "seed": sd, "n_train": len(tr),
                     **G.metrics(y_te, s), **incident_metrics(d_te, y_te, s)}
                rows.append(r)
                print(f"  {proto:12s} seed={sd} {fs:14s} PR-AUC={r['pr_auc']:.3f} ROC={r['roc_auc']:.3f} "
                      f"R@1%={r['rec_at_1fpr']:.3f} | incident PR-AUC={r['inc_pr_auc']:.3f} "
                      f"ROC={r['inc_roc_auc']:.3f} base={r['base_rate']:.3f}  [{time.time()-t0:.0f}s]",
                      flush=True)
                pd.DataFrame(rows).to_csv(out / "E_runs.csv", index=False)

    runs = pd.DataFrame(rows)
    cols = ["pr_auc", "roc_auc", "rec_at_1fpr", "inc_pr_auc", "inc_roc_auc", "base_rate", "inc_base_rate"]
    summ = runs.groupby(["protocol", "features"])[cols].agg(["mean", "std"]).round(4)
    summ.to_csv(out / "E_summary.csv")
    print("\n##### SUMMARY (mean / std over seeds)")
    print(summ.to_string())
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
