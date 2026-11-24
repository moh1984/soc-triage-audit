"""Step 16 - GUIDE: the three checks left open by review.

  A. AlertId: does one alert belong to several incidents? (no training)
  B. Strict temporal split: incidents that span --cut are DROPPED, so every training row is
     before the cut and every test incident starts after it. One seed (fixed split; step 15
     showed s.d. <= 0.003 across seeds), org prior + content + clean + all, with incident- and
     organization-level bootstrap.
  C. Identifier ablation on unseen organizations: 'all' versus 'all without IncidentId' versus
     'content' in the same three organization draws as steps 13 and 15, with paired
     organization-level bootstrap. Tests whether IncidentId explains the gain of 'all'.

Usage:  python 16_guide_final.py --dir guide          (~2 h on free Colab)
Output: results/guide_final/  alert_check.csv, runs.csv, incident_ci.csv, paired.csv
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS_DIR


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


G = _load("g12", "12_guide_audit.py")
R = _load("g13", "13_guide_robust.py")
V = _load("g15", "15_guide_v2.py")

FS = dict(R.FEATURE_SETS)
FS["all_no_incidentid"] = [c for c in FS["all"] if c != "IncidentId"]


def run_split(df, y, org, inc_key, tr, te, names, seed, max_train, n_boot, proto, draw):
    d_te, y_te = df.iloc[te], y[te]
    sc = {}
    if "org_prior" in names:
        sc["org_prior"] = R.org_prior(df, y, tr, te)
    for fs in names:
        if fs != "org_prior":
            sc[fs] = G.fit_score(df.iloc[tr], d_te, y[tr], y_te, FS[fs], seed, max_train)
    runs = []
    for k, s in sc.items():
        r = {"protocol": proto, "draw": draw, "features": k, "n_train": len(tr), "n_test": len(te),
             **G.metrics(y_te, s), **R.incident_metrics(d_te, y_te, s)}
        runs.append(r)
        print(f"  {proto:16s} draw={draw} {k:18s} row ROC={r['roc_auc']:.3f} R@1%={r['rec_at_1fpr']:.3f} | "
              f"incident ROC={r['inc_roc_auc']:.3f} PR={r['inc_pr_auc']:.3f} base={r['inc_base_rate']:.3f}", flush=True)
    tab = V.to_incidents(org[te], inc_key[te], y_te, {k: V.rank(s) for k, s in sc.items()})
    ci, pr = V.clustered_stats(tab, list(sc), n_boot, 0)
    for x in ci + pr:
        x.update({"protocol": proto, "draw": draw})
    return runs, ci, pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="guide")
    ap.add_argument("--cut", default="2024-06-10")
    ap.add_argument("--max-train", type=int, default=1_000_000)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--skip", default="", help="comma list of parts to skip: A,B,C")
    args = ap.parse_args()
    skip = set(args.skip.split(","))
    out = RESULTS_DIR / "guide_final"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df, y = G.add_derived(G.load_all(args.dir))
    org = df["OrgId"].values
    inc_key = pd.factorize(pd.MultiIndex.from_arrays([org, df["IncidentId"].values]))[0]
    print(f"labelled rows: {len(df):,}  [{time.time()-t0:.0f}s]")

    # ---------- A. AlertId across incidents
    if "A" not in skip:
        print("\n== A. ALERTS IN MORE THAN ONE INCIDENT")
        a = pd.DataFrame({"alert": df["AlertId"].values, "inc": inc_key, "split": df["split"].values})
        n_inc = a.groupby("alert")["inc"].nunique()
        n_split = a.groupby("alert")["split"].nunique()
        multi = n_inc[n_inc > 1].index
        rows_multi = a["alert"].isin(multi).mean()
        test_alerts = a.loc[a.split == "test", "alert"].unique()
        both = n_split[n_split > 1].index
        res = {"alerts": int(len(n_inc)), "alerts_in_more_than_one_incident": int(len(multi)),
               "share_alerts_multi_incident": float((n_inc > 1).mean()),
               "share_rows_of_multi_incident_alerts": float(rows_multi),
               "test_alerts_also_in_train_file": float(np.isin(test_alerts, both).mean()),
               "share_of_those_that_are_multi_incident": float(np.isin(both, multi).mean()) if len(both) else np.nan,
               "max_incidents_per_alert": int(n_inc.max())}
        for k, v in res.items():
            print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v:,}")
        pd.DataFrame([res]).to_csv(out / "alert_check.csv", index=False)

    runs, cis, pairs = [], [], []

    # ---------- B. strict temporal
    if "B" not in skip:
        ts = df["Timestamp"].values
        cut = pd.Timestamp(args.cut, tz="UTC").to_datetime64()
        first = pd.Series(ts).groupby(inc_key).transform("min").values
        last = pd.Series(ts).groupby(inc_key).transform("max").values
        tr = np.where(last < cut)[0]
        te = np.where(first >= cut)[0]
        print(f"\n== B. STRICT TEMPORAL: train {len(tr):,} rows (all before cut), test {len(te):,} rows "
              f"(incidents starting after cut), dropped {len(df)-len(tr)-len(te):,} rows of spanning incidents; "
              f"test base rate {y[te].mean():.3f}", flush=True)
        r, c, p = run_split(df, y, org, inc_key, tr, te, ["org_prior", "content", "clean", "all"],
                            0, args.max_train, args.n_boot, "strict_temporal", 0)
        runs += r; cis += c; pairs += p
        pd.DataFrame(runs).to_csv(out / "runs.csv", index=False)
        print(f"  [{time.time()-t0:.0f}s]", flush=True)

    # ---------- C. IncidentId ablation on unseen organizations
    if "C" not in skip:
        print("\n== C. IDENTIFIER ABLATION ON UNSEEN ORGANIZATIONS")
        orgs = np.unique(org)
        for sd in [0, 1, 2]:
            held = np.random.default_rng(sd).choice(orgs, int(0.3 * len(orgs)), replace=False)
            m = np.isin(org, held)
            tr, te = np.where(~m)[0], np.where(m)[0]
            r, c, p = run_split(df, y, org, inc_key, tr, te, ["content", "all", "all_no_incidentid"],
                                sd, args.max_train, args.n_boot, "org_holdout", sd)
            runs += r; cis += c; pairs += p
            pd.DataFrame(runs).to_csv(out / "runs.csv", index=False)
            print(f"  [{time.time()-t0:.0f}s]", flush=True)

    pd.DataFrame(cis).to_csv(out / "incident_ci.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "paired.csv", index=False)
    if cis:
        print("\n##### INCIDENT-LEVEL ROC-AUC, 95% CI")
        print(pd.DataFrame(cis)[["protocol", "draw", "level", "features", "roc", "roc_lo", "roc_hi"]]
              .round(3).to_string(index=False))
        print("\n##### PAIRED: feature set minus content, incident ROC-AUC")
        print(pd.DataFrame(pairs)[["protocol", "draw", "level", "a", "roc_diff", "lo", "hi", "p_two_sided"]]
              .round(4).to_string(index=False))
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
