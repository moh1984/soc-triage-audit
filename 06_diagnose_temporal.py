"""Step 6 - why does the temporal split collapse (ROC < 0.5) on the full capture?

Answers four questions before anything is claimed in the paper:
  Q1 where the 70% temporal cut falls (date), and what train/test positives look like
     by stream and org
  Q2 daily distribution of escalated (malicious) events by stream
  Q3 which features reverse direction between train and test (univariate ROC per split)
  Q4 per-stream performance of the temporal model, and a per-stream temporal split
Writes everything to results/diagnostics/ and prints a summary.

Usage: python 06_diagnose_temporal.py --data 100m
"""
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from common import load_signals, make_target, split_temporal
from config import DATA_DIR, RESULTS_DIR, TIME_COL, ORG_COL, SEED
from features import TabularFeaturizer


def safe_auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["small", "100m"], default="100m")
    args = ap.parse_args()
    out = RESULTS_DIR / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "100m" / "rulehits.parquet" if args.data == "100m" else None
    df, y = make_target(load_signals(path), "escalation")
    day = pd.to_datetime(df[TIME_COL], unit="s", utc=True).dt.date
    df["_day"] = day.astype(str)
    df["_y"] = y

    # ---- Q1 cut position
    tr, te = split_temporal(df, y)
    cut = pd.to_datetime(df[TIME_COL].iloc[te].min(), unit="s", utc=True)
    print(f"Q1  temporal cut: test starts {cut}  | train rows={len(tr):,} pos={y[tr].sum():,}"
          f" | test rows={len(te):,} pos={y[te].sum():,}")
    side = np.full(len(df), "gap")
    side[tr], side[te] = "train", "test"
    df["_side"] = side
    comp = (df[df._side != "gap"].groupby(["_side", "stream_name", ORG_COL])["_y"]
              .agg(rows="size", escalated="sum").reset_index())
    comp["rate_%"] = (100 * comp.escalated / comp.rows).round(2)
    comp = comp.sort_values(["_side", "escalated"], ascending=[False, False])
    comp.to_csv(out / "q1_split_composition.csv", index=False)
    print(comp[comp.escalated > 0].to_string(index=False))

    # ---- Q2 daily escalations by stream
    daily = pd.crosstab([df._day], [df.stream_name], values=df._y, aggfunc="sum").fillna(0)
    daily = daily.loc[:, daily.sum() > 0].astype(int)
    daily["ALL_escalated"] = daily.sum(axis=1)
    daily["ALL_rows"] = df.groupby("_day").size()
    daily.to_csv(out / "q2_daily_escalations_by_stream.csv")
    print("\nQ2  daily escalations by stream\n", daily.to_string())

    # ---- Q3 univariate direction flips
    fz = TabularFeaturizer().fit(df.iloc[tr])
    Xtr, Xte = fz.transform(df.iloc[tr]), fz.transform(df.iloc[te])
    rows = []
    for c in Xtr.columns:
        a_tr, a_te = safe_auc(y[tr], Xtr[c].values), safe_auc(y[te], Xte[c].values)
        if not (np.isnan(a_tr) or np.isnan(a_te)):
            rows.append({"feature": c, "auc_train": a_tr, "auc_test": a_te,
                         "flip": (a_tr - 0.5) * (a_te - 0.5) < 0,
                         "strength": abs(a_tr - 0.5) + abs(a_te - 0.5)})
    uni = pd.DataFrame(rows).sort_values("strength", ascending=False)
    uni.to_csv(out / "q3_univariate_flips.csv", index=False)
    print("\nQ3  strongest features (auc_train vs auc_test; flip = direction reversed)\n",
          uni.head(20).round(3).to_string(index=False))

    # ---- Q4a temporal model, per-stream evaluation + importances
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                             is_unbalance=True, random_state=SEED, verbose=-1)
    clf.fit(Xtr, y[tr])
    s = clf.predict_proba(Xte)[:, 1]
    imp = pd.Series(clf.booster_.feature_importance("gain"), index=Xtr.columns)
    imp.sort_values(ascending=False).head(25).to_csv(out / "q4_importance_gain.csv")
    print("\nQ4a top-10 features by gain:\n", imp.sort_values(ascending=False).head(10).round(0).to_string())
    per = []
    te_df = df.iloc[te]
    for st, g in te_df.groupby("stream_name"):
        idx = np.where(te_df.stream_name.values == st)[0]
        yy = y[te][idx]
        if yy.sum() > 0:
            per.append({"stream": st, "test_rows": len(idx), "test_pos": int(yy.sum()),
                        "roc": safe_auc(yy, s[idx]), "pr_auc": average_precision_score(yy, s[idx]),
                        "base_rate": yy.mean()})
    per = pd.DataFrame(per)
    print("\nQ4a temporal model, evaluated within each stream:\n", per.round(3).to_string(index=False))

    # ---- Q4b temporal split WITHIN each stream (train and test on the same source)
    rows = []
    for st in df.stream_name.value_counts().index:
        g = df[df.stream_name == st]
        yg = g._y.values
        if yg.sum() < 50:
            continue
        trg, teg = split_temporal(g.reset_index(drop=True), yg)
        if yg[trg].sum() == 0 or yg[teg].sum() == 0:
            rows.append({"stream": st, "note": "no positives on one side of the cut",
                         "train_pos": int(yg[trg].sum()), "test_pos": int(yg[teg].sum())})
            continue
        gg = g.reset_index(drop=True)
        f2 = TabularFeaturizer().fit(gg.iloc[trg])
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               is_unbalance=True, random_state=SEED, verbose=-1)
        m.fit(f2.transform(gg.iloc[trg]), yg[trg])
        sg = m.predict_proba(f2.transform(gg.iloc[teg]))[:, 1]
        rows.append({"stream": st, "train_pos": int(yg[trg].sum()), "test_pos": int(yg[teg].sum()),
                     "roc": safe_auc(yg[teg], sg), "pr_auc": average_precision_score(yg[teg], sg),
                     "base_rate": yg[teg].mean()})
    within = pd.DataFrame(rows)
    within.to_csv(out / "q4b_within_stream_temporal.csv", index=False)
    print("\nQ4b temporal split inside each stream:\n", within.round(3).to_string(index=False))
    per.to_csv(out / "q4a_per_stream_eval.csv", index=False)
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
