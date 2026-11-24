"""Step 3 - leakage / shortcut audit (contribution #1 of the paper).

Run on two tasks:
  mal_vs_benign : malicious vs benign (the "obvious" task most papers would run)
  escalation    : among rule-matched events, malicious (escalated to an incident)
                  vs suspicious (matched a rule, not escalated) - alert triage

A. Label-derived columns: how well does each one ALONE predict the label?
B. Sampling-artifact shortcut: local row density (rows per org-minute) only.
C. Log-source shortcut: a model that sees ONLY message_type + stream_name.
   EDA showed benign network-flow / firewall rows are absent, so on
   mal_vs_benign this alone should be near perfect.
D. Random vs incident-grouped vs temporal split, same model and features."""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from common import (load_signals, make_target, evaluate, split_random,
                    split_incident_group, split_temporal)
from config import LEAKY_COLS, RESULTS_DIR, TIME_COL, ORG_COL, SEED, DATA_DIR
from features import TabularFeaturizer


def audit(df_all, task, counts=None):
    df, y = make_target(df_all, task)
    print(f"\n### task={task}  rows={len(df)}  positives={y.sum()}")
    rows = []
    if y.min() == y.max():
        print("  skipped: only one class present in this data")
        return pd.DataFrame()

    # A. single-column leakage
    for c in LEAKY_COLS:
        if c not in df or c == "label_binary":
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            score = s.fillna(0).to_numpy(dtype=float)
        else:
            # in-sample target rate per value: an upper bound of what the column leaks
            score = pd.Series(y).groupby(s.astype(str).values).transform("mean").to_numpy()
        rows.append({"test": "A_single_column", "column": c, **evaluate(y, score)})

    tr, te = split_temporal(df, y)

    # B. density shortcut
    minute = (df[TIME_COL].values // 60).astype(np.int64)
    if counts is not None:
        # honest density: all events (benign included) in that org-minute, full capture
        tot = counts.groupby(["org_id", "minute"])["n"].sum()
        key = pd.MultiIndex.from_arrays([df[ORG_COL].values, minute])
        dens = tot.reindex(key).fillna(0).values
    else:
        dens = pd.Series(minute).groupby([df[ORG_COL].values, minute]).transform("size").values
    Xd = np.c_[np.log1p(dens)]
    if y[tr].min() != y[tr].max():
        m = LogisticRegression(max_iter=1000).fit(Xd[tr], y[tr])
        rows.append({"test": "B_density_only_temporal", "column": "rows_per_org_minute",
                     **evaluate(y[te], m.predict_proba(Xd[te])[:, 1])})

    # C. log-source shortcut
    src = pd.get_dummies(df[["message_type", "stream_name"]].astype(str), sparse=False)
    if y[tr].min() != y[tr].max():
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(src.values[tr], y[tr])
        rows.append({"test": "C_source_only_temporal", "column": "message_type+stream_name",
                     **evaluate(y[te], m.predict_proba(src.values[te])[:, 1])})

    # D. split-protocol inflation
    for name, fn in [("random_row", split_random), ("incident_group", split_incident_group),
                     ("temporal", split_temporal)]:
        tr2, te2 = fn(df, y)
        if y[tr2].min() == y[tr2].max():
            continue
        fz = TabularFeaturizer().fit(df.iloc[tr2])
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
        clf.fit(fz.transform(df.iloc[tr2]), y[tr2])
        s = clf.predict_proba(fz.transform(df.iloc[te2]))[:, 1]
        rows.append({"test": "D_split_protocol", "column": name, **evaluate(y[te2], s)})

    res = pd.DataFrame(rows)
    res.insert(0, "task", task)
    cols = ["task", "test", "column", "pr_auc", "roc_auc", "recall@fpr1%", "n_test", "n_pos"]
    print(res[[c for c in cols if c in res]].to_string())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=600_000,
                    help="rows to keep (all malicious+suspicious kept); 0 = all")
    ap.add_argument("--data", choices=["small", "100m"], default="small",
                    help="100m = rule-hit rows extracted by 05_extract_100m.py")
    args = ap.parse_args()
    out = RESULTS_DIR / "leakage"
    out.mkdir(parents=True, exist_ok=True)
    counts = None
    if args.data == "100m":
        df = load_signals(DATA_DIR / "100m" / "rulehits.parquet", sample=args.sample or None)
        counts = pd.read_parquet(DATA_DIR / "100m" / "minute_counts.parquet")
    else:
        df = load_signals(sample=args.sample or None)
    res = pd.concat([audit(df, "mal_vs_benign", counts), audit(df, "escalation", counts)])
    fn = out / f"leakage_audit_{args.data}.csv"
    res.to_csv(fn, index=False)
    print("\nsaved ->", fn)


if __name__ == "__main__":
    main()
