"""Step 11 - first look at Microsoft GUIDE (Kaggle: Microsoft/microsoft-security-incident-prediction).

Answers, before any modelling:
  1. schema: columns of the train and test files
  2. labels: IncidentGrade distribution at evidence, alert and incident level;
     is the grade constant within an incident?
  3. the official 70/30 split: is it temporal, random, or grouped? We check
     - time ranges of train and test, and how much they overlap
     - share of test IncidentIds / AlertIds / OrgIds that also occur in train
  4. organizations: how many, how skewed, how many have each grade

Reads only a few columns, in chunks, so it fits in Colab's free RAM.

Usage: python 11_guide_inspect.py --dir guide
Output: results/guide_inspect/*.csv and a printed summary
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS_DIR

WANT = ["Id", "OrgId", "IncidentId", "AlertId", "Timestamp", "DetectorId",
        "Category", "IncidentGrade", "EntityType", "EvidenceRole"]


def find(dirpath, word):
    files = sorted(Path(dirpath).rglob("*.csv"))
    hit = [f for f in files if word.lower() in f.name.lower()]
    if not hit:
        raise SystemExit(f"no CSV with '{word}' in its name under {dirpath}: {[f.name for f in files]}")
    return hit[0]


def load(path, chunksize=2_000_000):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    cols = [c for c in WANT if c in header]
    parts = []
    for ch in pd.read_csv(path, usecols=cols, chunksize=chunksize, low_memory=False):
        for c in cols:
            if c != "Timestamp":
                ch[c] = ch[c].astype("category") if ch[c].dtype == object else ch[c]
        parts.append(ch)
        print(f"  read {sum(len(p) for p in parts):,} rows of {path.name}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    return df, header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="guide")
    args = ap.parse_args()
    out = RESULTS_DIR / "guide_inspect"
    out.mkdir(parents=True, exist_ok=True)

    tr_path, te_path = find(args.dir, "train"), find(args.dir, "test")
    print("train file:", tr_path, "| test file:", te_path)
    tr, h_tr = load(tr_path)
    te, h_te = load(te_path)

    print("\n== 1. SCHEMA")
    print(f"train: {len(h_tr)} columns:", h_tr)
    print(f"test only: {sorted(set(h_te) - set(h_tr))} | train only: {sorted(set(h_tr) - set(h_te))}")
    pd.Series(h_tr).to_csv(out / "columns_train.csv", index=False)

    print("\n== 2. LABELS (IncidentGrade)")
    rows = []
    for name, d in [("train", tr), ("test", te)]:
        ev = d["IncidentGrade"].value_counts(dropna=False)
        inc = d.groupby("IncidentId", observed=True)["IncidentGrade"]
        n_grades = inc.nunique()
        inc_grade = inc.agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
        print(f"\n{name}: {len(d):,} evidence rows, {d['AlertId'].nunique():,} alerts, "
              f"{d['IncidentId'].nunique():,} incidents, {d['OrgId'].nunique():,} orgs")
        print("  evidence-level grade:\n", (ev / ev.sum()).round(4).to_string())
        print("  incident-level grade:\n", inc_grade.value_counts(normalize=True).round(4).to_string())
        print(f"  incidents with more than one grade: {(n_grades > 1).mean():.4%}")
        rows.append({"split": name, "evidence_rows": len(d), "alerts": d["AlertId"].nunique(),
                     "incidents": d["IncidentId"].nunique(), "orgs": d["OrgId"].nunique(),
                     "grade_missing_share": float(d["IncidentGrade"].isna().mean()),
                     "incidents_multi_grade": float((n_grades > 1).mean()),
                     **{f"evidence_{k}": float(v) for k, v in (ev / ev.sum()).items()}})
    pd.DataFrame(rows).to_csv(out / "labels.csv", index=False)

    print("\n== 3. OFFICIAL SPLIT: temporal, random or grouped?")
    t_tr, t_te = tr["Timestamp"], te["Timestamp"]
    print(f"train time: {t_tr.min()} -> {t_tr.max()}")
    print(f"test  time: {t_te.min()} -> {t_te.max()}")
    in_range = ((t_te >= t_tr.min()) & (t_te <= t_tr.max())).mean()
    after = (t_te > t_tr.max()).mean()
    ov = {}
    for key in ["IncidentId", "AlertId", "OrgId", "DetectorId"]:
        if key in tr and key in te:
            a = set(pd.unique(tr[key].astype(str)))
            b = pd.unique(te[key].astype(str))
            ov[key] = float(np.mean([x in a for x in b]))
    # incident ids are often only unique within an org: check the (OrgId, IncidentId) pair too
    pair_tr = set(zip(tr["OrgId"].astype(str), tr["IncidentId"].astype(str)))
    pair_te = pd.unique(pd.Series(list(zip(te["OrgId"].astype(str), te["IncidentId"].astype(str)))))
    ov["(OrgId, IncidentId)"] = float(np.mean([x in pair_tr for x in pair_te]))
    print(f"share of test rows inside the train time range: {in_range:.2%}; after it: {after:.2%}")
    for k, v in ov.items():
        print(f"share of test {k} values also present in train: {v:.2%}")
    day_tr = t_tr.dt.floor("D").value_counts().sort_index()
    day_te = t_te.dt.floor("D").value_counts().sort_index()
    days = pd.concat([day_tr.rename("train"), day_te.rename("test")], axis=1).fillna(0).astype(int)
    days["test_share"] = (days.test / (days.train + days.test)).round(3)
    print("\nrows per day (train, test, test share):\n", days.tail(40).to_string())
    days.to_csv(out / "rows_per_day.csv")
    pd.DataFrame([{"test_in_train_time_range": in_range, "test_after_train": after,
                   **{f"overlap_{k}": v for k, v in ov.items()}}]).to_csv(out / "split_overlap.csv", index=False)

    print("\n== 4. ORGANIZATIONS")
    org = tr.groupby("OrgId", observed=True).agg(rows=("Id", "size") if "Id" in tr else ("AlertId", "size"),
                                                 incidents=("IncidentId", "nunique"))
    org = org.sort_values("rows", ascending=False)
    share_top10 = org.rows.head(10).sum() / org.rows.sum()
    print(f"orgs in train: {len(org):,}; top-10 orgs hold {share_top10:.1%} of rows; "
          f"median incidents per org: {org.incidents.median():.0f}")
    org.to_csv(out / "orgs_train.csv")

    verdict = ("RANDOM (same incidents on both sides)" if ov.get("(OrgId, IncidentId)", 0) > 0.2
               else "TEMPORAL" if after > 0.9 else "GROUPED or mixed: inspect the tables")
    print(f"\n##### VERDICT on the official split: {verdict}")
    print("saved ->", out)


if __name__ == "__main__":
    main()
