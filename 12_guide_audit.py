"""Step 12 - GUIDE: label checks, leakage audit and protocol comparison.

Task (same question as Precinct6 escalation): an analyst graded the incident
TruePositive (escalate) vs FalsePositive/BenignPositive (do not). Unit = evidence row;
the grade of a row is its incident's grade. Rows without a grade are dropped.

A. label checks      - grade constant within (OrgId, IncidentId)? who are the ungraded rows?
B. leakage audit     - every column alone: value -> TP rate fitted on train, scored on test
                       (PR-AUC under the official split). Identifiers that should carry
                       no signal tell us how much a split lets the model memorise.
C. protocols         - one LightGBM on the same features under
                       official (Microsoft's 70/30, grouped by incident, random in time)
                       random_row  (all rows pooled, random 70/30)
                       temporal    (train before --cut, test from --cut on)
                       org_holdout (30% of organisations unseen in training)
                       with two feature sets: 'clean' (no identifiers, no post-triage
                       columns) and 'all'.

Usage:  python 12_guide_audit.py --dir guide
        python 12_guide_audit.py --dir guide --max-train 2000000   (less RAM)
Output: results/guide_audit/*.csv
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from config import RESULTS_DIR

ID_COLS = ["Id", "IncidentId", "AlertId"]            # row/incident/alert identifiers
POST_TRIAGE = ["ActionGrouped", "ActionGranular"]    # response actions, taken after triage
ENTITY_IDS = ["DeviceId", "Sha256", "IpAddress", "Url", "AccountSid", "AccountUpn",
              "AccountObjectId", "AccountName", "DeviceName", "NetworkMessageId",
              "EmailClusterId", "RegistryKey", "RegistryValueName", "RegistryValueData",
              "ApplicationId", "ApplicationName", "OAuthApplicationId", "FileName",
              "FolderPath", "ResourceIdName"]
CONTEXT = ["DetectorId", "AlertTitle", "Category", "MitreTechniques", "EntityType",
           "EvidenceRole", "ThreatFamily", "ResourceType", "Roles", "OSFamily", "OSVersion",
           "AntispamDirection", "SuspicionLevel", "LastVerdict", "CountryCode", "State", "City"]
CLEAN = CONTEXT + ENTITY_IDS + ["hour", "n_rows_alert"]
ALL = CLEAN + ["OrgId"] + ID_COLS + POST_TRIAGE
TOPK = 255


def recall_at_fpr(y, s, fpr_max=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= fpr_max
    return float(tpr[ok].max()) if ok.any() else 0.0


def metrics(y, s):
    return {"n_test": int(len(y)), "base_rate": float(y.mean()),
            "pr_auc": float(average_precision_score(y, s)),
            "roc_auc": float(roc_auc_score(y, s)),
            "rec_at_1fpr": recall_at_fpr(y, s, 0.01)}


STR_COLS = ["Category", "MitreTechniques", "IncidentGrade", "ActionGrouped", "ActionGranular",
            "EntityType", "EvidenceRole", "ThreatFamily", "ResourceType", "Roles",
            "AntispamDirection", "SuspicionLevel", "LastVerdict"]


def load(path, split, chunksize=1_500_000):
    """Chunked read: strings -> category per chunk, merged with union_categoricals (fits 12 GB RAM)."""
    from pandas.api.types import union_categoricals
    parts = []
    for ch in pd.read_csv(path, chunksize=chunksize, low_memory=False,
                          dtype={c: "category" for c in STR_COLS}):
        ch = ch.drop(columns=[c for c in ["Usage"] if c in ch])
        ch["Timestamp"] = pd.to_datetime(ch["Timestamp"], utc=True, errors="coerce")
        for c in ch.columns:
            if ch[c].dtype == "int64":
                ch[c] = ch[c].astype("int64" if c == "Id" else "int32")
            elif ch[c].dtype == "float64":
                ch[c] = ch[c].astype("float32")
        parts.append(ch)
        print(f"  {path.name}: {sum(len(p) for p in parts):,} rows", flush=True)
    df = pd.DataFrame({c: (union_categoricals([p[c] for p in parts], ignore_order=True)
                           if c in STR_COLS else np.concatenate([p[c].values for p in parts]))
                       for c in parts[0].columns})
    del parts
    df["split"] = split
    return df


def encode(tr, te, cols):
    """Top-K vocabulary from train only; everything else (and unseen) -> K ('other')."""
    Xtr, Xte = pd.DataFrame(index=tr.index), pd.DataFrame(index=te.index)
    for c in cols:
        if c in ("hour", "n_rows_alert"):
            Xtr[c], Xte[c] = tr[c].values, te[c].values
            continue
        a, b = _keys(tr[c]), _keys(te[c])
        top = pd.Index(pd.Series(a).value_counts().index[:TOPK])
        ia, ib = top.get_indexer(a), top.get_indexer(b)
        Xtr[c] = np.where(ia < 0, TOPK, ia).astype("int16")
        Xte[c] = np.where(ib < 0, TOPK, ib).astype("int16")
    return Xtr, Xte


def _keys(col):
    """Fast hashable keys: numbers stay numbers (NaN -> -1), categories -> str."""
    if pd.api.types.is_numeric_dtype(col):
        return col.fillna(-1).to_numpy()
    return col.astype(object).where(col.notna(), "NA").astype(str).to_numpy()


def single_feature_audit(tr, te, ytr, yte, cols):
    rows = []
    prior = ytr.mean()
    for c in cols:
        a, b = _keys(tr[c]), _keys(te[c])
        rate = pd.Series(ytr).groupby(a).agg(["sum", "count"])
        enc = (rate["sum"] + 10 * prior) / (rate["count"] + 10)     # smoothed TP rate
        mapped = pd.Series(b).map(enc)
        seen = float(mapped.notna().mean())                         # value occurred in train
        s = mapped.fillna(prior).values
        rows.append({"column": c, "pr_auc": float(average_precision_score(yte, s)),
                     "roc_auc": float(roc_auc_score(yte, s)),
                     "test_values_seen_in_train": seen,
                     "n_values_train": int(len(rate))})
        print(f"  {c:20s} PR-AUC={rows[-1]['pr_auc']:.3f}  seen={seen:.2f}", flush=True)
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False)


def fit_score(tr, te, ytr, yte, cols, seed, max_train):
    if max_train and len(tr) > max_train:
        keep = np.random.default_rng(seed).choice(len(tr), max_train, replace=False)
        tr, ytr = tr.iloc[np.sort(keep)], ytr[np.sort(keep)]
    Xtr, Xte = encode(tr, te, cols)
    cat = [c for c in cols if c not in ("hour", "n_rows_alert")]
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           max_cat_to_onehot=8, random_state=seed, verbose=-1)
    m.fit(Xtr, ytr, categorical_feature=cat)
    return m.predict_proba(Xte)[:, 1]


def load_all(dirpath):
    from pandas.api.types import union_categoricals
    files = sorted(Path(dirpath).rglob("*.csv"))
    tr_path = [f for f in files if "train" in f.name.lower()][0]
    te_path = [f for f in files if f.name.lower() == "guide_test.csv"][0]
    print("loading ...")
    a, b = load(tr_path, "train"), load(te_path, "test")
    df = pd.DataFrame({c: (union_categoricals([a[c], b[c]], ignore_order=True) if c in STR_COLS
                           else np.concatenate([a[c].values, b[c].values])) for c in a.columns})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    return df


def add_derived(df):
    """Drop ungraded rows; add hour and evidence rows per alert; return df, y."""
    df = df[df["IncidentGrade"].notna()].reset_index(drop=True)
    df["hour"] = df["Timestamp"].dt.hour.astype("int8")
    df["n_rows_alert"] = df.groupby(["OrgId", "AlertId"])["Id"].transform("size").astype("int32")
    y = (df["IncidentGrade"] == "TruePositive").values.astype("int8")
    return df, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="guide")
    ap.add_argument("--cut", default="2024-06-10", help="temporal protocol: test from this day")
    ap.add_argument("--max-train", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", default="", help="comma list of sections to skip: A,B,C")
    args = ap.parse_args()
    out = RESULTS_DIR / "guide_audit"
    out.mkdir(parents=True, exist_ok=True)
    skip = set(args.skip.split(","))
    t0 = time.time()

    df = load_all(args.dir)

    # ---------- A. label checks
    if "A" not in skip:
        print("\n== A. LABEL CHECKS")
        key = ["split", "OrgId", "IncidentId"]
        g = df.dropna(subset=["IncidentGrade"]).groupby(key, observed=True)["IncidentGrade"].nunique()
        print(f"(OrgId, IncidentId) with >1 grade: {(g > 1).mean():.4%} of {len(g):,} incidents")
        ung = df["IncidentGrade"].isna()
        act = df["ActionGrouped"].notna()
        print(f"ungraded rows: {ung.mean():.4%}; rows with a response action: {act.mean():.4%}")
        print(pd.crosstab(act.rename("has_action"), df["IncidentGrade"].astype(str)).to_string())
        pd.DataFrame([{"multi_grade_share": float((g > 1).mean()), "incidents": len(g),
                       "ungraded_share": float(ung.mean()), "action_share": float(act.mean()),
                       "ungraded_given_action": float(ung[act].mean()) if act.any() else np.nan}]
                     ).to_csv(out / "A_label_checks.csv", index=False)

    df, y = add_derived(df)
    print(f"\nlabelled rows: {len(df):,}; TP share: {y.mean():.4f}  [{time.time()-t0:.0f}s]")

    off_tr = np.where(df["split"].values == "train")[0]
    off_te = np.where(df["split"].values == "test")[0]

    # ---------- B. single-column leakage audit (official split)
    if "B" not in skip:
        print("\n== B. SINGLE-COLUMN AUDIT (official split)")
        aud = single_feature_audit(df.iloc[off_tr], df.iloc[off_te], y[off_tr], y[off_te], ALL)
        aud.to_csv(out / "B_single_column.csv", index=False)

    # ---------- C. protocols
    if "C" not in skip:
        print("\n== C. PROTOCOLS")
        rng = np.random.default_rng(args.seed)
        n = len(df)
        perm = rng.permutation(n)
        rnd_tr, rnd_te = np.sort(perm[: int(0.7 * n)]), np.sort(perm[int(0.7 * n):])
        cut = pd.Timestamp(args.cut, tz="UTC")
        tmp_tr = np.where(df["Timestamp"].values < cut.to_datetime64())[0]
        tmp_te = np.where(df["Timestamp"].values >= cut.to_datetime64())[0]
        orgs = df["OrgId"].unique()
        held = set(rng.choice(orgs, int(0.3 * len(orgs)), replace=False))
        is_held = df["OrgId"].isin(held).values
        org_tr, org_te = np.where(~is_held)[0], np.where(is_held)[0]
        protos = {"official": (off_tr, off_te), "random_row": (rnd_tr, rnd_te),
                  "temporal": (tmp_tr, tmp_te), "org_holdout": (org_tr, org_te)}
        rows = []
        for name, (a, b) in protos.items():
            for fs_name, cols in [("clean", CLEAN), ("all", ALL)]:
                s = fit_score(df.iloc[a], df.iloc[b], y[a], y[b], cols, args.seed, args.max_train)
                r = {"protocol": name, "features": fs_name, "n_train": len(a), **metrics(y[b], s)}
                rows.append(r)
                print(f"  {name:12s} {fs_name:5s} PR-AUC={r['pr_auc']:.3f} ROC={r['roc_auc']:.3f} "
                      f"R@1%={r['rec_at_1fpr']:.3f} base={r['base_rate']:.3f}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)
                pd.DataFrame(rows).to_csv(out / "C_protocols.csv", index=False)

    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
