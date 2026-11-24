"""Step 4 - baseline ladder for the escalation task under explicit protocols.

Protocols (06_diagnose_temporal.py showed one global time cut mixes two shifts:
attack waves come from different log sources, so we separate them):
  naive_global_temporal : one 70/30 time cut over all sources  (kept to show the failure)
  incident_grouped      : random split, incidents kept whole    (optimistic reference)
  asa_temporal          : cisco_asa only, 70/30 time cut, 1 h gap
  vpc_wave              : aws_vpc_flow_log only, train before --vpc-cut, test after
  asa_to_vpc / vpc_to_asa : train on one source, test on the other (cross-source)

Feature sets (models): logreg/lgbm on tabular, on tabular+context, on text,
and lgbm on everything; plus IsolationForest fitted on non-escalated rows.
Thresholds are tuned on the last 15% (in time) of TRAIN, never on test.

Usage: python 04_baselines.py --data 100m --sample 0
"""
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from common import (load_signals, make_target, evaluate, bootstrap_ci,
                    split_temporal, split_incident_group)
from config import RESULTS_DIR, SEED, TIME_COL, DATA_DIR
from features import TabularFeaturizer, context_features, text_features, combine

ASA, VPC = "cisco_asa", "aws_vpc_flow_log"


# ------------------------------------------------------------------ protocols
def _idx(mask):
    return np.where(mask)[0]


def protocols(df, y, vpc_cut):
    st = df["stream_name"].values
    t = df[TIME_COL].values
    P = {}
    P["naive_global_temporal"] = split_temporal(df, y)
    P["incident_grouped"] = split_incident_group(df, y)
    a = _idx(st == ASA)
    if len(a):
        tr, te = split_temporal(df.iloc[a].reset_index(drop=True), y[a])
        P["asa_temporal"] = (a[tr], a[te])
    v = _idx(st == VPC)
    if len(v):
        cut = pd.Timestamp(vpc_cut, tz="UTC").timestamp()
        P["vpc_wave"] = (v[t[v] < cut - 3600], v[t[v] >= cut])
    if len(a) and len(v):
        P["asa_to_vpc"] = (a, v)
        P["vpc_to_asa"] = (v, a)
    return P


# ------------------------------------------------------------------ helpers
def val_split(t_tr, frac=0.85):
    cut = np.quantile(t_tr, frac)
    return np.where(t_tr < cut)[0], np.where(t_tr >= cut)[0]


def best_thr(y, s):
    if y.min() == y.max():
        return 0.5
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / (p + r + 1e-12)
    return float(t[np.argmax(f[:-1])]) if len(t) else 0.5


def lgbm(seed, colsample=1.0):
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                              is_unbalance=True, colsample_bytree=colsample,
                              subsample=0.8, subsample_freq=1,
                              random_state=seed, verbose=-1)


MODELS = {
    # name: (feature set, factory(seed))
    "logreg_tab": ("tab", lambda s: LogisticRegression(max_iter=2000, class_weight="balanced")),
    "lgbm_tab": ("tab", lambda s: lgbm(s)),
    "logreg_tab+ctx": ("tabctx", lambda s: LogisticRegression(max_iter=2000, class_weight="balanced")),
    "lgbm_tab+ctx": ("tabctx", lambda s: lgbm(s)),
    "logreg_text": ("txt", lambda s: LogisticRegression(max_iter=2000, class_weight="balanced",
                                                        solver="liblinear")),
    "lgbm_all": ("all", lambda s: lgbm(s, 0.3)),
    "iforest_ctx": ("tabctx", None),        # fitted on non-escalated training events (uses labels to filter)
    "iforest_ctx_all": ("tabctx", None),    # fitted on ALL training events (no labels at all)
}


def run_protocol(df, y, CTX, TXT, tr, te, proto, seed=SEED, train_frac=1.0,
                 models=None, scores_out=None):
    """train_frac < 1 trains each model on a random subset of the training split
    (seeded), so even deterministic models show run-to-run variance."""
    rows = []
    if len(te) == 0 or y[te].sum() == 0 or y[tr].sum() == 0:
        print(f"  skip {proto}: no positives in train or test")
        return rows
    d_tr, d_te = df.iloc[tr], df.iloc[te]
    fit_i, val_i = val_split(d_tr[TIME_COL].values)
    if y[tr][fit_i].sum() == 0:            # validation slice took all positives
        fit_i, val_i = np.arange(len(tr)), np.arange(len(tr))
    if train_frac < 1.0:
        rng = np.random.RandomState(seed)
        fit_i = np.sort(rng.choice(fit_i, int(len(fit_i) * train_frac), replace=False))
    fz = TabularFeaturizer().fit(d_tr.iloc[fit_i])
    tab_tr, tab_te = fz.transform(d_tr).to_numpy(np.float32), fz.transform(d_te).to_numpy(np.float32)
    ctx_tr, ctx_te = CTX[tr], CTX[te]
    dense = {"tab": (tab_tr, tab_te),
             "tabctx": (np.hstack([tab_tr, ctx_tr]), np.hstack([tab_te, ctx_te]))}
    scalers = {k: StandardScaler().fit(v[0][fit_i]) for k, v in dense.items()}
    for name, (kind, make) in MODELS.items():
        if models and name not in models:
            continue
        if kind in dense:
            Xtr, Xte = dense[kind]
            if name.startswith(("logreg", "iforest")):
                Xtr, Xte = scalers[kind].transform(Xtr), scalers[kind].transform(Xte)
        elif kind == "txt":
            Xtr, Xte = TXT[tr], TXT[te]
        else:
            Xtr, Xte = combine(dense["tabctx"][0], TXT[tr]), combine(dense["tabctx"][1], TXT[te])
        ytr = y[tr]
        if name.startswith("iforest"):
            m = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
            m.fit(Xtr[fit_i] if name == "iforest_ctx_all" else Xtr[fit_i][ytr[fit_i] == 0])
            s_val, s_te = -m.score_samples(Xtr[val_i]), -m.score_samples(Xte)
        else:
            m = make(seed).fit(Xtr[fit_i], ytr[fit_i])
            if isinstance(m, LogisticRegression):
                # linear score, not predict_proba: the sigmoid saturates to exactly 1.0 for
                # test rows far outside the training range, creating ties that corrupt ranking
                s_val, s_te = m.decision_function(Xtr[val_i]), m.decision_function(Xte)
            else:
                s_val, s_te = m.predict_proba(Xtr[val_i])[:, 1], m.predict_proba(Xte)[:, 1]
        if scores_out is not None:
            scores_out[(proto, name, seed)] = s_te
        r = evaluate(y[te], s_te, thr=best_thr(ytr[val_i], s_val))
        r["top_tie_share"] = float((s_te == s_te.max()).mean())   # >0.001 signals saturation
        if "pr_auc" in r:
            r["pr_auc_ci95"] = bootstrap_ci(y[te], s_te, n=200)
            r["base_rate"] = float(y[te].mean())
            r["lift"] = r["pr_auc"] / r["base_rate"]
        rows.append({"protocol": proto, "model": name, "seed": seed, "n_train": len(fit_i),
                     "pos_train": int(ytr.sum()), **r})
        print(f"  {proto:22s} {name:15s} PR-AUC={r.get('pr_auc', np.nan):.3f} "
              f"(base {r.get('base_rate', np.nan):.3f}, lift {r.get('lift', np.nan):.1f}x) "
              f"ROC={r.get('roc_auc', np.nan):.3f} R@1%FPR={r.get('recall@fpr1%', np.nan):.3f}",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["small", "100m"], default="100m")
    ap.add_argument("--sample", type=int, default=0, help="0 = all rows")
    ap.add_argument("--task", default="escalation", choices=["escalation", "mal_vs_benign"])
    ap.add_argument("--vpc-cut", default="2024-07-28", help="vpc_wave: test starts this UTC date")
    ap.add_argument("--only", default="", help="comma list of protocols to run")
    args = ap.parse_args()

    path = DATA_DIR / "100m" / "rulehits.parquet" if args.data == "100m" else None
    df = load_signals(path, sample=args.sample or None)
    df, y = make_target(df, args.task)
    counts = None
    if args.data == "100m":
        counts = pd.read_parquet(DATA_DIR / "100m" / "minute_counts.parquet")
    print(f"rows={len(df):,} positives={y.sum():,}; computing context + text features...")
    CTX = context_features(df, counts).to_numpy(np.float32)
    TXT = text_features(df)

    rows = []
    P = protocols(df, y, args.vpc_cut)
    keep = [p for p in args.only.split(",") if p] or list(P)
    for name in keep:
        tr, te = P[name]
        print(f"\n== {name}: train {len(tr):,} ({y[tr].sum():,} pos)  test {len(te):,} ({y[te].sum():,} pos)")
        rows += run_protocol(df, y, CTX, TXT, tr, te, name)

    out = RESULTS_DIR / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    fn = out / f"baselines_{args.task}_{args.data}.csv"
    pd.DataFrame(rows).to_csv(fn, index=False)
    print("\nsaved ->", fn)


if __name__ == "__main__":
    main()
